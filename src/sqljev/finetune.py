"""Fine-tune Laya on labelled SQL rows, on one GPU (a free Colab T4 is enough), and publish the checkpoint.

    sqljev dataset "$DB_URL" "SELECT subject, body, team FROM tickets" --label team \\
        --choice "which team should handle this?" --test-fraction 0.2 -o tickets.jsonl
    sqljev finetune tickets.train.jsonl --out checkpoints/tickets --epochs 3
    sqljev eval tickets.test.jsonl --model checkpoints/tickets
    sqljev publish checkpoints/tickets --repo your-org/laya-tickets          # Hugging Face Hub

Training data is what `sqljev dataset` writes: one row per line with the state (the row, as the model sees it at
query time), the question built by core.laya_question and the expected answer, so the checkpoint learns exactly
the questions sqljev will ask it.

The training loop follows Laya's own fine-tuning notebook (Convai Innovations, Apache 2.0,
https://github.com/NandhaKishorM/laya/tree/main/notebooks): policy-gradient updates against a proper scoring
rule plus soft cross-entropy, then per-question-type temperature calibration on a held-out slice. It is
single-GPU here, with gradient checkpointing and mixed precision, so it fits a 16 GB T4.
"""
import json
import os
import random
import re
import time

from .core import JevError, check_question, laya_question, to_row_json

LAYA_BASE = "convaiinnovations/laya"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


# ---------------------------------------------------------------- examples

def examples_from_rows(rows, question, kind, options=None, label=None, columns=None, drop_nulls=True):
    """Labelled rows (dicts) -> training/eval records. The label column never reaches the state. For choice
    questions without options, the options are the distinct labels."""
    rows = [r for r in rows if r.get(label) is not None]
    if kind == "choice" and not options:
        options = sorted({str(r[label]) for r in rows})
    kind, opts = check_question(kind, options)
    q = {"q": laya_question(kind, question, opts)}
    out = []
    for r in rows:
        lab = r[label]
        if kind == "noul":
            expected = _truthy(lab)
        elif kind == "choice":
            expected = str(lab)
            if expected not in opts:
                continue
        else:
            expected = opts.index(str(lab)) if str(lab) in opts else int(lab)
        view = {k: v for k, v in r.items() if k != label and (not columns or k in columns)}
        out.append({"state": json.loads(to_row_json(view, drop_nulls)), "questions": q, "expected": {"q": expected}})
    return out


def split(records, test_fraction=0.2, seed=0):
    """Deterministic train/test split."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(records) * test_fraction)
    test = set(idx[:n_test])
    return [r for i, r in enumerate(records) if i not in test], [r for i, r in enumerate(records) if i in test]


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def _truthy(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise JevError("sqljev: label %r is not a boolean" % (v,))


def accuracy(records, **settings):
    """Share of expected answers the model gets right on `records` (a list or a JSONL path), with any engine
    settings (model=<checkpoint>, device=..., backend=...). Returns {"accuracy", "decisions", "seconds"}."""
    from .core import Jev
    records = read_jsonl(records) if isinstance(records, str) else records
    jev = Jev(**settings)
    groups = {}
    for rec in records:
        for qid, q in rec["questions"].items():
            crit = q.get("criteria")
            opts = list(crit) if crit else None
            groups.setdefault((q["type"], q["instructions"], json.dumps(opts)), []).append(
                (rec["state"], rec["expected"][qid]))
    t0, right, total = time.time(), 0, 0
    for (kind, instr, opts_json), items in groups.items():
        prefix = "Is it true that "
        query = instr[len(prefix):-1] if kind == "noul" and instr.startswith(prefix) and instr.endswith("?") else instr
        answers = jev.evaluate([s for s, _ in items], query, kind, json.loads(opts_json))
        for (_, exp), a in zip(items, answers):
            if kind == "noul":
                right += (a["noul"] >= 0.5) == bool(exp)
            elif kind == "choice":
                right += a["choice"] == exp
            else:
                right += round(a["score"]) == int(exp)
            total += 1
    return {"accuracy": round(right / total, 4) if total else 0.0, "decisions": total,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- training

def base_dir(base=LAYA_BASE):
    """Local directory of a Laya checkpoint: a path, a named base checkpoint, or a Hub id."""
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    sub = SUBFOLDERS.get(base, None)
    repo = LAYA_BASE if base in SUBFOLDERS else base
    d = snapshot_download(repo)
    try:
        from laya.agent import _fix_tokenizer_config
        _fix_tokenizer_config(d)
    except Exception:       # noqa: BLE001 -- older/newer laya: the files are already usable
        pass
    return os.path.join(d, sub) if sub else d


def _items(records, tok, cfg):
    from laya.common import QTYPES, build_sequence, render_options
    items, skipped = [], 0
    for rec in records:
        for qid, q in rec["questions"].items():
            t, crit = q["type"], q.get("criteria") or {}
            exp = rec["expected"][qid]
            if t == "noul":
                target = [0.0, 1.0] if exp else [1.0, 0.0]
            elif t == "choice":
                keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
                target = [float(k == exp) for k in keys]
            else:
                target = [float(i == int(exp)) for i in range(len(crit))]
            seq, markers = build_sequence(tok, rec["state"], {"t": t, "ins": q["instructions"], "crit": crit},
                                          cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options({"t": t, "crit": crit})) or sum(target) != 1:
                skipped += 1
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                          "label": target.index(1.0)})
    return items, skipped


def _collate(items, pad_id):
    import torch
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"])
    return ids, att, mpos, mmask, target, torch.tensor([it["qtype"] for it in items])


def _fit_temperature(pairs):
    import torch
    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def finetune(train, out_dir, base=LAYA_BASE, epochs=3, micro_batch=8, grad_accum=8, lr_encoder=2.5e-5,
             lr_head=1e-4, group_size=4, train_layers=None, device=None, seed=0, log=print):
    """Fine-tune a Laya checkpoint on `train` (records or a JSONL path) and save it to `out_dir` in Laya's
    format, loadable with laya.load(out_dir) and usable as SQLJEV_MODEL=out_dir.

    train_layers: train only the top N encoder layers plus the decision head, freezing the embeddings and the
    layers below (low-memory mode for GPUs under ~10 GB free). None trains everything."""
    import torch
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    records = read_jsonl(train) if isinstance(train, str) else list(train)
    if not records:
        raise JevError("sqljev: no training records")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src = base_dir(base)
    with open(os.path.join(src, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))
    items, skipped = _items(records, tok, cfg)
    if not items:
        raise JevError("sqljev: none of the records could be encoded for this checkpoint")
    rng = random.Random(seed)
    rng.shuffle(items)
    n_cal = min(400, len(items) // 10)
    calib, items = items[:n_cal], items[n_cal:]
    log("sqljev finetune: %d training items, %d held out for calibration%s, base %s, device %s"
        % (len(items), len(calib), ", %d skipped" % skipped if skipped else "", base, device))

    torch.manual_seed(seed)
    model = build_model(cfg, encoder_dir=os.path.join(src, "encoder"))
    model.load_state_dict(load_file(os.path.join(src, "model.safetensors")), strict=True)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device).train()
    if train_layers is not None:
        n_layers = 1 + max(int(m.group(1)) for n, _ in model.named_parameters()
                           for m in [re.match(r"encoder\.layers\.(\d+)\.", n)] if m)
        keep = n_layers - int(train_layers)
        for n, p in model.named_parameters():
            m = re.match(r"encoder\.layers\.(\d+)\.", n)
            if n.startswith("encoder.embeddings.") or (m and int(m.group(1)) < keep):
                p.requires_grad_(False)
        log("sqljev finetune: training the top %d of %d encoder layers and the head" % (min(int(train_layers), n_layers), n_layers))

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}], weight_decay=0.01)
    updates = max(1, -(-len(items) // (micro_batch * grad_accum)) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    t0 = time.time()
    for epoch in range(epochs):
        rng.shuffle(items)
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, epochs - 1))
        total, steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(items), micro_batch):
            ids, att, mpos, mmask, target, qtype = (x.to(device) for x in _collate(items[b:b + micro_batch], tok.pad_token_id))
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
                logits, act = model(ids, att, mpos, mmask, qtype)
            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True))
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (-(adv * logp).mean() + loss_ce) / grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            steps += 1
            total += loss.item() * grad_accum
            if steps % grad_accum == 0 or b + micro_batch >= len(items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc + head, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if steps % 100 == 0:
                log("  epoch %d/%d  step %d/%d  loss %.4f  %.0fs"
                    % (epoch + 1, epochs, steps, -(-len(items) // micro_batch), total / steps, time.time() - t0))
        log("  epoch %d/%d done, mean loss %.4f, %.0fs" % (epoch + 1, epochs, total / max(1, steps), time.time() - t0))

    # Calibrate one temperature per question type on items the run never trained on.
    model.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(calib), 16):
            chunk = calib[b:b + 16]
            ids, att, mpos, mmask, target, qtype = (x.to(device) for x in _collate(chunk, tok.pad_token_id))
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
                lg, _ = model(ids, att, mpos, mmask, qtype)
            lg = lg.float().cpu().numpy()
            preds += [(it["qtype"], lg[i, :len(it["markers"])].tolist(), it["target"]) for i, it in enumerate(chunk)]
    temps = list(cfg.get("temperature") or [1.2, 1.2, 1.2])
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        if len(sel) >= 10:
            temps[qt] = _fit_temperature(sel)

    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg.update(fine_tuned=True, model_name="laya-sqljev", temperature=temps)
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    questions = sorted({q["instructions"] for r in records for q in r["questions"].values()})
    with open(os.path.join(out_dir, "sqljev_finetune.json"), "w") as f:
        json.dump({"base": base, "records": len(records), "items": len(items), "epochs": epochs,
                   "questions": questions, "seconds": round(time.time() - t0, 1)}, f, indent=2)
    log("sqljev finetune: saved to %s (%.0fs)" % (out_dir, time.time() - t0))
    return out_dir


# ---------------------------------------------------------------- publishing

def publish(out_dir, repo_id, token=None, private=True, metrics=None):
    """Upload a fine-tuned checkpoint to the Hugging Face Hub with a model card. Every database then uses it
    with SQLJEV_MODEL=<repo_id>."""
    from huggingface_hub import HfApi
    meta_path = os.path.join(out_dir, "sqljev_finetune.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if metrics:
        meta["metrics"] = metrics
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    card = """---
license: apache-2.0
base_model: %s
tags: [laya, sqljev, sql, decision-model]
---

# %s

A [Laya](https://github.com/NandhaKishorM/laya) checkpoint fine-tuned with [sqljev](https://github.com/singhpratech/sqljev)
on labelled SQL rows.

Questions it was trained on:
%s

%s
Use it from any database sqljev supports:

```bash
SQLJEV_MODEL=%s sqljev gateway --host 0.0.0.0
```
""" % (meta.get("base", LAYA_BASE), repo_id.split("/")[-1],
       "\n".join("- %s" % q for q in meta.get("questions", [])),
       ("Held-out accuracy: %s\n" % json.dumps(metrics)) if metrics else "", repo_id)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=out_dir, commit_message="sqljev finetune")
    return "https://huggingface.co/" + repo_id
