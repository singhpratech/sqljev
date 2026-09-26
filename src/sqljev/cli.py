"""sql-jev-laya command line: plain-language questions over any SQLAlchemy database, the gateway, and the
fine-tuning loop (dataset -> fine-tune Laya -> eval).

  sql-jev-laya query URL "SELECT * FROM tickets" --where "the customer is angry" --limit 20
  sql-jev-laya query URL "SELECT * FROM tickets" --rank "the customer is angry" --limit 10
  sql-jev-laya query URL "SELECT * FROM tickets" --choice "which team?" --options billing,technical,sales
  sql-jev-laya materialize URL "SELECT id, body FROM tickets" --key id --prob "the customer is angry" --into t_angry
  sql-jev-laya dataset URL "SELECT * FROM tickets" --label team --choice "which team?" -o tickets.jsonl
  sql-jev-laya eval tickets.test.jsonl
  sql-jev-laya gateway --port 8765
"""
import argparse
import csv
import hashlib
import json
import sys
import time

from . import core
from .core import Jev, JevError, check_question, laya_question, to_row_json


# ---------------------------------------------------------------- database access (SQLAlchemy)

def _engine(url):
    try:
        import sqlalchemy
    except ImportError:
        raise JevError("sql-jev-laya: database access needs SQLAlchemy: pip install 'sql-jev-laya[db]'")
    return sqlalchemy.create_engine(url)


def stream_rows(url, sql, chunk=1000):
    """Rows as dicts, streamed with a server-side cursor where the driver supports it."""
    import sqlalchemy
    eng = _engine(url)
    with eng.connect() as conn:
        res = conn.execution_options(stream_results=True, yield_per=chunk).execute(sqlalchemy.text(sql))
        for m in res.mappings():
            yield dict(m)


def _model_view(columns, exclude=()):
    """The part of a row the model sees: --columns selects, the label column is always excluded."""
    cols = [c.strip() for c in columns.split(",")] if columns else None

    def view(row):
        return {k: v for k, v in row.items() if (cols is None or k in cols) and k not in exclude}
    return view


# ---------------------------------------------------------------- question selection shared by commands

def _question(a, need=True):
    picked = [(k, getattr(a, k)) for k in ("where", "rank", "prob", "choice", "score", "noul")
              if getattr(a, k, None)]
    if len(picked) != 1:
        if not need:
            return None
        raise JevError("sql-jev-laya: give exactly one of --where / --rank / --prob / --choice / --score")
    mode, text = picked[0]
    kind = {"choice": "choice", "score": "score"}.get(mode, "noul")
    opts = a.options if kind == "choice" else getattr(a, "levels", None) if kind == "score" else None
    if kind == "choice" and opts is None and mode == "choice" and getattr(a, "label", None):
        return mode, text, kind, None          # dataset: options come from the label column
    kind, opts = check_question(kind, opts)
    return mode, text, kind, opts


def _settings(a):
    return {k: getattr(a, k) for k in ("backend", "api_url", "model", "device", "batch_size", "concurrency",
                                       "max_rows", "max_len") if getattr(a, k, None) is not None} | \
        ({"drop_nulls": False} if getattr(a, "keep_nulls", False) else {})


def _value(kind, a, opts):
    if a is None:
        return None
    if kind == "noul":
        return a["noul"]
    if kind == "choice":
        return a["choice"]
    return a["score"]


# ---------------------------------------------------------------- output

def _fmt(v, width=40):
    if isinstance(v, float):
        return "%.3f" % v
    s = "" if v is None else str(v)
    s = s.replace("\n", " ")
    return s if len(s) <= width else s[:width - 1] + "…"


def write_rows(rows, fmt, out=None):
    out = out or sys.stdout
    rows = list(rows)
    if not rows:
        print("(0 rows)", file=sys.stderr)
        return
    cols = list(rows[0].keys())
    if fmt == "jsonl":
        for r in rows:
            out.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    elif fmt == "csv":
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    else:
        cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
        widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
        out.write(" | ".join(c.ljust(w) for c, w in zip(cols, widths)) + "\n")
        out.write("-+-".join("-" * w for w in widths) + "\n")
        for row in cells:
            out.write(" | ".join(v.ljust(w) for v, w in zip(row, widths)) + "\n")
        print("(%d row%s)" % (len(rows), "" if len(rows) == 1 else "s"), file=sys.stderr)


# ---------------------------------------------------------------- commands

def cmd_query(a):
    mode, text, kind, opts = _question(a)
    jev = Jev(**_settings(a))
    view = _model_view(a.columns)
    rows = stream_rows(a.url, a.sql)
    col = {"where": "jev_prob", "rank": "jev_prob", "prob": "jev_prob", "choice": "jev_choice",
           "score": "jev_score"}[mode]
    t0 = time.time()
    if mode == "where":
        # Streaming: rows are judged in order with a bounded read-ahead, so --limit stops the scan early.
        threshold = a.threshold if a.threshold is not None else jev.cfg["threshold"]

        def passing():
            n = 0
            for row, ans in jev.evaluate_iter(rows, text, kind, opts, key=view):
                p = ans["noul"]
                if p >= threshold:
                    yield {**row, col: p}
                    n += 1
                    if a.limit and n >= a.limit:
                        return
        write_rows(passing(), a.format)
    else:
        all_rows = list(rows)
        answers = jev.evaluate([view(r) for r in all_rows], text, kind, opts)
        out = [{**r, col: _value(kind, ans, opts)} for r, ans in zip(all_rows, answers)]
        if mode == "rank" or (mode == "score" and a.sort):
            out.sort(key=lambda r: -(r[col] or 0))
        write_rows(out[:a.limit] if a.limit else out, a.format)
    _report(jev, t0, a)


def cmd_materialize(a):
    """Judge every row and write (key, value, answer) into a table, joinable in any database."""
    import sqlalchemy as sa
    mode, text, kind, opts = _question(a)
    jev = Jev(**_settings(a))
    view = _model_view(a.columns)
    rows = list(stream_rows(a.url, a.sql))
    if rows and a.key not in rows[0]:
        raise JevError("sql-jev-laya: key column %r is not in the query result" % a.key)
    t0 = time.time()
    answers = jev.evaluate([view(r) for r in rows], text, kind, opts)
    col = "jev_" + ("choice" if kind == "choice" else "score" if kind == "score" else "prob")
    key_type = sa.BigInteger() if rows and isinstance(rows[0][a.key], int) else sa.String(255)
    eng = _engine(a.url)
    meta = sa.MetaData()
    schema, _, name = a.into.rpartition(".")
    table = sa.Table(name, meta, sa.Column(a.key, key_type, primary_key=True),
                     sa.Column(col, sa.String(255) if kind == "choice" else sa.Float()),
                     sa.Column("jev_answer", sa.Text()), schema=schema or None)
    with eng.begin() as conn:
        if a.replace:
            table.drop(conn, checkfirst=True)
        table.create(conn, checkfirst=True)
        payload = [{a.key: r[a.key], col: _value(kind, ans, opts), "jev_answer": json.dumps(ans)}
                   for r, ans in zip(rows, answers)]
        for i in range(0, len(payload), 1000):
            conn.execute(table.insert(), payload[i:i + 1000])
    print("sql-jev-laya: wrote %d rows to %s (%s)" % (len(rows), a.into, col), file=sys.stderr)
    _report(jev, t0, a)


def _truthy(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise JevError("sql-jev-laya: label %r is not a boolean" % (v,))


def cmd_dataset(a):
    """Labelled SQL rows -> Laya training/eval data, using exactly the state and question sql-jev-laya asks at
    query time, so a checkpoint fine-tuned on it sees what it will be asked."""
    mode, text, kind, opts = _question(a)
    view = _model_view(a.columns, exclude={a.label})
    rows = list(stream_rows(a.url, a.sql))
    if not rows:
        raise JevError("sql-jev-laya: the query returned no rows")
    if a.label not in rows[0]:
        raise JevError("sql-jev-laya: label column %r is not in the query result" % a.label)
    if kind == "choice" and opts is None:
        opts = sorted({str(r[a.label]) for r in rows if r[a.label] is not None})
        kind, opts = check_question(kind, opts)
    q = {"q": laya_question(kind, text, opts)}
    base = a.output[:-6] if a.output.endswith(".jsonl") else a.output
    outs = {"train": open(base + ".train.jsonl", "w"), "test": open(base + ".test.jsonl", "w")} \
        if a.test_fraction else {"all": open(a.output, "w")}
    counts = {k: 0 for k in outs}
    skipped = 0
    for r in rows:
        lab = r[a.label]
        if lab is None:
            skipped += 1
            continue
        if kind == "noul":
            expected = _truthy(lab)
            gold = {"label": "true" if expected else "false",
                    "probabilities": {"true": float(expected), "false": float(not expected)}}
        elif kind == "choice":
            expected = str(lab)
            if expected not in opts:
                skipped += 1
                continue
            gold = {"label": expected, "probabilities": {o: float(o == expected) for o in opts}}
        else:
            idx = opts.index(str(lab)) if str(lab) in opts else int(lab)
            expected = idx
            gold = {"label": idx, "probabilities": {str(i): float(i == idx) for i in range(len(opts))}}
        state = json.loads(to_row_json(view(r), not a.keep_nulls))
        rid = hashlib.sha1(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]
        split = "all"
        if a.test_fraction:
            split = "test" if int(rid[:8], 16) / 0xFFFFFFFF < a.test_fraction else "train"
        if a.format == "typed-decisions":
            rec = {"id": rid, "workflow": a.workflow, "state": json.dumps(state, ensure_ascii=False),
                   "questions": json.dumps(q), "gold": json.dumps({"q": gold})}
        else:
            rec = {"state": state, "questions": q, "expected": {"q": expected}, "tags": [a.workflow]}
        outs[split].write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        counts[split] += 1
    for f in outs.values():
        f.close()
    print("sql-jev-laya: %s%s" % (", ".join("%s %d" % kv for kv in counts.items()),
                             ", skipped %d (no/unknown label)" % skipped if skipped else ""), file=sys.stderr)


def cmd_eval(a):
    """Accuracy of the configured backend on a laya-evals JSONL file (from `sql-jev-laya dataset`)."""
    jev = Jev(**_settings(a))
    groups = {}
    for line in open(a.file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        for qid, qd in rec["questions"].items():
            crit = qd.get("criteria")
            opts = list(crit) if crit else None
            groups.setdefault((qd["type"], qd["instructions"], json.dumps(opts)), []).append(
                (rec["state"], rec["expected"][qid]))
    t0 = time.time()
    total = correct = 0
    for (kind, instr, opts_json), items in groups.items():
        opts = json.loads(opts_json)
        answers = jev.evaluate([s for s, _ in items], _unphrase(kind, instr), kind, opts)
        for (_, exp), ans in zip(items, answers):
            if kind == "noul":
                ok = (ans["noul"] >= 0.5) == bool(exp)
            elif kind == "choice":
                ok = ans["choice"] == exp
            else:
                ok = round(ans["score"]) == int(exp)
            correct += ok
            total += 1
    acc = correct / total if total else 0.0
    print(json.dumps({"backend": jev.cfg["backend"], "model": jev.cfg["model"], "decisions": total,
                      "accuracy": round(acc, 4), "seconds": round(time.time() - t0, 2)}))
    if a.min_accuracy is not None and acc < a.min_accuracy:
        sys.exit(1)


def _unphrase(kind, instructions):
    """Recover the user's condition from laya_question()'s wording, so eval asks the same question."""
    prefix = "Is it true that "
    if kind == "noul" and instructions.startswith(prefix) and instructions.endswith("?"):
        return instructions[len(prefix):-1]
    return instructions


def cmd_gateway(a):
    from .gateway import serve
    serve(a.host, a.port, Jev(**_settings(a)), certfile=a.certfile, keyfile=a.keyfile)


def cmd_judge_sqlserver(a):
    """Fill jev.answers (sql/sqlserver/install.sql) from outside the server, for SQL Server 2016-2022 or
    servers without outbound HTTPS. Row JSON and hashes are computed by SQL Server itself (FOR JSON,
    jev.row_hash), so jev.prob() finds exactly these answers."""
    import sqlalchemy as sa
    mode, text, kind, opts = _question(a)
    opts_json = json.dumps(opts) if opts else None
    jev = Jev(**_settings(a))
    t0 = time.time()
    with _engine(a.url).begin() as conn:
        name = conn.execute(sa.text("SELECT QUOTENAME(OBJECT_SCHEMA_NAME(OBJECT_ID(:s))) + N'.' + "
                                    "QUOTENAME(OBJECT_NAME(OBJECT_ID(:s)))"), {"s": a.source}).scalar()
        if not name:
            raise JevError("sql-jev-laya: %r is not a table or view" % a.source)
        qkey = conn.execute(sa.text("SELECT jev.question_key(:q, :k, :o)"),
                            {"q": text, "k": kind, "o": opts_json}).scalar()
        sql = ("SELECT jev.row_hash(x.j) AS h, x.j FROM (SELECT (SELECT t.* FOR JSON PATH, WITHOUT_ARRAY_WRAPPER)"
               " AS j FROM %s AS t%s) AS x WHERE NOT EXISTS (SELECT 1 FROM jev.answers AS a"
               " WHERE a.question_key = :qk AND a.row_hash = jev.row_hash(x.j))"
               % (name, " WHERE " + a.filter if a.filter else ""))
        todo = {}
        for h, j in conn.execute(sa.text(sql), {"qk": qkey}):
            todo.setdefault(bytes(h), j)
        answers = jev.evaluate(list(todo.values()), text, kind, opts)
        rows = [{"qk": qkey, "h": h, "a": json.dumps(ans)} for h, ans in zip(todo, answers)]
        for i in range(0, len(rows), 500):
            conn.execute(sa.text("INSERT jev.answers (question_key, row_hash, answer) VALUES (:qk, :h, :a)"),
                         rows[i:i + 500])
    print("sql-jev-laya: judged %d new rows of %s" % (len(rows), name), file=sys.stderr)
    _report(jev, t0, a)


def _report(jev, t0, a):
    if a.stats:
        s = jev.stats()
        s["seconds"] = round(time.time() - t0, 2)
        print("sql-jev-laya: " + json.dumps(s), file=sys.stderr)


# ---------------------------------------------------------------- argument parsing

def _common(p):
    g = p.add_argument_group("model")
    g.add_argument("--backend", choices=core.BACKENDS, help="default: local (Laya in-process); env SQLJEV_BACKEND")
    g.add_argument("--api-url", help="gateway / laya-serve / Jev endpoint")
    g.add_argument("--model", help="Laya checkpoint (english, multilingual, typed-decisions), a fine-tuned "
                                   "checkpoint directory or Hub id, or a Jev model id")
    g.add_argument("--device", help="cpu, cuda, mps (local backend)")
    g.add_argument("--batch-size", type=int, help="rows per forward pass / request")
    g.add_argument("--concurrency", type=int)
    g.add_argument("--max-len", type=int, help="Laya token budget per row (multilingual reads up to 8192)")
    g.add_argument("--max-rows", type=int, help="spend guard: refuse to send more rows than this")
    g.add_argument("--keep-nulls", action="store_true", help="send NULL columns to the model too")
    g.add_argument("--stats", action="store_true", help="print engine stats to stderr")


def _questions(p, dataset=False):
    g = p.add_argument_group("question (exactly one)")
    if dataset:
        g.add_argument("--noul", metavar="CONDITION", help="yes/no condition; label column holds true/false")
    else:
        g.add_argument("--where", metavar="CONDITION", help="keep rows that satisfy the condition (streams)")
        g.add_argument("--rank", metavar="CONDITION", help="order rows by probability, most likely first")
        g.add_argument("--prob", metavar="CONDITION", help="add jev_prob for every row")
    g.add_argument("--choice", metavar="QUESTION", help="classify each row into one of --options")
    g.add_argument("--score", metavar="QUESTION", help="rate each row along ordered --levels")
    g.add_argument("--options", type=core.parse_options, help="comma list or JSON array")
    g.add_argument("--levels", type=core.parse_options, help="comma list or JSON array, lowest first")
    p.add_argument("--columns", help="comma list of columns the model sees (default: all)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sql-jev-laya", description="Ask your SQL rows questions in plain language.")
    p.add_argument("--version", action="version", version="sql-jev-laya " + core.__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="run a SELECT and filter / rank / classify its rows")
    q.add_argument("url", help="SQLAlchemy URL, e.g. postgresql://..., mssql+pyodbc://..., snowflake://...")
    q.add_argument("sql")
    _questions(q)
    q.add_argument("--threshold", type=float)
    q.add_argument("--limit", type=int)
    q.add_argument("--sort", action="store_true", help="with --score: highest first")
    q.add_argument("--format", choices=("table", "csv", "jsonl"), default="table")
    _common(q)
    q.set_defaults(fn=cmd_query)

    m = sub.add_parser("materialize", help="write judgments into a table, keyed by a column")
    m.add_argument("url")
    m.add_argument("sql")
    m.add_argument("--key", required=True)
    m.add_argument("--into", required=True, help="table name ([schema.]table)")
    m.add_argument("--replace", action="store_true")
    _questions(m)
    _common(m)
    m.set_defaults(fn=cmd_materialize)

    d = sub.add_parser("dataset", help="labelled rows -> Laya fine-tuning / eval JSONL")
    d.add_argument("url")
    d.add_argument("sql")
    d.add_argument("--label", required=True, help="column holding the ground truth (never shown to the model)")
    d.add_argument("-o", "--output", required=True)
    d.add_argument("--format", choices=("laya-evals", "typed-decisions"), default="laya-evals")
    d.add_argument("--test-fraction", type=float, default=0.0, help="write OUTPUT.train/.test.jsonl")
    d.add_argument("--workflow", default="sql")
    d.add_argument("--keep-nulls", action="store_true")
    _questions(d, dataset=True)
    d.set_defaults(fn=cmd_dataset)

    e = sub.add_parser("eval", help="accuracy of a backend / checkpoint on a dataset file")
    e.add_argument("file")
    e.add_argument("--min-accuracy", type=float)
    _common(e)
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("gateway", help="serve the HTTP gateway for SQL Server, Snowflake, BigQuery, Redshift")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--certfile", help="serve HTTPS with this certificate (PEM)")
    g.add_argument("--keyfile", help="private key for --certfile")
    _common(g)
    g.set_defaults(fn=cmd_gateway)

    s = sub.add_parser("judge-sqlserver", help="fill jev.answers on SQL Server from outside (2016-2022)")
    s.add_argument("url", help="mssql+pyodbc://... or mssql+pymssql://...")
    s.add_argument("--source", required=True, help="table or view, e.g. dbo.tickets")
    s.add_argument("--where", dest="filter", help="T-SQL filter applied before judging")
    g2 = s.add_argument_group("question (exactly one)")
    g2.add_argument("--prob", metavar="CONDITION")
    g2.add_argument("--choice", metavar="QUESTION")
    g2.add_argument("--score", metavar="QUESTION")
    g2.add_argument("--options", type=core.parse_options)
    g2.add_argument("--levels", type=core.parse_options)
    _common(s)
    s.set_defaults(fn=cmd_judge_sqlserver)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except JevError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
