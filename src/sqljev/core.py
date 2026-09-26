"""sqljev.core — plain-language predicates for SQL rows, judged by a System One decision model.

    j = Jev()                                     # backend "local": Laya in-process
    j.prob(rows, "the customer is angry")          # [0.93, 0.08, ...]
    j.call("jev_choice", [(row, "which team?", ["billing", "technical"]), ...])

A SQL question is the same question over many rows, so everything here is organised around one
(question, rows) pair: rows are serialised to compact JSON objects, de-duplicated by content, looked up
in an answer cache and only the misses are sent to the model, in as few forward passes / requests as the
backend allows.

Backends (setting `backend`, env SQLJEV_BACKEND):

  local       Laya (Convai Innovations, Apache 2.0) in this process via `laya` — `predict_batch` packs
              many rows into shared forward passes. Default. Fastest, free, data never leaves the host.
  gateway     a `sqljev gateway` over HTTP (which runs Laya locally) — many rows per request.
              For databases that call out over HTTP: SQL Server, Snowflake, BigQuery, Redshift.
  laya-serve  a stock `laya-serve` over the Jev wire protocol (POST /v1/systemone) — one row per request.
  jev         TypeSafe's hosted Jev (POST /v1/systemone) — 20 rows per request in one shared state.

This file is deliberately one stdlib-only module, so it can be pasted into a database UDF body.
`laya` (and torch) are imported lazily and only by the `local` backend.

The connection pool, retries and the Jev batch prompt are ported from pg-jev
(https://github.com/realZachi/pg-jev, PostgreSQL License, Copyright (c) 2026 Zachi).
"""
import hashlib
import http.client
import json
import os
import random
import select
import socket
import ssl
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

__version__ = "0.1.0"

BACKENDS = ("local", "gateway", "laya-serve", "jev")
KINDS = ("noul", "score", "choice")
LAYA_CHECKPOINTS = ("english", "multilingual", "typed-decisions")
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000   # jev-1.13 list price; output tokens are free

# Per-backend defaults. batch_size means: rows per forward pass (local), rows per HTTP request (the rest).
BACKEND_DEFAULTS = {
    "local": {"api_url": None, "batch_size": 64, "concurrency": 1},
    "gateway": {"api_url": "http://127.0.0.1:8765/v1/eval", "batch_size": 256, "concurrency": 4},
    # Laya reads one state of 512-1,024 tokens: packing several rows into it would cut rows off.
    "laya-serve": {"api_url": "http://127.0.0.1:8000/v1/systemone", "batch_size": 1, "concurrency": 4},
    # pg-jev measured: batches of 1-20 rows 100 % correct, 40 rows 92-98 %, 80 rows 77-94 %.
    "jev": {"api_url": "https://api.typesafe.ai/v1/systemone", "batch_size": 20, "concurrency": 16,
            "model": "jev-latest"},
}
COMMON_DEFAULTS = {"backend": "local", "api_key": None, "model": None, "device": None, "max_len": None,
                   "threshold": 0.5, "timeout": 60.0, "keepalive": 600.0, "max_rows": 0, "max_chars": 0,
                   "cache_size": 200_000, "drop_nulls": True}


def _bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "on", "yes")


SETTINGS = {"backend": str, "api_url": str, "api_key": str, "model": str, "device": str, "max_len": int,
            "batch_size": int, "concurrency": int, "threshold": float, "timeout": float, "keepalive": float,
            "max_rows": int, "max_chars": int, "cache_size": int, "drop_nulls": _bool}


class JevError(RuntimeError):
    """Raised for configuration, validation, spend-guard and model/API errors."""


def make_config(env=None, **overrides):
    """Settings from SQLJEV_<NAME> environment variables, then keyword overrides (None = unset),
    then the backend's defaults. Returns a plain dict (picklable, so it travels to Spark executors)."""
    env = os.environ if env is None else env
    cfg = {}
    for name, cast in SETTINGS.items():
        v = env.get("SQLJEV_" + name.upper())
        if v not in (None, ""):
            cfg[name] = cast(v)
    for name, v in overrides.items():
        if name not in SETTINGS:
            raise JevError("sqljev: unknown setting %r (known: %s)" % (name, ", ".join(sorted(SETTINGS))))
        if v is not None:
            cfg[name] = SETTINGS[name](v)
    backend = cfg.get("backend", COMMON_DEFAULTS["backend"])
    if backend not in BACKENDS:
        raise JevError("sqljev: unknown backend %r (one of %s)" % (backend, ", ".join(BACKENDS)))
    for k, v in {**COMMON_DEFAULTS, **BACKEND_DEFAULTS[backend]}.items():
        cfg.setdefault(k, v)
    if not cfg["api_key"]:
        cfg["api_key"] = env.get({"jev": "TYPESAFE_API_KEY", "laya-serve": "LAYA_API_KEY",
                                  "gateway": "SQLJEV_GATEWAY_TOKEN"}.get(backend, ""), None)
    if backend == "laya-serve":
        cfg["batch_size"] = 1
    cfg["batch_size"] = max(1, cfg["batch_size"])
    cfg["concurrency"] = max(1, cfg["concurrency"])
    return cfg


# ---------------------------------------------------------------- rows

def to_row_json(row, drop_nulls=True):
    """Compact JSON text for a row. dicts (column -> value) become objects; SQL NULL columns are dropped
    by default, which saves tokens in Laya's 512-1,024-token window. A string that already holds JSON
    (to_json(t), OBJECT_CONSTRUCT(*), FOR JSON ...) is re-encoded the same way; other strings become a
    JSON string."""
    if isinstance(row, (bytes, bytearray)):
        row = row.decode()
    if isinstance(row, str):
        try:
            row = json.loads(row)
        except ValueError:
            return json.dumps(row, ensure_ascii=False)
    if drop_nulls and isinstance(row, dict):
        row = {k: v for k, v in row.items() if v is not None}
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str)


def row_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()


def parse_options(v):
    """text[] / ARRAY / list / numpy array / '["a","b"]' / 'a,b' -> list of str, or None."""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            v = json.loads(s)
        else:
            v = [p.strip() for p in s.split(",") if p.strip()]
    return [str(o) for o in v]


def check_question(kind, options):
    if kind not in KINDS:
        raise JevError("sqljev: unknown kind %r (one of noul, score, choice)" % (kind,))
    opts = parse_options(options)
    if kind == "noul":
        return kind, None
    if not opts or len(opts) < 2:
        raise JevError("sqljev: %s needs at least two %s" % (kind, "levels" if kind == "score" else "options"))
    return kind, opts


# ---------------------------------------------------------------- questions: one place, used at inference and for training data

def _clause(text):
    return text.strip().rstrip("?.!").strip()


def laya_question(kind, query, opts):
    """The question Laya is asked about one SQL row (the row is the state). `sqljev dataset` builds
    training data with this same function, so a fine-tuned checkpoint sees exactly what it is asked."""
    if kind == "noul":
        # A statement ("the customer is angry") becomes "Is it true that the customer is angry?"; a question
        # is asked as written. On labelled rows this beat "Does this record satisfy ..." phrasings.
        q = query.strip()
        return {"type": "noul", "instructions": q if q.endswith("?") else "Is it true that %s?" % _clause(q)}
    if kind == "score":
        return {"type": "score", "instructions": query, "criteria": list(opts)}
    return {"type": "choice", "instructions": query, "criteria": {o: o for o in opts}}


def jev_question(i, kind, query, opts):
    """pg-jev's batch prompt: one shared state {"condition", "rows"}, one question per row index."""
    ref = "rows[%d]" % i
    if kind == "noul":
        return {"type": "noul",
                "instructions": "Does the record `%s` satisfy the condition stated in `condition`?" % ref}
    if kind == "score":
        return {"type": "score", "instructions": "Rate the record `%s`: %s" % (ref, query), "criteria": opts}
    return {"type": "choice", "instructions": "For the record `%s`: %s" % (ref, query),
            "criteria": {o: None for o in opts}}


# ---------------------------------------------------------------- UDF-style dispatch shared by every database adapter

# name -> (kind or None when it is an argument, min args, max args); args start with (row, question)
FUNCTIONS = {
    "jev": ("noul", 2, 3),              # row, condition [, threshold]           -> boolean
    "jev_prob": ("noul", 2, 2),         # row, condition                          -> float
    "jev_score": ("score", 3, 3),       # row, question, levels                   -> float 0..n-1
    "jev_score_norm": ("score", 3, 3),  # row, question, levels                   -> float 0..1
    "jev_choice": ("choice", 3, 3),     # row, question, options                  -> text
    "jev_confidence": (None, 4, 4),     # row, question, kind, options            -> float
    "jev_eval": (None, 2, 4),           # row, question [, kind [, options]]      -> answer object
}


def function_name(fn):
    name = str(fn).rsplit(".", 1)[-1].strip('`"[] ').lower()
    if name not in FUNCTIONS:
        raise JevError("sqljev: unknown function %r (one of %s)" % (fn, ", ".join(FUNCTIONS)))
    return name


# ---------------------------------------------------------------- the engine

class Jev:
    """Judges SQL rows against plain-language questions. Thread-safe; keep one per process."""

    def __init__(self, cfg=None, **overrides):
        self.cfg = make_config(**{**(cfg or {}), **overrides})
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._cache = OrderedDict()
        self._conns = []
        self._pool = None
        self._ssl = None
        self._laya = None
        self._sent = [0, 0]      # rows, chars sent since the last reset_budget()
        self._stats = {"requests": 0, "forward_batches": 0, "rows_evaluated": 0, "cache_hits": 0,
                       "duplicates": 0, "input_tokens": 0, "output_tokens": 0, "model_ms": 0.0,
                       "errors": 0, "retries": 0}

    # ------------------------------------------------ public API

    def evaluate(self, rows, query, kind="noul", options=None):
        """Answer dicts aligned with `rows` (None for a None row). Duplicates and cached rows cost nothing."""
        kind, opts = check_question(kind, options)
        key = self._qkey(kind, query, opts)
        texts = [None if r is None else to_row_json(r, self.cfg["drop_nulls"]) for r in rows]
        hashes = [None if t is None else row_hash(t) for t in texts]
        found, todo = {}, {}
        for h, t in zip(hashes, texts):
            if h is None:
                continue
            if h in found or h in todo:
                self._bump("duplicates")
                continue
            a = self._cache_get(key, h)
            if a is None:
                todo[h] = t
            else:
                found[h] = a
                self._bump("cache_hits")
        if todo:
            pairs = list(todo.items())
            bs = self.cfg["batch_size"]
            batches = [pairs[i:i + bs] for i in range(0, len(pairs), bs)]
            self._guard(pairs)
            if len(batches) == 1 or self.cfg["concurrency"] == 1:
                for b in batches:
                    found.update(self._run_batch(key, kind, query, opts, b))
            else:
                futs = [self._executor().submit(self._run_batch, key, kind, query, opts, b) for b in batches]
                for f in futs:
                    found.update(f.result())
        return [None if h is None else found[h] for h in hashes]

    def evaluate_iter(self, rows, query, kind="noul", options=None, key=None):
        """Stream (row, answer) pairs in input order with a bounded read-ahead, so a consumer that stops
        early (LIMIT) only pays for the rows in flight. Memory stays constant for any number of rows.
        `key(item)` picks what the model sees of each item (default: the item itself)."""
        kind, opts = check_question(kind, options)
        view = key or (lambda item: item)
        key = self._qkey(kind, query, opts)
        bs, cap = self.cfg["batch_size"], 2 * self.cfg["concurrency"]
        limit = max(bs * cap, 64)
        it, window, inflight, active = iter(rows), deque(), {}, []
        pending, pending_set, exhausted = [], set(), False

        def submit(pairs):
            self._guard(pairs)
            fut = self._executor().submit(self._run_batch, key, kind, query, opts, pairs)
            for h, _ in pairs:
                inflight[h] = fut
            active.append(fut)
            return fut

        def flush():
            nonlocal pending
            if pending:
                submit(pending)
                pending_set.clear()
                pending = []

        while True:
            active[:] = [f for f in active if not f.done()]
            while not exhausted and len(window) < limit and len(active) < cap:
                try:
                    row = next(it)
                except StopIteration:
                    exhausted = True
                    break
                seen = None if row is None else view(row)
                if seen is None:
                    window.append([row, None, None, None])
                    continue
                t = to_row_json(seen, self.cfg["drop_nulls"])
                h = row_hash(t)
                a = self._cache_get(key, h)
                window.append([row, h, t, a])
                if a is not None:
                    self._bump("cache_hits")
                elif h in inflight or h in pending_set:
                    self._bump("duplicates")
                else:
                    pending.append((h, t))
                    pending_set.add(h)
                    if len(pending) >= bs:
                        flush()
            if exhausted:
                flush()
            if not window:
                return
            row, h, t, a = window[0]
            if a is None and h is not None:
                a = self._cache_get(key, h)
                if a is None:
                    if h in pending_set:
                        flush()
                    fut = inflight.get(h) or submit([(h, t)])
                    a = fut.result()[h]
                inflight.pop(h, None)
            window.popleft()
            yield row, a

    def call(self, fn, calls):
        """Evaluate a UDF call batch: `calls` is a list of argument tuples (row, question, ...) as a
        database hands them over. Calls are grouped by question, so one batch can mix questions."""
        name = function_name(fn)
        fixed_kind, lo, hi = FUNCTIONS[name]
        calls = [tuple(_null(v) for v in c) for c in calls]
        out = [None] * len(calls)
        groups = {}
        for i, args in enumerate(calls):
            if not lo <= len(args) <= hi:
                raise JevError("sqljev: %s takes %d-%d arguments, got %d" % (name, lo, hi, len(args)))
            row, query = args[0], args[1]
            if row is None or query is None:
                continue
            if fixed_kind:
                kind = fixed_kind
                opts = args[2] if kind != "noul" else None
            else:
                kind = args[2] if len(args) > 2 and args[2] is not None else "noul"
                opts = args[3] if len(args) > 3 else None
            kind, opts = check_question(kind, opts)
            gk = (kind, query, None if opts is None else tuple(opts))
            groups.setdefault(gk, []).append(i)
        for (kind, query, opts), idx in groups.items():
            answers = self.evaluate([calls[i][0] for i in idx], query, kind, opts and list(opts))
            for i, a in zip(idx, answers):
                out[i] = self._extract(name, a, calls[i], opts)
        return out

    def prob(self, rows, condition):
        return self.call("jev_prob", [(r, condition) for r in rows])

    def where(self, rows, condition, threshold=None):
        return self.call("jev", [(r, condition, threshold) for r in rows])

    def score(self, rows, question, levels, normalize=False):
        return self.call("jev_score_norm" if normalize else "jev_score", [(r, question, levels) for r in rows])

    def choice(self, rows, question, options):
        return self.call("jev_choice", [(r, question, options) for r in rows])

    def stats(self):
        with self._lock:
            s = dict(self._stats)
            s["cached_answers"] = len(self._cache)
            s["pooled_connections"] = len(self._conns)
        s["backend"] = self.cfg["backend"]
        s["estimated_cost_usd"] = round(s["input_tokens"] * JEV_USD_PER_INPUT_TOKEN, 6) \
            if self.cfg["backend"] == "jev" else 0.0
        return s

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def reset_budget(self):
        with self._lock:
            self._sent = [0, 0]

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        with self._lock:
            conns, self._conns = self._conns, []
        for c, _ in conns:
            c.close()

    # ------------------------------------------------ internals

    def _extract(self, name, a, args, opts):
        if a is None:
            return None
        if name == "jev":
            t = args[2] if len(args) > 2 and args[2] is not None else self.cfg["threshold"]
            return a["noul"] >= float(t)
        if name == "jev_prob":
            return a["noul"]
        if name == "jev_score":
            return a["score"]
        if name == "jev_score_norm":
            return a["score"] / max(len(opts) - 1, 1)
        if name == "jev_choice":
            return a["choice"]
        if name == "jev_confidence":
            return a.get("confidence")
        return a

    def _qkey(self, kind, query, opts):
        return json.dumps([kind, query, opts])

    def _bump(self, name, n=1):
        with self._lock:
            self._stats[name] += n

    def _cache_get(self, key, h):
        if not self.cfg["cache_size"]:
            return None
        with self._lock:
            a = self._cache.get((key, h))
            if a is not None:
                self._cache.move_to_end((key, h))
            return a

    def _cache_put(self, key, answers):
        size = self.cfg["cache_size"]
        if not size:
            return
        with self._lock:
            for h, a in answers.items():
                self._cache[(key, h)] = a
            while len(self._cache) > size:
                self._cache.popitem(last=False)

    def _guard(self, pairs):
        """Spend guard: refuse to send more than max_rows rows / max_chars characters of row data
        from this engine until reset_budget()."""
        mr, mc = self.cfg["max_rows"], self.cfg["max_chars"]
        if not (mr or mc):
            return
        with self._lock:
            rows, chars = self._sent[0] + len(pairs), self._sent[1] + sum(len(t) for _, t in pairs)
            if mr and rows > mr:
                raise JevError("sqljev: would send %d rows to the model, above max_rows = %d" % (rows, mr))
            if mc and chars > mc:
                raise JevError("sqljev: would send %d characters of row data, above max_chars = %d" % (chars, mc))
            self._sent = [rows, chars]

    def _executor(self):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.cfg["concurrency"], thread_name_prefix="sqljev")
            return self._pool

    def _run_batch(self, key, kind, query, opts, pairs):
        """Judge one batch of (row_hash, row_json) pairs; returns {row_hash: answer}."""
        try:
            b = self.cfg["backend"]
            if b == "local":
                got = self._run_local(kind, query, opts, pairs)
            elif b == "gateway":
                got = self._run_gateway(kind, query, opts, pairs)
            else:
                got = self._run_systemone(kind, query, opts, pairs)
        except Exception:
            self._bump("errors")
            raise
        self._cache_put(key, got)
        self._bump("rows_evaluated", len(pairs))
        return got

    # -- local: Laya in-process, many rows per forward pass

    def _laya_model(self):
        if self._laya is None:
            try:
                import laya
            except ImportError:
                raise JevError("sqljev: backend 'local' needs Laya: pip install 'sqljev[laya]'")
            m = self.cfg["model"]
            if m and m not in LAYA_CHECKPOINTS:     # a fine-tuned checkpoint: local directory or Hub id
                self._laya = ("agent", laya.load(m, device=self.cfg["device"]))
            else:
                self._laya = ("router", laya.Router(device=self.cfg["device"]))
        return self._laya

    def _run_local(self, kind, query, opts, pairs):
        q = {"q": laya_question(kind, query, opts)}
        states = [json.loads(t) for _, t in pairs]
        t0 = time.time()
        with self._model_lock:                      # one forward pass at a time; the model is the bottleneck
            how, m = self._laya_model()
            if how == "agent":
                res = m.predict_batch(states, q, batch_size=self.cfg["batch_size"], max_len=self.cfg["max_len"],
                                      sort_by_length=True)
            else:
                req = [{"state": s, "questions": q} for s in states]
                if self.cfg["model"]:
                    for r in req:
                        r["model"] = self.cfg["model"]
                res = m.predict_batch(req, batch_size=self.cfg["batch_size"])
        with self._lock:
            self._stats["forward_batches"] += -(-len(pairs) // self.cfg["batch_size"])
            self._stats["model_ms"] += (time.time() - t0) * 1000
        return {h: r["answers"]["q"] for (h, _), r in zip(pairs, res)}

    # -- gateway: sqljev's own batch protocol

    def _run_gateway(self, kind, query, opts, pairs):
        body = json.dumps({"kind": kind, "question": query, "options": opts,
                           "rows": [json.loads(t) for _, t in pairs]}).encode()
        data = self._post(body)
        answers = data.get("answers") or []
        if len(answers) != len(pairs):
            raise JevError("sqljev: gateway returned %d answers for %d rows" % (len(answers), len(pairs)))
        return {h: a for (h, _), a in zip(pairs, answers)}

    # -- /v1/systemone: TypeSafe Jev (20 rows in one state) or laya-serve (one row per request)

    def _run_systemone(self, kind, query, opts, pairs):
        rows = [json.loads(t) for _, t in pairs]
        if self.cfg["backend"] == "jev":
            state = {"condition": query, "rows": rows} if kind == "noul" else {"rows": rows}
            qs = {("r%d" % i): jev_question(i, kind, query, opts) for i in range(len(rows))}
            ids = ["r%d" % i for i in range(len(rows))]
        else:
            state, qs, ids = rows[0], {"q": laya_question(kind, query, opts)}, ["q"]
        req = {"state": state, "questions": qs}
        if self.cfg["model"]:
            req["model"] = self.cfg["model"]
        data = self._post(json.dumps(req).encode())
        answers = data.get("answers") or {}
        if any(i not in answers for i in ids):
            raise JevError("sqljev: the model returned %d of %d answers" % (len(answers), len(ids)))
        usage = data.get("usage") or {}
        with self._lock:
            self._stats["input_tokens"] += usage.get("input_tokens", 0)
            self._stats["output_tokens"] += usage.get("output_tokens", 0)
        return {h: answers[i] for (h, _), i in zip(pairs, ids)}

    # -- HTTP: persistent keep-alive connections, retries honouring Retry-After (from pg-jev)

    def _borrow(self, url):
        while True:
            with self._lock:
                entry = self._conns.pop() if self._conns else None
            if entry is None:
                break
            c, last = entry
            if time.time() - last < self.cfg["keepalive"] and _alive(c):
                return c, True
            c.close()
        if url.scheme == "https":
            if self._ssl is None:
                self._ssl = ssl.create_default_context()
            c = http.client.HTTPSConnection(url.hostname, url.port, timeout=self.cfg["timeout"], context=self._ssl)
        else:
            c = http.client.HTTPConnection(url.hostname, url.port, timeout=self.cfg["timeout"])
        return c, False

    def _release(self, c, reusable):
        if reusable:
            with self._lock:
                if len(self._conns) < self.cfg["concurrency"]:
                    self._conns.append((c, time.time()))
                    return
        c.close()

    def _post(self, body):
        url = urlsplit(self.cfg["api_url"] or "")
        if url.scheme not in ("http", "https"):
            raise JevError("sqljev: api_url %r is not an http(s) URL" % self.cfg["api_url"])
        if self.cfg["backend"] == "jev" and not self.cfg["api_key"]:
            raise JevError("sqljev: no API key. Set SQLJEV_API_KEY or TYPESAFE_API_KEY.")
        path = (url.path or "/") + ("?" + url.query if url.query else "")
        headers = {"Content-Type": "application/json", "User-Agent": "sqljev/" + __version__}
        if self.cfg["api_key"]:
            headers["Authorization"] = "Bearer " + self.cfg["api_key"]
        delay, last = 0.5, None
        for attempt in range(7):
            c, reused = self._borrow(url)
            t0 = time.time()
            try:
                if not reused:
                    c.connect()
                    _tcp_keepalive(c)
                c.request("POST", path, body=body, headers=headers)
                resp = c.getresponse()
                raw = resp.read()
            except (http.client.HTTPException, OSError) as e:
                c.close()
                last = "%s: %s" % (type(e).__name__, e)
                self._bump("retries")
                if reused and attempt == 0:
                    continue                  # a keep-alive connection went stale: retry at once on a fresh one
                time.sleep(delay + random.random() * 0.25)
                delay = min(delay * 2, 8)
                continue
            self._release(c, not resp.will_close)
            if resp.status == 200:
                with self._lock:
                    self._stats["requests"] += 1
                    self._stats["model_ms"] += (time.time() - t0) * 1000
                return json.loads(raw.decode())
            last = "%s %s" % (resp.status, raw.decode(errors="replace")[:300])
            if resp.status in (408, 429, 503, 529) or resp.status >= 500:
                self._bump("retries")
                wait = _retry_after(resp)
                time.sleep(min(wait if wait is not None else delay, 30) + random.random() * 0.25)
                delay = min(delay * 2, 8)
                continue
            raise JevError("sqljev: %s error %s" % (self.cfg["backend"], last))
        raise JevError("sqljev: %s unreachable after retries: %s" % (self.cfg["backend"], last))


def _null(v):
    """SQL NULL arrives as None, or as NaN from pandas-based UDF runtimes (Snowflake, Spark)."""
    return None if isinstance(v, float) and v != v else v


def _alive(c):
    """An idle keep-alive connection never has unread data: readability means EOF or a TLS close alert."""
    sock = c.sock
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return not readable
    except (OSError, ValueError):
        return False


def _tcp_keepalive(c):
    try:
        sock = c.sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
            if hasattr(socket, name):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    except OSError:
        pass


def _retry_after(resp):
    v = resp.getheader("retry-after-ms")
    if v and v.strip().isdigit():
        return int(v) / 1000.0
    v = resp.getheader("retry-after")
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    return None


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default_engine(**overrides):
    """A process-wide engine, so a UDF's answer cache and connections survive across calls."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Jev(**overrides)
        return _DEFAULT
