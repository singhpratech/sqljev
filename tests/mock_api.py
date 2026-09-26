#!/usr/bin/env python3
"""Deterministic stand-in for a /v1/systemone server (TypeSafe Jev or laya-serve), and a fake `laya`
module for the in-process backend. Adapted from pg-jev's test/mock_api.py (PostgreSQL License).

Rules (so expected output is stable):
  noul   -> 0.9 if the LAST word of the condition appears (case-insensitively) in the row JSON, else 0.1
  score  -> level index = length of the row JSON (sorted keys) modulo number of levels
  choice -> option index = length of the row JSON (sorted keys) modulo number of options
  A condition containing "trigger422" returns HTTP 422; "trigger429" returns 429 on its first request.
  Jev requests (state.rows, questions r0..rN) require "Bearer test-key".
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAYA_PREFIX = "Is it true that "


def condition_of(instructions):
    if instructions.startswith(LAYA_PREFIX):
        return instructions[len(LAYA_PREFIX):].rstrip("?")
    return instructions.rstrip("?")


def judge(row, qtype, condition, criteria):
    row_json = json.dumps(row, sort_keys=True)
    if qtype == "noul":
        words = condition.split()
        needle = words[-1].lower() if words else ""
        p = 0.9 if needle and needle in row_json.lower() else 0.1
        return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}
    opts = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
    k = len(row_json) % len(opts)
    probs = {(o if qtype == "choice" else str(j)): (1.0 if j == k else 0.0) for j, o in enumerate(opts)}
    if qtype == "choice":
        return {"type": "choice", "choice": opts[k], "probabilities": probs, "confidence": 1.0}
    return {"type": "score", "score": float(k), "probabilities": probs, "confidence": 1.0}


def answer_request(req):
    """Returns (status, body) for one /v1/systemone request."""
    state, questions = req["state"], req["questions"]
    if isinstance(state, dict) and "rows" in state and all(q.startswith("r") for q in questions):
        rows, cond = state["rows"], state.get("condition", "")
        answers = {}
        for qid, q in questions.items():
            c = cond if q["type"] == "noul" else q["instructions"]
            answers[qid] = judge(rows[int(qid[1:])], q["type"], c, q.get("criteria"))
        return answers, cond
    answers = {qid: judge(state, q["type"], condition_of(q["instructions"]), q.get("criteria"))
               for qid, q in questions.items()}
    return answers, " ".join(q["instructions"] for q in questions.values())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen429 = set()
    requests = 0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body)
        jev_format = isinstance(req.get("state"), dict) and "rows" in req["state"]
        if jev_format and self.headers.get("Authorization", "") != "Bearer test-key":
            return self._send(401, {"error": "invalid api key"})
        answers, text = answer_request(req)
        if "trigger422" in text:
            return self._send(422, {"error": "mock validation failure"})
        if "trigger429" in text:
            with Handler.lock:
                first = text not in Handler.seen429
                Handler.seen429.add(text)
            if first:
                return self._send(429, {"error": "slow down"}, {"retry-after-ms": "10"})
        with Handler.lock:
            Handler.requests += 1
        self._send(200, {"model": "mock", "answers": answers,
                         "usage": {"input_tokens": len(body) // 4, "output_tokens": len(answers)}})

    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------- fake `laya` package for backend "local"

class FakeRouter:
    batches = []

    def __init__(self, device=None, **kw):
        self.device = device

    def predict_batch(self, requests, batch_size=None):
        FakeRouter.batches.append(len(requests))
        return [{"answers": answer_request(r)[0], "routing": {"model": r.get("model", "english")}}
                for r in requests]


class FakeAgent:
    def __init__(self, path):
        self.path = path

    def predict_batch(self, states, questions, batch_size=None, max_len=None, sort_by_length=False):
        FakeRouter.batches.append(len(states))
        return [{"answers": answer_request({"state": s, "questions": questions})[0]} for s in states]


def fake_laya_module():
    import types
    m = types.ModuleType("laya")
    m.Router = FakeRouter
    m.load = lambda path, device=None: FakeAgent(path)
    m.__version__ = "fake"
    return m


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
