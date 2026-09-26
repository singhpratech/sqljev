#!/usr/bin/env python3
"""Benchmark sqljev end to end through real SQL (DuckDB) on bench/synthetic.py's 10 tables, against their labels.

    python -m sqljev.demo --out bench/bench.duckdb                        # 100,000 rows, 13 tasks
    python bench/run.py --device cuda --batch-size 256                    # every task, Laya on the GPU
    python bench/run.py --task ae_serious --task company_match --limit 2000
    python bench/run.py --fake-model                                      # sqljev's own overhead, no model

Each task is one SQL query, e.g.
    SELECT serious, jev_prob(to_json({'drug': drug, 'narrative': narrative}), '...') FROM adverse_events
The model sees only the task's columns, never the label. Each query runs twice; the second run shows what
re-running costs once answers are cached.
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from sqljev.demo import TASKS  # noqa: E402


def lit(s):
    return "'" + s.replace("'", "''") + "'"


def task_sql(table, cols, question, kind, opts, label, limit):
    row = "to_json({%s})" % ", ".join("%s: %s" % (lit(c), c) for c in cols)
    call = {"noul": "jev_prob(%s, %s)" % (row, lit(question)),
            "choice": "jev_choice(%s, %s, [%s])" % (row, lit(question), ", ".join(map(lit, opts or []))),
            "score": "jev_score(%s, %s, [%s])" % (row, lit(question), ", ".join(map(lit, opts or [])))}[kind]
    # Limit in a subquery: engines compute the SELECT list before ORDER BY ... LIMIT, so a jev() call next to a
    # LIMIT on the same level would judge every row of the table.
    src = "(SELECT * FROM %s ORDER BY id LIMIT %d)" % (table, limit) if limit else table
    return "SELECT %s AS label, %s AS pred FROM %s AS t" % (label, call, src)


def metrics(kind, opts, rows):
    n = len(rows)
    if kind == "noul":
        tp = sum(1 for y, p in rows if y and p >= .5)
        fp = sum(1 for y, p in rows if not y and p >= .5)
        fn = sum(1 for y, p in rows if y and p < .5)
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        return {"accuracy": round((n - fp - fn) / n, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                "baseline": round(max(sum(1 for y, _ in rows if y), sum(1 for y, _ in rows if not y)) / n, 4)}
    if kind == "choice":
        per = {o: [0, 0] for o in opts}
        for y, p in rows:
            per[y][1] += 1
            per[y][0] += y == p
        return {"accuracy": round(sum(v[0] for v in per.values()) / n, 4),
                "recall_per_class": {o: round(v[0] / v[1], 3) if v[1] else None for o, v in per.items()},
                "baseline": round(max(v[1] for v in per.values()) / n, 4)}
    idx = {o: i for i, o in enumerate(opts)}
    exact = sum(1 for y, p in rows if round(p) == idx[y])
    within = sum(1 for y, p in rows if abs(p - idx[y]) <= 1)
    counts = {o: sum(1 for y, _ in rows if y == o) for o in opts}
    return {"accuracy": round(exact / n, 4), "within_one_level": round(within / n, 4),
            "baseline": round(max(counts.values()) / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "bench.duckdb"))
    ap.add_argument("--task", action="append", help="task name (repeatable); default: all")
    ap.add_argument("--limit", type=int, help="rows per task")
    ap.add_argument("--backend")
    ap.add_argument("--api-url")
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--fake-model", action="store_true", help="instant stand-in model: sqljev's own overhead")
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args()

    if a.fake_model:
        sys.path.insert(0, os.path.join(HERE, "..", "tests"))
        import mock_api
        sys.modules["laya"] = mock_api.fake_laya_module()
    import duckdb
    import sqljev.duckdb
    from sqljev import Jev

    settings = {k: v for k, v in {"backend": a.backend, "api_url": a.api_url, "model": a.model, "device": a.device,
                                  "batch_size": a.batch_size}.items() if v is not None}
    eng = Jev(**settings)
    con = duckdb.connect(a.db, read_only=True)
    con.execute("SET threads = 1")            # one UDF chunk at a time: the model is the bottleneck, not DuckDB
    sqljev.duckdb.register(con, engine=eng)

    stop = threading.Event()
    t_start = time.time()

    def progress():
        while not stop.wait(60):
            n = eng.stats()["rows_evaluated"]
            el = time.time() - t_start
            print("  [%5.0fs] %d rows judged so far (%.0f rows/s)" % (el, n, n / el), file=sys.stderr, flush=True)
    threading.Thread(target=progress, daemon=True).start()

    results = []
    for name, table, cols, question, kind, opts, label in TASKS:
        if a.task and name not in a.task:
            continue
        sql = task_sql(table, cols, question, kind, opts, label, a.limit)
        before = eng.stats()
        t0 = time.time()
        rows = con.execute(sql).fetchall()
        first = time.time() - t0
        after = eng.stats()
        t0 = time.time()
        con.execute(sql).fetchall()
        second = time.time() - t0
        r = {"task": name, "kind": kind, "rows": len(rows), "seconds": round(first, 2),
             "rows_per_s": round(len(rows) / first, 1) if first else None,
             "judged_by_model": after["rows_evaluated"] - before["rows_evaluated"],
             "rerun_seconds": round(second, 3), **metrics(kind, opts, rows)}
        results.append(r)
        print("%-17s %6d rows %7.1fs %8.1f rows/s  accuracy %.3f (baseline %.3f)%s" % (
            name, r["rows"], r["seconds"], r["rows_per_s"] or 0, r["accuracy"], r["baseline"],
            "  f1 %.3f" % r["f1"] if "f1" in r else ""), flush=True)
    stop.set()
    total_rows = sum(r["rows"] for r in results)
    total_s = sum(r["seconds"] for r in results)
    summary = {"backend": eng.cfg["backend"], "model": eng.cfg["model"], "device": eng.cfg["device"],
               "batch_size": eng.cfg["batch_size"], "fake_model": a.fake_model, "decisions": total_rows,
               "seconds": round(total_s, 1), "decisions_per_s": round(total_rows / total_s, 1) if total_s else None,
               "tasks": results}
    print("total: %d decisions in %.1fs (%.1f/s)" % (total_rows, total_s, summary["decisions_per_s"] or 0))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
