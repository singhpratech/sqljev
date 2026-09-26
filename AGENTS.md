# AGENTS.md

Guidance for AI coding agents working in this repository.

sqljev adds plain-language predicates (`jev`, `jev_prob`, `jev_choice`, `jev_score`, ...) to SQL Server,
Snowflake, Databricks/Spark, BigQuery, Redshift, DuckDB and any SQLAlchemy database. Rows are judged by a
System One decision model: Laya (open weights, default) or TypeSafe Jev.

## Layout

- `src/sqljev/core.py`: **the engine; one stdlib-only module** (it is inlined into the Snowflake UDF).
  `Jev.evaluate` (batch), `Jev.evaluate_iter` (streaming, bounded read-ahead), `Jev.call(fn, calls)` (the UDF
  dispatch every adapter uses), the four backends, the answer cache, the spend guard, HTTP pooling/retries.
  `laya_question()` is the single source of the Laya prompt; `sqljev dataset` uses it too, so training data
  matches inference. Never import third-party packages at module level here.
- `src/sqljev/gateway.py`: stdlib HTTP server; one route per database batch protocol, all ending in `engine.call`.
- `src/sqljev/{duckdb,spark,aws_lambda}.py`: adapters. `cli.py`: query / materialize / dataset / eval /
  gateway / judge-sqlserver.
- `sql/<db>/`: install scripts. `sql/snowflake/install_udf.sql` is **generated** by `scripts/build_sql.py`;
  edit the script or core.py, then regenerate (CI and `tests/test_snowflake_udf.py` fail when stale).
- `site/`: the GitHub Pages site (mascot: Rowl the owl, `site/assets/rowl.svg`).
- `private/`: **git-ignored, never commit**: credentials, notes, reference clones, datasets.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q                          # never calls a real model (tests/mock_api.py + fake laya module)
python scripts/build_sql.py        # regenerate generated SQL after touching core.py
```

The mock's rules decide expected values: noul → 0.9 if the last word of the condition appears in the row JSON,
else 0.1; choice/score → index `len(row_json) % n`; `trigger422` → HTTP 422; `trigger429` → one 429.

## Rules

- Every behaviour change gets a test against the mock.
- Keep function names and signatures identical across databases (SQL Server: `jev.` schema).
- Changing `laya_question()` or `jev_question()` changes results for every user and invalidates fine-tuned
  checkpoints trained on the old wording: measure on labelled rows first and say so in CHANGELOG.
- Don't raise the Jev `batch_size` above ~20 (pg-jev measured accuracy loss). Laya over HTTP must stay at one
  row per request: Laya reads one 512–1,024-token state.
- Threads never touch database APIs; adapters call `engine.call` from the database's own thread.
