# Changelog

## 0.1.0 (unreleased)

First version.
- Engine with four backends: Laya in-process (default, `predict_batch`), sqljev gateway, laya-serve, TypeSafe Jev.
- Row de-duplication, LRU answer cache, spend guard, streaming evaluation with bounded read-ahead.
- SQL Server (`jev.judge` + lookup functions; `sqljev judge` for 2016–2022), Snowflake (SPCS service
  functions and vectorized UDF), Databricks/Spark (pandas UDFs), BigQuery (remote functions), Redshift (Lambda),
  DuckDB (Arrow UDFs), any SQLAlchemy database (CLI).
- PostgreSQL and MySQL/MariaDB (answers table + lookup functions, no extension needed) and one `sqljev judge`
  command for SQL Server, PostgreSQL, MySQL and MariaDB; the gateway also speaks Jev's `/v1/systemone`, so pg-jev
  runs on Laya.
- Fine-tuning: `sqljev dataset`, `sqljev finetune` (single GPU, `--train-layers` low-memory mode), `sqljev eval`,
  `sqljev publish` (Hugging Face Hub), and a Colab notebook (`notebooks/sqljev_finetune_colab.ipynb`).
- Benchmark: `python -m sqljev.demo` (100,000 rows, 10 domains, 13 questions with exact labels) and `bench/run.py`.
- One-command releases: `scripts/release.sh 0.1.0`.
- Laya prompt: "Is it true that <condition>?" (11/12 vs 9/12 for "Is this true of this database row: ...", on a
  small labelled set of tickets and names, English checkpoint).
