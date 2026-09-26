# Changelog

## 0.1.0 (unreleased)

First version.
- Engine with four backends: Laya in-process (default, `predict_batch`), sql-jev-laya gateway, laya-serve, TypeSafe Jev.
- Row de-duplication, LRU answer cache, spend guard, streaming evaluation with bounded read-ahead.
- SQL Server (`jev.judge` + lookup functions; `sql-jev-laya judge-sqlserver` for 2016–2022), Snowflake (SPCS service
  functions and vectorized UDF), Databricks/Spark (pandas UDFs), BigQuery (remote functions), Redshift (Lambda),
  DuckDB (Arrow UDFs), any SQLAlchemy database (CLI).
- Fine-tuning loop: `sql-jev-laya dataset` (laya-evals / typed-decisions JSONL) and `sql-jev-laya eval`.
- Laya prompt: "Is it true that <condition>?" (11/12 vs 9/12 for "Is this true of this database row: ...", on a
  small labelled set of tickets and names, English checkpoint).
