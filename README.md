<p align="center">
  <img src="site/assets/rowl.svg" alt="Rowl, the sqljev owl, reading a table row" width="160">
</p>

# sqljev — ask your SQL rows questions in plain language, answered by Laya

Write the condition the way you would say it. Your database does the rest, on **SQL Server, PostgreSQL,
MySQL / MariaDB, Snowflake, Databricks, BigQuery, Redshift, DuckDB** and anything SQLAlchemy can reach.

```sql
-- Snowflake / Databricks / DuckDB / BigQuery / Redshift: the row goes in as JSON
SELECT * FROM tickets t WHERE jev(OBJECT_CONSTRUCT(t.*), 'the customer threatens to cancel');

SELECT subject, jev_prob(to_json(t), 'the customer is angry') AS p
FROM tickets t ORDER BY p DESC LIMIT 20;

SELECT jev_choice(to_json(t), 'which team should handle this?',
                  ['billing', 'technical', 'security', 'sales']) AS team, count(*)
FROM tickets t GROUP BY team;

-- SQL Server / Azure SQL
EXEC jev.judge N'dbo.tickets', N'the customer is angry';
SELECT * FROM dbo.tickets AS t
WHERE jev.prob((SELECT t.* FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), N'the customer is angry') >= 0.5;
```

Every row is judged by a **System One decision model**, a model that does not generate text but returns
calibrated probabilities for typed questions (yes/no, choice, score). By default that is
**[Laya](https://github.com/NandhaKishorM/laya)** (Convai Innovations, Apache 2.0, open weights), running on
*your* hardware: row data never leaves your network, costs nothing per token, and **you can fine-tune it on
your own tables**. TypeSafe's hosted [Jev](https://docs.typesafe.ai) is one setting away.

Inspired by, and partly ported from, [pg-jev](https://github.com/realZachi/pg-jev), which does this inside
PostgreSQL. sqljev takes the idea to every other database and swaps in an open model you can train.

## What it is for

The questions SQL cannot express, asked right inside the queries you already write:

| | Example |
| --- | --- |
| Support / CRM | `jev(t, 'the customer threatens to cancel')`, route with `jev_choice(... ['billing','technical','sales'])` |
| Risk & compliance | flag contracts, emails or chat logs: *"mentions a price guarantee"*, *"contains health information"* |
| Pharma & life sciences | *"the adverse event report mentions liver injury"*, classify free-text indications by therapeutic area |
| Data cleanup | *"this address is a business, not a residence"*, *"the product description is not in English"* |
| Triage & ranking | `ORDER BY jev_prob(...)` for the most urgent incidents, likeliest leads, riskiest claims |

`jev()` is an ordinary boolean function, so it composes with everything else in SQL: `AND created_at > ...`,
joins, `GROUP BY`, `LIMIT`. Keep arithmetic, dates and exact matches in SQL; let the model judge *meaning*.

## How it works

```
 your SQL ──► database UDF / procedure ──► sqljev engine ──► Laya (in-process, GPU/CPU)
             (rows arrive in batches)      dedupe · cache ·    or a gateway, laya-serve, or Jev
                                           forward-pass batching
```

A SQL question is *one question over many rows*, and everything is organised around that:

1. **Rows go in as compact JSON objects** (column → value). NULL columns are dropped, which matters for
   Laya's 512–1,024-token window. Pass only the columns the judgment needs (a view, `struct(subject, body)`).
2. **Every database already batches UDF calls.** Snowflake, BigQuery, Redshift, Spark and DuckDB hand
   over hundreds to thousands of rows per call, and SQL Server's `jev.judge` sends 500 at a time. Each batch
   becomes one engine call.
3. **De-duplicate, then cache.** Identical rows are judged once. Answers are cached by row content and
   question, so re-running, changing the threshold or sorting by probability is free, and only rows that
   changed are judged again.
4. **Shared forward passes.** Misses go to Laya's `predict_batch`, which packs many rows into one forward
   pass (`batch_size`, default 64), sorted by length to minimise padding. Measured on CPU (English
   checkpoint): 0.19 s/row batched vs 0.35 s/row one at a time. On a GPU, Laya is ~33 ms for a single
   decision and ~7 ms per decision batched.
5. **Streaming when it pays.** `sqljev query --where ... --limit N` judges rows in order with a bounded
   read-ahead, so it stops after the first N matches instead of scanning everything.

### Backends

| `backend` | Model | Rows per call | Use it for |
| --- | --- | --- | --- |
| `local` (default) | Laya in this process | `batch_size` per forward pass (64) | CLI, DuckDB, Spark/Databricks executors, the gateway itself |
| `gateway` | a `sqljev gateway` (which runs Laya) | 256 per HTTP request | SQL Server, Snowflake UDFs, Redshift Lambda, anything remote |
| `laya-serve` | stock `laya-serve` (`POST /v1/systemone`) | 1 (Laya reads one state) | an existing Laya deployment |
| `jev` | TypeSafe Jev, hosted | 20 in one shared state (pg-jev's measured optimum) | best zero-shot accuracy, if data may leave your network |

## Benchmark: 100,000 rows, 13 questions

Ten synthetic but realistic tables with exact labels (`python -m sqljev.demo`), one plain SQL query per question
through DuckDB, answered by the **base Laya English checkpoint with no training**, on an RTX 4090 Laptop GPU
that another model was sharing:

| Question | Rows | Accuracy | Always-majority baseline | Rows/s |
| --- | --- | --- | --- | --- |
| Contract clause: which type? (5) | 10000 | **0.995** | 0.203 | 1606 |
| Job post: fully remote? | 10000 | **0.922** | 0.602 | 375 |
| Support ticket: is the customer angry? | 10000 | **0.894** | 0.697 | 567 |
| Advisor email: guarantees returns? | 10000 | **0.865** | 0.747 | 514 |
| Support ticket: which team? | 10000 | **0.854** | 0.352 | 580 |
| Job post: how senior? (3 levels) | 10000 | **0.853** | 0.336 | 402 |
| Expense: which category? (5) | 10000 | **0.851** | 0.209 | 607 |
| Product review: reports a defect? | 10000 | **0.850** | 0.702 | 855 |
| Product review: sentiment (3 levels) | 10000 | **0.828** | 0.453 | 879 |
| Adverse event: was it serious? | 10000 | **0.699** | 0.650 | 345 |
| Two company records: same company? (join) | 20000 | **0.685** | 0.506 | 531 |
| Adverse event: which body system? (5) | 10000 | **0.680** | 0.203 | 371 |
| Rental listing: pets allowed? | 10000 | **0.595** | 0.546 | 384 |

**140,000 decisions in 271 s (516/s). Re-running all 13 queries: 0.9 s**, every answer from the cache. With an
instant stand-in model, sqljev's own overhead measured about 98,000 decisions/s: the model is the only cost.
Reproduce with `python -m sqljev.demo --out bench/bench.duckdb && python bench/run.py --device cuda`.

The weak rows are where fine-tuning pays: the model has to learn *your* definition (what counts as "serious"
in pharmacovigilance, what a pet policy sentence means). See below.

## Which model? Laya vs Jev, honestly

| | Laya (default) | Jev |
| --- | --- | --- |
| License / weights | Apache 2.0, open weights | proprietary API |
| Where it runs | your server, GPU or CPU | TypeSafe's cloud |
| Cost | $0 per token | $0.042 / 1M input tokens |
| Latency | ~33 ms / decision (T4) | ~250 ms / request |
| **Zero-shot** accuracy (typed-decisions) | 0.362 (base checkpoint) | **0.727** |
| **Fine-tuned** accuracy (typed-decisions) | **0.766** | not fine-tunable |
| Wide option sets (Banking77) | 0.425 | **0.870** |

*(Figures from Laya's published benchmarks.)* Laya out of the box is good at clear-cut yes/no conditions
and weaker at fine-grained choices. **Its strength is that you can train it on your data**, and SQL tables
are full of labels. So sqljev ships the loop:

## Fine-tune Laya on your own tables

```bash
# 1. Labelled rows -> training/eval data. The label column is never shown to the model, and the state and
#    question are built by the same code the runtime uses, so the checkpoint learns exactly what it will be asked.
sqljev dataset "$DB_URL" "SELECT subject, body, team FROM tickets WHERE team IS NOT NULL" \
    --label team --choice "which team should handle this?" --test-fraction 0.2 -o tickets.jsonl
#    -> tickets.train.jsonl, tickets.test.jsonl   (--format typed-decisions for Laya's fine-tuning notebook)

# 2. Baseline: how does the base model (or Jev) do on YOUR rows?
sqljev eval tickets.test.jsonl                         # Laya base
sqljev eval tickets.test.jsonl --backend jev           # Jev, for comparison

# 3. Fine-tune (one GPU; a free Colab T4 is enough. --train-layers 12 for GPUs with less than ~10 GB free)
sqljev finetune tickets.train.jsonl --out checkpoints/tickets --epochs 3

# 4. Measure again on the same held-out rows, then share it with the team
sqljev eval tickets.test.jsonl --model checkpoints/tickets --min-accuracy 0.85
HF_TOKEN=... sqljev publish checkpoints/tickets --repo your-org/laya-tickets
SQLJEV_MODEL=your-org/laya-tickets sqljev gateway --host 0.0.0.0      # every database now uses it
```

Measured on the built-in pharma demo (*"the adverse event was serious"*, 3,000 training rows, 1,000 held out):
**69.4% → 100% after 2 minutes** of training on an RTX 4090 Laptop GPU (`--train-layers 12`, 3 epochs). The demo
reports come from templates and are easy to learn; expect a smaller jump on real data, and measure it the same way.

**No terminal?** Open [`notebooks/sqljev_finetune_colab.ipynb`](notebooks/sqljev_finetune_colab.ipynb) in
[Colab](https://colab.research.google.com/github/singhpratech/sqljev/blob/main/notebooks/sqljev_finetune_colab.ipynb),
pick a question (or the built-in pharma demo) and press *Run all*: it measures, fine-tunes, measures again and
publishes. The training loop follows Laya's own fine-tuning notebook (proper-scoring-rule policy gradient plus
cross-entropy, temperature calibration on held-out rows), adapted to a single GPU.

`laya-evals run tickets.test.jsonl` works on the same files for calibration (ECE) and per-slice reports.

## Install

```bash
pip install "sqljev[laya,db]"     # engine + Laya + SQLAlchemy CLI  (Python 3.10+)
pip install "sqljev"              # engine, gateway client, Lambda handler only: stdlib, no dependencies
```

For a GPU, install the matching PyTorch build first. The first use downloads the Laya checkpoint (~1.7 GB)
from Hugging Face.

## Per database

<details open><summary><b>Any database: the CLI</b> (Oracle, Db2, Trino, and everything above)</summary>

```bash
sqljev query "$DB_URL" "SELECT * FROM tickets" --where "the customer is angry" --limit 20
sqljev query "$DB_URL" "SELECT * FROM tickets" --rank "the customer is angry" --limit 10 --format csv
sqljev query "$DB_URL" "SELECT * FROM tickets" --choice "which team?" --options billing,technical,sales
sqljev query "$DB_URL" "SELECT * FROM tickets" --prob "mentions a refund" --columns subject,body
# Write judgments back as a table you can join in any database:
sqljev materialize "$DB_URL" "SELECT id, subject, body FROM tickets" --key id \
    --prob "the customer is angry" --into ticket_anger
```
</details>

<details><summary><b>SQL Server / Azure SQL</b></summary>

Run [`sql/sqlserver/install.sql`](sql/sqlserver/install.sql). It creates schema `jev` with `jev.judge`,
`jev.prob`, `jev.matches`, `jev.choice`, `jev.score`, `jev.score_norm`, `jev.eval`, `jev.answers_for`
(set-based join) and `jev.forget`.

- **SQL Server 2025 / Azure SQL:** `jev.judge` calls a `sqljev gateway` through
  `sp_invoke_external_rest_endpoint` (HTTPS on 443, publicly trusted certificate). See the header of the script.
- **SQL Server 2016–2022**, or no outbound HTTPS: fill the same answer table from outside, and every function
  works the same:
  ```bash
  sqljev judge "mssql+pymssql://user:pw@host/db" --source dbo.tickets --prob "the customer is angry"
  ```

Row JSON (`FOR JSON`) and hashes are computed by SQL Server itself, so a row edited later is judged again
on the next `jev.judge` run, and unchanged rows are skipped.
</details>

<details><summary><b>PostgreSQL</b> (RDS, Aurora, Cloud SQL, AlloyDB, Azure, Supabase, Neon, ...)</summary>

Run [`sql/postgres/install.sql`](sql/postgres/install.sql): no extension, no superuser. Judge once, then read
with ordinary functions:

```bash
sqljev judge "postgresql://..." --source public.tickets --prob "the customer is angry"
```
```sql
SELECT * FROM tickets t WHERE jev.prob(to_jsonb(t), 'the customer is angry') >= 0.5;
SELECT jev.choice(to_jsonb(t), 'which team?', ARRAY['billing', 'technical', 'sales']) FROM tickets t;
```

Rows are keyed by their `jsonb` content, so edited rows are judged again on the next run. Self-hosted Postgres
with `plpython3u` can also run [pg-jev](https://github.com/realZachi/pg-jev) on Laya:
`SET jev.api_url = 'http://<sqljev gateway>:8765/v1/systemone'`.
</details>

<details><summary><b>MySQL / MariaDB</b> (RDS, Aurora, Cloud SQL, Azure)</summary>

Run [`sql/mysql/install.sql`](sql/mysql/install.sql) (stored functions and a `jev_answers` table), then:

```bash
sqljev judge "mysql+pymysql://..." --source tickets --prob "the customer is angry"
# prints the exact row expression to use, e.g. JSON_OBJECT('id', t.id, 'subject', t.subject, 'body', t.body)
```
```sql
SELECT * FROM tickets t
WHERE jev_prob(JSON_OBJECT('id', t.id, 'subject', t.subject, 'body', t.body), 'the customer is angry') >= 0.5;
```
</details>

<details><summary><b>Snowflake</b></summary>

- [`sql/snowflake/install_spcs.sql`](sql/snowflake/install_spcs.sql): **Laya inside Snowflake** on Snowpark
  Container Services (GPU). Rows never leave your account; service functions batch up to 1,000 rows per call.
- [`sql/snowflake/install_udf.sql`](sql/snowflake/install_udf.sql): a vectorized Python UDF (engine inlined)
  calling your gateway or Jev through an External Access Integration.

```sql
SELECT * FROM tickets t WHERE jev(OBJECT_CONSTRUCT(t.*), 'the customer is angry');
```
</details>

<details><summary><b>Databricks / Spark</b></summary>

```python
import sqljev.spark
sqljev.spark.register(spark, device="cuda")      # Laya on every executor (or backend="gateway")
```
```sql
SELECT * FROM tickets WHERE jev(to_json(struct(subject, body)), 'the customer is angry');
```
See [`sql/databricks/README.md`](sql/databricks/README.md).
</details>

<details><summary><b>BigQuery</b></summary>

Remote functions backed by the gateway on Cloud Run (GPU optional):
[`sql/bigquery/install.sql`](sql/bigquery/install.sql), [`deploy/cloudrun`](deploy/cloudrun/README.md).
```sql
SELECT * FROM `proj.support.tickets` t WHERE jev.jev(TO_JSON_STRING(t), 'the customer is angry');
```
</details>

<details><summary><b>Amazon Redshift</b></summary>

Lambda UDFs (`sqljev.aws_lambda.handler`, stdlib-only zip) forwarding to a gateway:
[`sql/redshift/install.sql`](sql/redshift/install.sql), [`deploy/lambda`](deploy/lambda/README.md).
</details>

<details><summary><b>DuckDB</b></summary>

```python
import duckdb, sqljev.duckdb
con = duckdb.connect("support.duckdb"); sqljev.duckdb.register(con)
con.sql("SELECT * FROM tickets t WHERE jev(to_json(t), 'the customer is angry')")
```
</details>

## The gateway

One small HTTP service that speaks every database's batch protocol and runs Laya in-process:

```bash
sqljev gateway --host 0.0.0.0 --port 8765                 # SQLJEV_GATEWAY_TOKEN=... to require a token
docker build -f deploy/Dockerfile -t sqljev .        # checkpoint baked in; CUDA via --build-arg TORCH_INDEX=...
```

| Route | Caller |
| --- | --- |
| `POST /v1/eval` | sqljev clients (`backend=gateway`), Snowflake UDF |
| `POST /sqlserver` | `jev.judge` via `sp_invoke_external_rest_endpoint` |
| `POST /snowflake/<fn>` | Snowflake service / external functions |
| `POST /bigquery` | BigQuery remote functions |
| `POST /redshift` | Redshift Lambda payloads |
| `POST /v1/systemone` | Jev clients such as pg-jev (Jev's own API, answered by Laya) |
| `GET /health`, `GET /stats` | probes, cache/throughput stats |

## Functions

Identical names everywhere (SQL Server uses the `jev.` schema: `jev.prob`, `jev.matches`, ...).

| Function | Returns | Purpose |
| --- | --- | --- |
| `jev(row, condition [, threshold])` | boolean | `WHERE` predicate; threshold defaults to 0.5 |
| `jev_prob(row, condition)` | float | probability 0..1 that the row satisfies the condition |
| `jev_score(row, question, levels)` | float | probability-weighted position on ordered levels (0..n-1) |
| `jev_score_norm(row, question, levels)` | float | same, normalised to 0..1 |
| `jev_choice(row, question, options)` | text | the most likely option |
| `jev_confidence(row, question, kind, options)` | float | confidence of a choice / score answer |
| `jev_eval(row, question, kind, options)` | json | the full answer: probabilities, confidence |

## Settings

Environment variables `SQLJEV_<NAME>`, keyword arguments (`Jev(backend="gateway")`,
`register(con, device="cuda")`) or CLI flags.

| Setting | Default | Meaning |
| --- | --- | --- |
| `backend` | `local` | `local`, `gateway`, `laya-serve`, `jev` |
| `model` | Laya router | `english`, `multilingual`, `typed-decisions`, a fine-tuned checkpoint path / Hub id, or a Jev model id |
| `device` | auto | `cpu`, `cuda`, `mps` (local) |
| `api_url` | per backend | gateway / laya-serve / Jev endpoint |
| `api_key` | env | Jev: `TYPESAFE_API_KEY`; gateway: `SQLJEV_GATEWAY_TOKEN`; laya-serve: `LAYA_API_KEY` |
| `batch_size` | 64 / 256 / 1 / 20 | rows per forward pass (local) or per request |
| `concurrency` | 1 / 4 / 4 / 16 | parallel requests |
| `threshold` | 0.5 | for `jev()` |
| `max_len` | checkpoint | Laya token budget per row (multilingual reads up to 8,192) |
| `drop_nulls` | true | omit NULL columns from what the model sees |
| `cache_size` | 200,000 | answers kept in memory (LRU); 0 disables |
| `max_rows` / `max_chars` | 0 (off) | spend guard: refuse to send more than this |
| `timeout` / `keepalive` | 60 / 600 s | HTTP backends |

## Caveats

- **Limit before you judge.** Databases compute the `SELECT` list before `ORDER BY ... LIMIT`, so
  `SELECT jev_prob(...) FROM t ORDER BY id LIMIT 100` judges *every* row. Limit in a subquery first:
  `SELECT jev_prob(...) FROM (SELECT * FROM t ORDER BY id LIMIT 100) t`. In `WHERE`, put cheap predicates first.
- **Every row that reaches `jev()` is judged** (once, thanks to the cache). No index can answer a
  plain-language condition. Filter with cheap SQL first; `LIMIT` and streaming stop early.
- **Laya reads 512 tokens** (English) or 1,024 (multilingual, up to 8,192 with `max_len`). Wide rows are cut
  off, so send only the columns that matter.
- **Zero-shot Laya is not Jev.** Measure on your data with `sqljev eval` before trusting a threshold, and
  fine-tune when it matters.
- With `backend=jev`, row contents go to a third-party API. Don't use it on data you may not share.
- Answers can change between checkpoints. Pin `model` when results feed reports.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                               # never calls a real model: tests/mock_api.py + a fake laya module
python scripts/build_sql.py          # regenerate sql/snowflake/install_udf.sql after editing core.py
```

Releasing is one command, `scripts/release.sh 0.1.0`: it dates the CHANGELOG entry, runs the tests, tags and
pushes. GitHub Actions then builds the package, creates the GitHub Release with the wheel and every database's
install scripts attached, and publishes to PyPI once `PYPI_PUBLISH` is enabled.

See [AGENTS.md](AGENTS.md) for the architecture.

## License

Apache 2.0. Parts ported from [pg-jev](https://github.com/realZachi/pg-jev) (PostgreSQL License); see
[NOTICE](NOTICE). Laya is by Convai Innovations (Apache 2.0). Jev and TypeSafe are trademarks of their
respective owners; this project is not affiliated with TypeSafe or Convai Innovations.
