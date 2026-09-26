# Databricks / Spark

Install on the cluster (`%pip install "sqljev[laya]"`; GPU clusters: ML runtime with CUDA), then:

```python
import sqljev.spark
sqljev.spark.register(spark, device="cuda")          # Laya in-process on every executor
# or forward to a gateway:
# sqljev.spark.register(spark, backend="gateway", api_url="https://gateway.example.com/v1/eval",
#                       api_key=dbutils.secrets.get("sqljev", "token"))
```

```sql
SELECT * FROM support.tickets WHERE jev(to_json(struct(*)), 'the customer is angry');

SELECT jev_choice(to_json(struct(subject, body)), 'which team should handle this?',
                  array('billing', 'technical', 'sales')) AS team, count(*)
FROM support.tickets GROUP BY team;
```

Spark hands each UDF Arrow batches (`spark.sql.execution.arrow.maxRecordsPerBatch`, 10,000 rows); each batch is
one engine call, de-duplicated and cached per executor. Put cheap predicates in the same `WHERE` so fewer rows
reach `jev()`, and pass only the columns the judgment needs (`struct(subject, body)`).
