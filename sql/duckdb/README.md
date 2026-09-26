# DuckDB

```python
import duckdb, sqljev.duckdb
con = duckdb.connect("support.duckdb")
sqljev.duckdb.register(con)                   # Laya in-process; or register(con, backend="gateway", api_url=...)
con.sql("SELECT * FROM tickets t WHERE jev(to_json(t), 'the customer is angry')").show()
con.sql("""SELECT jev_choice(to_json(t), 'which team?', ['billing', 'technical', 'sales']) AS team, count(*)
           FROM tickets t GROUP BY team""").show()
```
