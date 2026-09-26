"""Spark / Databricks: the jev functions as pandas (Arrow) UDFs usable from SQL.

    import sqljev.spark
    sqljev.spark.register(spark)                                  # Laya in-process on each executor
    # or: sqljev.spark.register(spark, backend="gateway", api_url="https://.../v1/eval",
    #                           api_key=dbutils.secrets.get("sqljev", "token"))

    SELECT * FROM tickets WHERE jev(to_json(struct(*)), 'the customer is angry')

Spark passes Arrow batches (spark.sql.execution.arrow.maxRecordsPerBatch, 10,000 rows by default) to each
UDF; each batch becomes one engine.call(). One engine per Python worker keeps the model loaded and the
answer cache warm across batches. On a GPU cluster set device="cuda".
"""
_ENGINES = {}

_RETURNS = {"jev": "boolean", "jev_prob": "double", "jev_score": "double", "jev_score_norm": "double",
            "jev_choice": "string", "jev_confidence": "double", "jev_eval": "string"}
_ARITY = {"jev": 2, "jev_prob": 2, "jev_score": 3, "jev_score_norm": 3, "jev_choice": 3,
          "jev_confidence": 4, "jev_eval": 4}


def _engine(settings):
    from .core import Jev
    key = tuple(sorted(settings.items()))
    eng = _ENGINES.get(key)
    if eng is None:
        eng = _ENGINES[key] = Jev(**settings)
    return eng


def _make(name, settings):
    import json
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    n = _ARITY[name]

    def run(*cols):
        calls = list(zip(*(c.tolist() for c in cols)))
        out = _engine(settings).call(name, calls)
        if name == "jev_eval":
            out = [None if v is None else json.dumps(v) for v in out]
        return pd.Series(out, dtype="object" if name in ("jev_choice", "jev_eval", "jev") else "float64")

    # pandas_udf needs a fixed-arity signature with type hints.
    if n == 2:
        def f(a: pd.Series, b: pd.Series) -> pd.Series:
            return run(a, b)
    elif n == 3:
        def f(a: pd.Series, b: pd.Series, c: pd.Series) -> pd.Series:
            return run(a, b, c)
    else:
        def f(a: pd.Series, b: pd.Series, c: pd.Series, d: pd.Series) -> pd.Series:
            return run(a, b, c, d)
    return pandas_udf(f, _RETURNS[name])


def register(spark, prefix="", **settings):
    """Register jev, jev_prob, jev_score, jev_score_norm, jev_choice, jev_confidence, jev_eval for SQL.
    Settings (backend, api_url, api_key, model, device, ...) are shipped to the executors."""
    fns = {}
    for name in _RETURNS:
        fns[name] = spark.udf.register(prefix + name, _make(name, dict(settings)))
    return fns
