"""DuckDB: register the jev functions on a connection as vectorized (Arrow) UDFs.

    import duckdb, sqljev.duckdb
    con = duckdb.connect()
    sqljev.duckdb.register(con)                       # SQLJEV_* settings, or register(con, backend="gateway")
    con.sql("SELECT * FROM tickets t WHERE jev(to_json(t), 'the customer is angry')")

DuckDB hands each UDF a whole vector (2,048 rows), so every call is one engine.call() over the chunk.
"""
from .core import Jev

_SIGNATURES = {
    # name: (parameter types, return type)
    "jev": (["VARCHAR", "VARCHAR"], "BOOLEAN"),
    "jev_prob": (["VARCHAR", "VARCHAR"], "DOUBLE"),
    "jev_score": (["VARCHAR", "VARCHAR", "VARCHAR[]"], "DOUBLE"),
    "jev_score_norm": (["VARCHAR", "VARCHAR", "VARCHAR[]"], "DOUBLE"),
    "jev_choice": (["VARCHAR", "VARCHAR", "VARCHAR[]"], "VARCHAR"),
    "jev_confidence": (["VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR[]"], "DOUBLE"),
    "jev_eval": (["VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR[]"], "VARCHAR"),
}


def register(con, engine=None, prefix="", **settings):
    """Create jev, jev_prob, jev_score, jev_score_norm, jev_choice, jev_confidence, jev_eval on `con`.
    Returns the engine, whose .stats() reports requests, cache hits and forward passes."""
    import json
    import pyarrow as pa
    from duckdb import sqltype

    eng = engine or Jev(**settings)

    def make(name, ret, arity):
        def run(*cols):
            calls = list(zip(*(c.to_pylist() for c in cols)))
            out = eng.call(name, calls)
            if name == "jev_eval":
                out = [None if v is None else json.dumps(v) for v in out]
            return pa.array(out, type=ret)
        # DuckDB checks the Python signature, so each UDF needs a fixed arity.
        if arity == 2:
            return lambda a, b: run(a, b)
        if arity == 3:
            return lambda a, b, c: run(a, b, c)
        return lambda a, b, c, d: run(a, b, c, d)

    arrow_types = {"BOOLEAN": pa.bool_(), "DOUBLE": pa.float64(), "VARCHAR": pa.string()}
    for name, (params, ret) in _SIGNATURES.items():
        full = prefix + name
        try:
            con.remove_function(full)
        except Exception:      # noqa: BLE001 -- not registered yet
            pass
        con.create_function(full, make(name, arrow_types[ret], len(params)), [sqltype(p) for p in params], sqltype(ret),
                            type="arrow", null_handling="special", side_effects=False)
    return eng
