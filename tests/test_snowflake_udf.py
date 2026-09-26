"""Runs the Python body of the generated Snowflake UDF with a fake `_snowflake` module."""
import pathlib
import re
import sys
import types

import pytest

pd = pytest.importorskip("pandas")
ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_generated_file_is_fresh():
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_sql
    for path, text in build_sql.render().items():
        assert path.read_text() == text, "run python scripts/build_sql.py"


def test_udf_body(monkeypatch, gateway_for_snowflake):
    sql = (ROOT / "sql/snowflake/install_udf.sql").read_text()
    body = re.search(r"CREATE OR REPLACE FUNCTION jev_eval.*?AS \$\$\n(.*?)\n\$\$;", sql, re.S).group(1)
    body = body.replace('"https://gateway.example.com/v1/eval"', repr(gateway_for_snowflake))
    fake = types.ModuleType("_snowflake")
    fake.vectorized = lambda **kw: (lambda f: f)
    fake.get_generic_secret_string = lambda name: "gw-secret"
    monkeypatch.setitem(sys.modules, "_snowflake", fake)
    ns = {"__name__": "snowflake_udf"}
    exec(compile(body, "install_udf.sql", "exec"), ns)
    df = pd.DataFrame([[{"b": "angry"}, "is angry", "noul", None],
                       ['{"b": "calm"}', "is angry", None, None],
                       [{"b": "x"}, "team", "choice", ["a", "b"]]])
    out = ns["sqljev_eval_batch"](df)
    assert out[0]["noul"] == 0.9 and out[1]["noul"] == 0.1 and out[2]["choice"] in ("a", "b")


@pytest.fixture
def gateway_for_snowflake(jev_mock):
    import threading
    from sqljev.gateway import make_server
    srv = make_server("127.0.0.1", 0, engine=jev_mock, token="gw-secret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/v1/eval" % srv.server_address[1]
    srv.shutdown()
