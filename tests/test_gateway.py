import json
import threading
import urllib.error
import urllib.request

import pytest

from sqljev import Jev
from sqljev.aws_lambda import handler
from sqljev.gateway import make_server


@pytest.fixture
def gateway(jev_mock):
    srv = make_server("127.0.0.1", 0, engine=jev_mock, token="gw-secret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def post(url, body, token="gw-secret"):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_health_and_auth(gateway):
    assert json.loads(urllib.request.urlopen(gateway + "/health").read())["status"] == "ok"
    with pytest.raises(urllib.error.HTTPError) as e:
        post(gateway + "/v1/eval", {"question": "x", "rows": []}, token="wrong")
    assert e.value.code == 401


def test_native_and_sqlserver(gateway):
    body = {"question": "is angry", "rows": [{"b": "angry"}, {"b": "calm"}]}
    assert [a["noul"] for a in post(gateway + "/v1/eval", body)["answers"]] == [0.9, 0.1]
    body = {"question": "team", "kind": "choice", "options": ["x", "y"], "rows": [{"b": "angry"}]}
    assert post(gateway + "/sqlserver", body)["answers"][0]["choice"] in ("x", "y")


def test_bigquery(gateway):
    body = {"userDefinedContext": {"fn": "jev"},
            "calls": [['{"b":"angry"}', "is angry"], ['{"b":"calm"}', "is angry"], [None, "is angry"]]}
    assert post(gateway + "/bigquery", body) == {"replies": [True, False, None]}
    body = {"userDefinedContext": {"fn": "jev_eval"}, "calls": [['{"b":"angry"}', "is angry", "noul", None]]}
    assert json.loads(post(gateway + "/bigquery", body)["replies"][0])["noul"] == 0.9
    with pytest.raises(urllib.error.HTTPError) as e:
        post(gateway + "/bigquery", {"calls": [["{}", "x"]]})
    assert e.value.code == 400


def test_snowflake(gateway):
    body = {"data": [[0, {"b": "angry"}, "is angry"], [1, {"b": "calm"}, "is angry", 0.05]]}
    assert post(gateway + "/snowflake/jev", body) == {"data": [[0, True], [1, True]]}
    body = {"data": [[0, {"b": "angry"}, "is angry", "noul", None]]}
    assert post(gateway + "/snowflake/jev_eval", body)["data"][0][1]["noul"] == 0.9


def test_redshift(gateway, jev_mock, monkeypatch):
    event = {"external_function": "public.jev_prob", "num_records": 2,
             "arguments": [['{"b":"angry"}', "is angry"], ['{"b":"calm"}', "is angry"]]}
    assert post(gateway + "/redshift", event) == {"success": True, "num_records": 2, "results": [0.9, 0.1]}
    import sqljev.core
    monkeypatch.setattr(sqljev.core, "_DEFAULT", jev_mock)
    assert json.loads(handler(event))["results"] == [0.9, 0.1]
    assert json.loads(handler({"external_function": "nope", "arguments": []}))["success"] is False


def test_gateway_backend_client(gateway, requests_made):
    j = Jev(backend="gateway", api_url=gateway + "/v1/eval", api_key="gw-secret", batch_size=100)
    rows = [{"i": i, "b": "angry" if i % 2 else "calm"} for i in range(250)]
    probs = j.prob(rows, "is angry")
    assert probs[:2] == [0.1, 0.9] and j.stats()["requests"] == 3


def test_systemone_jev_compatible(gateway, mock_url):
    """A Jev client (sqljev's own jev backend speaks pg-jev's batch format) runs unchanged against the gateway."""
    j = Jev(backend="jev", api_url=gateway + "/v1/systemone", api_key="gw-secret", batch_size=20)
    rows = [{"i": i, "b": "angry" if i % 2 else "calm"} for i in range(30)]
    assert j.prob(rows, "is angry")[:3] == [0.1, 0.9, 0.1]
    ch = j.choice(rows[:5], "which team?", ["a", "b"])
    assert set(ch) <= {"a", "b"}
    sc = j.score(rows[:5], "how angry", ["lo", "mid", "hi"])
    assert all(0 <= v <= 2 for v in sc)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(gateway + "/v1/systemone", {"state": "hello", "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert e.value.code == 400
