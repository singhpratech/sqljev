import pytest

from sqljev import Jev, JevError, laya_question, make_config, to_row_json
from sqljev.core import parse_options

TICKETS = [{"id": i, "body": "customer is angry" if i % 3 == 0 else "all good", "note": None} for i in range(45)]


def test_config_defaults_per_backend():
    assert make_config()["backend"] == "local"
    assert make_config(backend="jev")["batch_size"] == 20
    assert make_config(backend="laya-serve", batch_size=50)["batch_size"] == 1   # one state per request
    assert make_config(env={"SQLJEV_BACKEND": "gateway", "SQLJEV_CONCURRENCY": "7"})["concurrency"] == 7
    with pytest.raises(JevError):
        make_config(backend="nope")
    with pytest.raises(JevError):
        make_config(bogus=1)


def test_row_serialisation_is_compact_and_drops_nulls():
    assert to_row_json({"a": 1, "b": None}) == '{"a":1}'
    assert to_row_json('{"a": 1, "b": null}') == '{"a":1}'
    assert to_row_json({"a": None}, drop_nulls=False) == '{"a":null}'
    assert to_row_json("plain text") == '"plain text"'


def test_laya_question_wording():
    assert laya_question("noul", "the customer is angry.", None)["instructions"] == \
        "Is it true that the customer is angry?"
    assert laya_question("noul", "Is the customer angry?", None)["instructions"] == "Is the customer angry?"
    assert laya_question("choice", "team?", ["a", "b"])["criteria"] == {"a": "a", "b": "b"}


def test_parse_options():
    assert parse_options("a, b,c") == ["a", "b", "c"]
    assert parse_options('["x","y"]') == ["x", "y"]
    assert parse_options(None) is None


def test_local_backend_batches_dedupes_and_caches(fake_laya):
    j = Jev(batch_size=16)
    probs = j.prob(TICKETS, "the customer is angry")
    assert probs[0] == 0.9 and probs[1] == 0.1
    # null column dropped -> rows differ only by id, so 45 distinct rows in ceil(45/16) = 3 forward batches
    assert fake_laya.batches == [16, 16, 13]
    again = j.prob(TICKETS + TICKETS[:5], "the customer is angry")
    assert again[:45] == probs and fake_laya.batches == [16, 16, 13]
    s = j.stats()
    assert s["rows_evaluated"] == 45 and s["cache_hits"] == 45 and s["duplicates"] == 5


def test_local_backend_finetuned_checkpoint(fake_laya):
    j = Jev(model="/models/my-tickets")
    assert j.where([{"t": "angry"}, {"t": "calm"}], "the customer is angry") == [True, False]


def test_jev_backend_packs_20_rows_per_request(jev_mock, requests_made):
    before = requests_made()
    probs = jev_mock.prob(TICKETS, "the customer is angry")
    assert probs[:4] == [0.9, 0.1, 0.1, 0.9]
    assert requests_made() - before == 3
    s = jev_mock.stats()
    assert s["requests"] == 3 and s["input_tokens"] > 0 and s["estimated_cost_usd"] > 0


def test_laya_serve_backend_one_row_per_request(mock_url, requests_made):
    j = Jev(backend="laya-serve", api_url=mock_url)
    before = requests_made()
    assert j.prob(TICKETS[:5], "the customer is angry") == [0.9, 0.1, 0.1, 0.9, 0.1]
    assert requests_made() - before == 5


def test_call_dispatch(jev_mock):
    rows = [{"body": "angry"}, {"body": "calm"}, None]
    assert jev_mock.call("jev", [(r, "is angry") for r in rows]) == [True, False, None]
    assert jev_mock.call("jev", [(rows[0], "is angry", 0.95)]) == [False]
    levels = ["low", "mid", "high"]
    s = jev_mock.call("jev_score", [(r, "how angry", levels) for r in rows[:2]])
    n = jev_mock.call("public.JEV_SCORE_NORM", [(r, "how angry", levels) for r in rows[:2]])
    assert n == [v / 2 for v in s]
    c = jev_mock.call("jev_choice", [(rows[0], "team", '["billing","tech"]')])
    assert c[0] in ("billing", "tech")
    e = jev_mock.call("jev_eval", [(rows[0], "is angry")])
    assert e[0]["noul"] == 0.9
    conf = jev_mock.call("jev_confidence", [(rows[0], "team", "choice", ["billing", "tech"])])
    assert conf == [1.0]
    # one UDF batch may mix questions
    mixed = jev_mock.call("jev_prob", [(rows[0], "is angry"), (rows[0], "is calm"), (rows[1], "is calm")])
    assert mixed == [0.9, 0.1, 0.9]
    with pytest.raises(JevError):
        jev_mock.call("nope", [])
    with pytest.raises(JevError):
        jev_mock.call("jev_choice", [(rows[0], "team", ["only-one"])])


def test_errors(mock_url):
    with pytest.raises(JevError, match="422"):
        Jev(backend="jev", api_url=mock_url, api_key="test-key").prob([{"a": 1}], "trigger422 x")
    with pytest.raises(JevError, match="401"):
        Jev(backend="jev", api_url=mock_url, api_key="wrong").prob([{"a": 1}], "x")
    with pytest.raises(JevError, match="API key"):
        Jev(backend="jev", api_url=mock_url).prob([{"a": 1}], "x")


def test_retry_after_429(jev_mock):
    assert jev_mock.prob([{"a": "x"}], "trigger429 x") == [0.9]
    assert jev_mock.stats()["retries"] >= 1


def test_spend_guard(jev_mock):
    j = Jev(jev_mock.cfg, max_rows=30)
    with pytest.raises(JevError, match="max_rows"):
        j.prob(TICKETS, "is angry")
    j2 = Jev(jev_mock.cfg, max_rows=30)
    j2.prob(TICKETS[:20], "is angry")
    j2.prob(TICKETS[:20], "is angry")       # cached: nothing sent
    with pytest.raises(JevError):
        j2.prob(TICKETS[20:], "is angry")
    j2.reset_budget()
    j2.prob(TICKETS[20:40], "is angry")


def test_evaluate_iter_streams_in_order_and_stops_early(mock_url, requests_made):
    j = Jev(backend="jev", api_url=mock_url, api_key="test-key", concurrency=2)
    rows = [{"id": i, "body": "angry" if i % 2 else "calm"} for i in range(1000)]
    before = requests_made()
    got = []
    for row, ans in j.evaluate_iter(rows, "is angry"):
        got.append((row["id"], ans["noul"]))
        if len(got) == 3:
            break
    assert got == [(0, 0.1), (1, 0.9), (2, 0.1)]
    import time
    time.sleep(0.3)
    assert requests_made() - before <= 4          # 2 x concurrency window, not 50 requests
    full = list(j.evaluate_iter(rows[:55] + [None] + rows[:3], "is angry"))
    assert [r["id"] if r else None for r, _ in full] == list(range(55)) + [None, 0, 1, 2]
    assert full[55][1] is None and full[-1][1]["noul"] == 0.1


def test_evaluate_iter_key_selects_model_view(fake_laya):
    j = Jev()
    rows = [{"id": 1, "secret": "angry", "body": "calm"}]
    [(row, ans)] = j.evaluate_iter(rows, "is angry", key=lambda r: {"body": r["body"]})
    assert row is rows[0] and ans["noul"] == 0.1
