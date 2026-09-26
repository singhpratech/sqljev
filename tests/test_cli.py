import json
import sqlite3

import pytest

from sqljev.cli import main

pytest.importorskip("sqlalchemy")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, body TEXT, team TEXT, upset INTEGER)")
    con.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?)",
                    [(i, "I am angry" if i % 3 == 0 else "fine thanks", ["billing", "tech"][i % 2], int(i % 3 == 0))
                     for i in range(30)])
    con.commit()
    return "sqlite:///%s" % path


def run(args, capsys):
    main(args)
    return capsys.readouterr()


def jev_args(mock_url):
    return ["--backend", "jev", "--api-url", mock_url]


def test_query_where_streams_with_limit(db, mock_url, capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    out = run(["query", db, "SELECT * FROM tickets", "--where", "is angry", "--limit", "3", "--format", "jsonl",
               *jev_args(mock_url)], capsys).out
    rows = [json.loads(l) for l in out.splitlines()]
    assert [r["id"] for r in rows] == [0, 3, 6] and rows[0]["jev_prob"] == 0.9


def test_query_rank_choice_and_columns(db, mock_url, capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    out = run(["query", db, "SELECT * FROM tickets", "--rank", "is angry", "--limit", "2", "--format", "csv",
               *jev_args(mock_url)], capsys).out
    assert out.splitlines()[0] == "id,body,team,upset,jev_prob" and len(out.splitlines()) == 3
    # the model only sees `team`: "angry" never appears in what is judged
    out = run(["query", db, "SELECT * FROM tickets", "--prob", "is angry", "--columns", "team", "--format", "jsonl",
               *jev_args(mock_url)], capsys).out
    assert {json.loads(l)["jev_prob"] for l in out.splitlines()} == {0.1}
    out = run(["query", db, "SELECT * FROM tickets LIMIT 4", "--choice", "which team?", "--options", "billing,tech",
               *jev_args(mock_url)], capsys).out
    assert "jev_choice" in out


def test_materialize(db, mock_url, capsys, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    run(["materialize", db, "SELECT id, body FROM tickets", "--key", "id", "--prob", "is angry", "--into", "t_angry",
         *jev_args(mock_url)], capsys)
    con = sqlite3.connect(db[len("sqlite:///"):])
    n = con.execute("SELECT count(*) FROM tickets t JOIN t_angry a ON a.id = t.id WHERE a.jev_prob > 0.5").fetchone()
    assert n == (10,)
    run(["materialize", db, "SELECT id, body FROM tickets", "--key", "id", "--prob", "is angry", "--into", "t_angry",
         "--replace", *jev_args(mock_url)], capsys)


def test_dataset_and_eval(db, mock_url, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    out = str(tmp_path / "angry.jsonl")
    run(["dataset", db, "SELECT id, body, upset FROM tickets", "--label", "upset", "--noul", "is angry",
         "-o", out, "--test-fraction", "0.3"], capsys)
    train = [json.loads(l) for l in open(out[:-6] + ".train.jsonl")]
    test = [json.loads(l) for l in open(out[:-6] + ".test.jsonl")]
    assert len(train) + len(test) == 30 and test
    assert "upset" not in train[0]["state"]                            # the label never leaks into the state
    assert train[0]["questions"]["q"]["instructions"] == "Is it true that is angry?"
    res = run(["eval", out[:-6] + ".train.jsonl", *jev_args(mock_url)], capsys).out
    assert json.loads(res)["accuracy"] == 1.0
    # choice: options come from the label column; typed-decisions format for the fine-tuning notebook
    out2 = str(tmp_path / "team.jsonl")
    run(["dataset", db, "SELECT id, body, team FROM tickets", "--label", "team", "--choice", "which team?",
         "-o", out2, "--format", "typed-decisions"], capsys)
    rec = json.loads(open(out2).readline())
    assert json.loads(rec["gold"])["q"]["label"] in ("billing", "tech")
    assert json.loads(rec["questions"])["q"]["criteria"] == {"billing": "billing", "tech": "tech"}


def test_errors_exit_2(db, capsys):
    with pytest.raises(SystemExit) as e:
        main(["query", db, "SELECT * FROM tickets", "--choice", "x", "--options", "only"])
    assert e.value.code == 2
