import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")


def test_duckdb_functions(jev_mock):
    import sqljev.duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE tickets AS SELECT * FROM (VALUES (1, 'I am angry'), (2, 'all calm'), (3, NULL)) t(id, body)")
    eng = sqljev.duckdb.register(con, engine=jev_mock)
    got = con.sql("SELECT id FROM tickets t WHERE jev(to_json(t), 'is angry') ORDER BY id").fetchall()
    assert got == [(1,)]
    probs = con.sql("SELECT id, jev_prob(to_json(t), 'is angry') FROM tickets t ORDER BY id").fetchall()
    assert probs == [(1, 0.9), (2, 0.1), (3, 0.1)]
    team = con.sql("SELECT jev_choice(to_json(t), 'team', ['a', 'b']) FROM tickets t").fetchall()
    assert {r[0] for r in team} <= {"a", "b"}
    norm = con.sql("SELECT jev_score_norm(to_json(t), 'how angry', ['lo','mid','hi']) FROM tickets t").fetchall()
    assert all(0 <= r[0] <= 1 for r in norm)
    assert con.sql("SELECT jev_prob(NULL, 'x')").fetchone() == (None,)
    assert eng.stats()["cache_hits"] >= 3
