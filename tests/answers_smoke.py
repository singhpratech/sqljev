#!/usr/bin/env python3
"""PostgreSQL / MySQL / MariaDB smoke test: install sql/<db>/install.sql, judge a table with `sqljev judge`
against the mock model, and read the answers back through the lookup functions.

    SQLJEV_TEST_URL="postgresql+psycopg://postgres:pw@127.0.0.1:5432/postgres" python tests/answers_smoke.py
    SQLJEV_TEST_URL="mysql+pymysql://root:pw@127.0.0.1:3306/demo" python tests/answers_smoke.py
    SQLJEV_TEST_URL="mariadb+pymysql://root:pw@127.0.0.1:3306/demo" python tests/answers_smoke.py
"""
import os
import pathlib
import re
import sys
import threading
from http.server import ThreadingHTTPServer

import sqlalchemy as sa

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mock_api  # noqa: E402
from sqljev.cli import main  # noqa: E402

url = os.environ["SQLJEV_TEST_URL"]
eng = sa.create_engine(url, isolation_level="AUTOCOMMIT")
pg = eng.dialect.name == "postgresql"


def statements(script):
    if pg:
        return [script]                       # psycopg runs a multi-statement script in one call
    out, delim, buf = [], ";", []
    for line in script.splitlines():
        m = re.match(r"^\s*DELIMITER\s+(\S+)\s*$", line)
        if m:
            delim = m.group(1)
            continue
        buf.append(line)
        if line.rstrip().endswith(delim):
            stmt = "\n".join(buf).rstrip()[: -len(delim)]
            if stmt.strip() and not all(l.strip().startswith("--") or not l.strip() for l in stmt.splitlines()):
                out.append(stmt)
            buf = []
    return out


script = (HERE.parent / ("sql/postgres/install.sql" if pg else "sql/mysql/install.sql")).read_text()
with eng.connect() as c:
    for st in statements(script):
        c.exec_driver_sql(st)
    c.exec_driver_sql("DROP TABLE IF EXISTS tickets")
    c.exec_driver_sql("CREATE TABLE tickets (id int PRIMARY KEY, body varchar(200), note varchar(50) NULL)")
    c.exec_driver_sql("INSERT INTO tickets VALUES (1, 'I am angry', NULL), (2, 'all calm', 'vip'), (3, 'angry again', NULL)")
    if pg:
        c.exec_driver_sql("DELETE FROM jev.answers")
    else:
        c.exec_driver_sql("DELETE FROM jev_answers")

srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_api.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
api = "http://127.0.0.1:%d/v1/systemone" % srv.server_address[1]
os.environ["TYPESAFE_API_KEY"] = "test-key"
common = ["--backend", "jev", "--api-url", api]
main(["judge", url, "--source", "tickets", "--prob", "is angry", *common])
main(["judge", url, "--source", "tickets", "--prob", "is angry", *common])                  # nothing new
main(["judge", url, "--source", "tickets", "--choice", "team", "--options", '["a", "b"]', *common])

if pg:
    ROW, P, M, C = "to_jsonb(t)", "jev.prob", "jev.matches(to_jsonb(t), 'is angry')", "jev.choice(to_jsonb(t), 'team', ARRAY['a','b'])"
else:
    ROW = "JSON_OBJECT('id', t.id, 'body', t.body, 'note', t.note)"
    P, M, C = "jev_prob", "jev_matches(%s, 'is angry', NULL)" % ROW, "jev_choice(%s, 'team', '[\"a\",   \"b\"]')" % ROW
with eng.connect() as c:
    got = c.exec_driver_sql("SELECT id, %s(%s, 'is angry'), %s, %s FROM tickets t ORDER BY id" % (P, ROW, M, C)).fetchall()
    assert [(r[0], float(r[1]), bool(r[2])) for r in got] == [(1, 0.9, True), (2, 0.1, False), (3, 0.9, True)], got
    assert all(r[3] in ("a", "b") for r in got), got
    c.exec_driver_sql("UPDATE tickets SET body = 'now calm' WHERE id = 3")
    assert c.exec_driver_sql("SELECT %s(%s, 'is angry') FROM tickets t WHERE id = 3" % (P, ROW)).scalar() is None
    if pg:
        n = c.exec_driver_sql("SELECT count(*) FROM tickets t JOIN jev.answers_for('is angry') a "
                              "ON a.row_hash = jev.row_hash(to_jsonb(t)) WHERE a.prob >= 0.5").scalar()
        assert n == 1, n
        assert c.exec_driver_sql("SELECT jev.forget('team', 'choice', ARRAY['a','b'])").scalar() == 3
print("%s smoke: ok" % eng.dialect.name)
