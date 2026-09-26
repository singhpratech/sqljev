#!/usr/bin/env python3
"""SQL Server smoke test: install sql/sqlserver/install.sql, judge a table with `sql-jev-laya judge-sqlserver`
against the mock model, and read answers back through jev.prob / jev.matches / jev.choice / jev.answers_for.

    SQLJEV_TEST_MSSQL="mssql+pymssql://sa:pw@127.0.0.1:1433/master" python tests/sqlserver_smoke.py
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

url = os.environ["SQLJEV_TEST_MSSQL"]
admin = sa.create_engine(url, isolation_level="AUTOCOMMIT")
with admin.connect() as c:
    c.execute(sa.text("IF DB_ID('sqljev_smoke') IS NOT NULL DROP DATABASE sqljev_smoke"))
    c.execute(sa.text("CREATE DATABASE sqljev_smoke"))
db_url = url.rsplit("/", 1)[0] + "/sqljev_smoke"
eng = sa.create_engine(db_url, isolation_level="AUTOCOMMIT")
script = (HERE.parent / "sql/sqlserver/install.sql").read_text()
with eng.connect() as c:
    for batch in re.split(r"^\s*GO\s*$", script, flags=re.M):
        if batch.strip():
            c.exec_driver_sql(batch)
    c.exec_driver_sql("CREATE TABLE dbo.tickets (id int PRIMARY KEY, body nvarchar(200), note nvarchar(50) NULL)")
    c.exec_driver_sql("INSERT dbo.tickets VALUES (1, N'I am angry', NULL), (2, N'all calm', N'vip'), (3, N'angry again', NULL)")

srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_api.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
api = "http://127.0.0.1:%d/v1/systemone" % srv.server_address[1]
os.environ["TYPESAFE_API_KEY"] = "test-key"
common = ["--backend", "jev", "--api-url", api]
main(["judge-sqlserver", db_url, "--source", "dbo.tickets", "--prob", "is angry", *common])
main(["judge-sqlserver", db_url, "--source", "tickets", "--prob", "is angry", *common])            # nothing new
main(["judge-sqlserver", db_url, "--source", "dbo.tickets", "--choice", "team", "--options", '["a", "b"]', *common])

ROW = "(SELECT t.* FOR JSON PATH, WITHOUT_ARRAY_WRAPPER)"
with eng.connect() as c:
    got = c.exec_driver_sql(
        "SELECT id, jev.prob(%s, N'is angry'), jev.matches(%s, N'is angry', DEFAULT), "
        "jev.choice(%s, N'team', N'[\"a\",\"b\"]') FROM dbo.tickets t ORDER BY id" % (ROW, ROW, ROW)).fetchall()
    assert [(r[0], r[1], r[2]) for r in got] == [(1, 0.9, True), (2, 0.1, False), (3, 0.9, True)], got
    assert all(r[3] in ("a", "b") for r in got), got
    n = c.exec_driver_sql("SELECT COUNT(*) FROM dbo.tickets t JOIN jev.answers_for(N'is angry', 'noul', NULL) a "
                          "ON a.row_hash = jev.row_hash(%s) WHERE a.prob >= 0.5" % ROW).scalar()
    assert n == 2, n
    assert c.exec_driver_sql("SELECT COUNT(*) FROM jev.answers").scalar() == 6
    c.exec_driver_sql("UPDATE dbo.tickets SET body = N'now calm' WHERE id = 3")
    assert c.exec_driver_sql("SELECT jev.prob(%s, N'is angry') FROM dbo.tickets t WHERE id = 3" % ROW).scalar() is None
    removed = c.exec_driver_sql("EXEC jev.forget N'team', 'choice', N'[\"a\",\"b\"]'").scalar()
    assert removed == 3, removed
print("sqlserver smoke: ok")
