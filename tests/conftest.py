import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, __import__("os").path.dirname(__file__))
import mock_api  # noqa: E402


@pytest.fixture(scope="session")
def mock_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_api.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/v1/systemone" % srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("SQLJEV_") or k in ("TYPESAFE_API_KEY", "LAYA_API_KEY"):
            monkeypatch.delenv(k)


@pytest.fixture
def fake_laya(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", mock_api.fake_laya_module())
    mock_api.FakeRouter.batches.clear()
    return mock_api.FakeRouter


@pytest.fixture
def jev_mock(mock_url):
    from sqljev import Jev
    return Jev(backend="jev", api_url=mock_url, api_key="test-key", concurrency=4)


@pytest.fixture
def requests_made():
    def count():
        return mock_api.Handler.requests
    return count
