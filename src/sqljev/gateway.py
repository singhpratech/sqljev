"""sql-jev-laya gateway: one HTTP service that answers the batch protocols databases use to call out.

    sql-jev-laya gateway --port 8765            # Laya in-process (backend 'local') by default

Routes (all POST, JSON):

  /v1/eval               sql-jev-laya native: {"question", "kind", "options", "rows": [...]} -> {"answers": [...]}
  /sqlserver             same body; called by jev.judge through sp_invoke_external_rest_endpoint
  /bigquery              BigQuery remote functions: {"calls": [[args]], "userDefinedContext": {"fn": ...}}
                         -> {"replies": [...]}
  /snowflake/<fn>        Snowflake external / service functions: {"data": [[rownum, args...]]}
                         -> {"data": [[rownum, result]]}
  /redshift              Redshift Lambda UDF payload over HTTP (the same handler as sqljev.aws_lambda)
  GET /health, GET /stats

Each database already sends rows in batches (BigQuery up to max_batching_rows, Snowflake/Redshift in their
own batch sizes), so every request becomes one engine.call(): rows grouped by question, de-duplicated,
looked up in the cache and the misses judged in shared forward passes.

Auth: set SQLJEV_GATEWAY_TOKEN and send it as "Authorization: Bearer <token>" or "X-Jev-Token: <token>".
BigQuery cannot send headers: run the gateway on Cloud Run with IAM auth instead.
"""
import hmac
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import FUNCTIONS, JevError, Jev, __version__, function_name

MAX_BODY_BYTES = 64 * 1024 * 1024


def handle_eval(engine, body):
    kind = body.get("kind") or "noul"
    rows = body.get("rows")
    if not isinstance(rows, list) or not body.get("question"):
        raise JevError("sql-jev-laya: body needs 'question' and a 'rows' array")
    return {"answers": engine.evaluate(rows, body["question"], kind, body.get("options"))}


def handle_bigquery(engine, body):
    ctx = body.get("userDefinedContext") or {}
    fn = ctx.get("fn") or ctx.get("function")
    if not fn:
        raise JevError("sql-jev-laya: set user_defined_context = [('fn', 'jev_prob')] on the remote function")
    return {"replies": _json_safe(engine.call(fn, body.get("calls") or []))}


def handle_snowflake(engine, body, fn):
    data = body.get("data") or []
    results = engine.call(fn, [r[1:] for r in data])    # VARIANT takes jev_eval's object as is
    return {"data": [[r[0], v] for r, v in zip(data, results)]}


def handle_redshift(engine, event):
    """Redshift Lambda UDF protocol. Errors are reported in-band, as Redshift expects."""
    try:
        fn = event.get("external_function") or ""
        args = event.get("arguments") or []
        results = engine.call(fn, args)
        return {"success": True, "num_records": len(results), "results": _json_safe(results)}
    except Exception as e:    # noqa: BLE001 -- Redshift shows error_msg to the user
        return {"success": False, "error_msg": str(e)}


def _json_safe(values):
    # jev_eval answers go back to warehouses as JSON text columns (BigQuery JSON/STRING, Snowflake VARIANT
    # accepts objects directly but a string is safe everywhere).
    return [json.dumps(v) if isinstance(v, dict) else v for v in values]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine = None
    token = None
    server_version = "sql-jev-laya/" + __version__

    def log_message(self, fmt, *args):
        if os.environ.get("SQLJEV_GATEWAY_LOG"):
            sys.stderr.write("sql-jev-laya gateway: " + fmt % args + "\n")

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        if not self.token:
            return True
        got = self.headers.get("X-Jev-Token") or ""
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            got = got or auth[7:]
        return hmac.compare_digest(got.encode(), self.token.encode())

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "version": __version__, "backend": self.engine.cfg["backend"],
                                    "functions": sorted(FUNCTIONS)})
        if self.path == "/stats":
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            return self._send(200, self.engine.stats())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY_BYTES:
            return self._send(413, {"error": "body too large"})
        raw = self.rfile.read(n)
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            body = json.loads(raw or b"{}")
            if path in ("/v1/eval", "/sqlserver"):
                return self._send(200, handle_eval(self.engine, body))
            if path == "/bigquery":
                return self._send(200, handle_bigquery(self.engine, body))
            if path.startswith("/snowflake/"):
                return self._send(200, handle_snowflake(self.engine, body, function_name(path.rsplit("/", 1)[1])))
            if path == "/redshift":
                return self._send(200, handle_redshift(self.engine, body))
            return self._send(404, {"error": "not found: " + path})
        except (JevError, ValueError, KeyError, TypeError, IndexError) as e:
            # 400 is not retried by BigQuery/Snowflake: validation errors surface as query errors.
            return self._send(400, {"error": str(e), "errorMessage": str(e)})
        except Exception as e:     # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": "internal error: %s" % type(e).__name__,
                                    "errorMessage": "sql-jev-laya gateway internal error"})


def make_server(host="127.0.0.1", port=8765, engine=None, token=None):
    handler = type("SqlJevHandler", (Handler,), {
        "engine": engine or Jev(),
        "token": token if token is not None else os.environ.get("SQLJEV_GATEWAY_TOKEN"),
    })
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((host, port), handler)


def serve(host="127.0.0.1", port=8765, engine=None, token=None, certfile=None, keyfile=None):
    """certfile/keyfile serve HTTPS directly (sp_invoke_external_rest_endpoint only calls https:// URLs);
    behind a TLS-terminating proxy or Cloud Run, leave them unset."""
    srv = make_server(host, port, engine, token)
    scheme = "http"
    if certfile:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        accept = srv.get_request

        def get_request():
            # Handshake lazily, in the worker thread: a slow or failing client never blocks accept().
            sock, addr = accept()
            return ctx.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr
        srv.get_request = get_request
        scheme = "https"
    eng = srv.RequestHandlerClass.engine
    print("sql-jev-laya gateway %s on %s://%s:%d (backend %s)" % (__version__, scheme, host, port, eng.cfg["backend"]),
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
