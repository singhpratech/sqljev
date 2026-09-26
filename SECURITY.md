# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through
[GitHub's private vulnerability reporting](https://github.com/singhpratech/sqljev/security/advisories/new).
You will get an acknowledgement within a few days, and a fix or mitigation plan as soon as the issue is confirmed.

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/sqljev/).

## Things to know when deploying sqljev

- **Row data and the model.** With the default `local` backend, Laya runs on your own machine and rows never
  leave it. With `backend=jev`, row contents are sent to TypeSafe's API: do not use it on data you may not share.
- **The gateway** (`sqljev gateway`) answers anyone who can reach it. Set `SQLJEV_GATEWAY_TOKEN`, serve it over
  HTTPS (`--certfile/--keyfile` or a TLS proxy), and do not expose it to the internet without both.
- **Credentials.** Pass database URLs and API keys through environment variables or your platform's secret store
  (Snowflake secrets, Databricks secret scopes, Colab secrets), never in SQL files or notebooks you commit.
- **`sqljev judge --where` and `jev.judge @where`** take raw SQL on purpose; only let trusted users call them.
