#!/usr/bin/env bash
# Cut a release in one command. The launch is:   scripts/release.sh 0.1.0
# Sets the version only if it differs (0.1.0 already matches), dates the CHANGELOG entry, runs the tests,
# commits, tags vX.Y.Z and pushes.
# GitHub Actions (.github/workflows/release.yml) then builds the package, creates the GitHub Release with the
# wheel and every database's install script attached, and publishes to PyPI when PYPI_PUBLISH is enabled.
set -euo pipefail
V="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "version must look like 1.2.3"; exit 1; }
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo "commit or stash your changes first"; exit 1; }
[[ "$(git branch --show-current)" == "main" ]] || { echo "release from main"; exit 1; }

CUR=$(python -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('src/sqljev/core.py').read()).group(1))")
if [[ "$CUR" != "$V" ]]; then
  sed -i.bak -E "s/^__version__ = \".*\"/__version__ = \"$V\"/" src/sqljev/core.py && rm src/sqljev/core.py.bak
  python scripts/build_sql.py
fi
if grep -q "^## $V" CHANGELOG.md; then :; elif grep -q "(unreleased)" CHANGELOG.md; then
  sed -i.bak -E "0,/\(unreleased\)/s//($(date +%Y-%m-%d))/" CHANGELOG.md && rm CHANGELOG.md.bak
  sed -i.bak -E "0,/^## [0-9]+\.[0-9]+\.[0-9]+ /s//## $V /" CHANGELOG.md && rm CHANGELOG.md.bak
else
  printf '# Changelog\n\n## %s (%s)\n\n- \n' "$V" "$(date +%Y-%m-%d)" | cat - <(tail -n +2 CHANGELOG.md) > CHANGELOG.tmp
  mv CHANGELOG.tmp CHANGELOG.md; echo "Add release notes to CHANGELOG.md, then run again."; git checkout -- src sql; exit 1
fi
python -m pytest -q
git add -A
git diff --cached --quiet || git commit -m "Release $V"
git tag -a "v$V" -m "sqljev $V"
git push origin main "v$V"
echo "Tagged v$V. Watch the release build: gh run watch"
