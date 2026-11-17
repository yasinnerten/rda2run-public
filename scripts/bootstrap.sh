#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
revision=$(cat docs/upstream-commit.txt)
if [ ! -d more-than-a-hack-2027/.git ]; then
  git clone https://github.com/metrica-sports/more-than-a-hack-2027.git more-than-a-hack-2027
  git -C more-than-a-hack-2027 checkout --detach "$revision"
fi
actual=$(git -C more-than-a-hack-2027 rev-parse HEAD)
if [ "$actual" != "$revision" ]; then
  echo "Upstream revision differs. Preserve local work and inspect docs/upstream-commit.txt before proceeding." >&2
  exit 1
fi
