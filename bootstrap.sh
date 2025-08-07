#!/usr/bin/env bash
set -e

code-server --install-extension ritwickdey.liveserver
code-server --install-extension ms-python.python
code-server --install-extension ms-toolsai.jupyter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "✅ Pakete installiert"