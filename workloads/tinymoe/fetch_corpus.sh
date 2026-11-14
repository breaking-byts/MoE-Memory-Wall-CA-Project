#!/usr/bin/env bash
# Fetch the two text domains used to train / trace TinyMoE.
#   prose: Tiny Shakespeare (karpathy/char-rnn, 1.1 MB)
#   code : a fixed list of CPython v3.12.0 standard-library modules (pinned tag)
# Output: corpus/prose.txt, corpus/code.txt   (not committed; ~3.6 MB total)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p corpus
curl -fsSL -o corpus/prose.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
MODS="argparse ast asyncio/base_events asyncio/tasks calendar collections/__init__ configparser \
contextlib csv dataclasses datetime decimal difflib email/message enum fractions ftplib functools \
gettext gzip heapq http/client http/server imaplib inspect ipaddress json/decoder json/encoder \
logging/__init__ mailbox nntplib optparse pathlib pdb pickle platform plistlib pprint pydoc \
random shutil smtplib socket statistics string subprocess tarfile tempfile textwrap threading \
tokenize traceback typing unittest/case urllib/parse urllib/request uuid zipfile/__init__"
: > corpus/code.txt
for m in $MODS; do
  curl -fsSL "https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/$m.py" >> corpus/code.txt
  printf '\n' >> corpus/code.txt
done
wc -c corpus/prose.txt corpus/code.txt
