"""Проверка здоровья контейнера: stt-ru serve отвечает на /healthz, другие команды здоровы"""

import sys
import urllib.request
from pathlib import Path

# Командная строка процесса 1: python /opt/venv/bin/stt-ru <команда> [параметры]
path = Path(sys.argv[1] if len(sys.argv) > 1 else "/proc/1/cmdline")
args = path.read_bytes().decode(errors="replace").split("\0")
command = next((args[i + 1] for i, arg in enumerate(args[:-1]) if arg.endswith("stt-ru")), None)
if command != "serve":
    sys.exit(0)
port = "8000"
for i, arg in enumerate(args):
    if arg == "--port" and i + 1 < len(args):
        port = args[i + 1]
    elif arg.startswith("--port="):
        port = arg.removeprefix("--port=")
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=4)
except OSError as error:
    sys.exit(f"stt-ru serve не отвечает на порте {port}: {error}")
