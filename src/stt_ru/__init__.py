# Распознавание русской речи (stt-ru)
#
# Copyright (C) 2019-2026
# Автор: Шкарин Сергей <kouki.sergey@gmail.com>
# URL: <https://github.com/SergeyShk/Speech-to-Text-Russian>

import os
from importlib.metadata import PackageNotFoundError, version

# До импорта onnxruntime: иначе он пишет идентификатор устройства в домашний каталог и отправляет
# сведения об устройстве в Microsoft, а на выходе может зависнуть, дожидаясь отправки
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from .pipeline import Transcriber
from .result import Segment, Transcript

try:
    __version__ = version("stt-ru")
except PackageNotFoundError:
    __version__ = "0.0.0"
__description__ = """Распознавание русской речи: библиотека Python, CLI stt-ru, веб-сервис и Docker-образ.
Требует версию Python 3.11 и выше"""
__author__ = "Шкарин Сергей"
__author_email__ = "kouki.sergey@gmail.com"

__all__ = ["Segment", "Transcriber", "Transcript", "__version__"]
