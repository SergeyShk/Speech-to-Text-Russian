import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import stt_ru

ROOT = Path(__file__).parents[1]


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", stt_ru.__version__)
    assert stt_ru.__version__ != "0.0.0"


def test_metadata():
    assert "русской речи" in stt_ru.__description__
    assert "__version__" in stt_ru.__all__
    assert stt_ru.__all__ == sorted(stt_ru.__all__)


def test_version_fallback(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        assert importlib.reload(stt_ru).__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(stt_ru)
    assert stt_ru.__version__ != "0.0.0"


def test_entry_point():
    (point,) = importlib.metadata.entry_points(group="console_scripts", name="stt-ru")
    assert point.value == "stt_ru.cli:main"


def test_package_data():
    assert (Path(stt_ru.__file__).parent / "py.typed").is_file()


@pytest.mark.skipif(shutil.which("uv") is None, reason="колесо собирает uv")
def test_wheel_contents(tmp_path):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        capture_output=True,
        check=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "stt_ru/py.typed" in names
    assert all(name.startswith(("stt_ru/", "stt_ru-")) for name in names)


class TestTelemetry:
    def run(self, tmp_path, **env):
        code = "import os, stt_ru; print(os.environ['ORT_DISABLE_TELEMETRY'])"
        environ = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path), **env}
        environ = {key: value for key, value in environ.items() if value is not None}
        return subprocess.run(
            [sys.executable, "-c", code], env=environ, capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_off_by_default(self, tmp_path):
        assert self.run(tmp_path, ORT_DISABLE_TELEMETRY=None) == "1"
        assert not list(tmp_path.rglob("deviceid"))

    def test_user_value_kept(self, tmp_path):
        assert self.run(tmp_path, ORT_DISABLE_TELEMETRY="0") == "0"
