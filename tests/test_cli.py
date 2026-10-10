import importlib
import io
import json
import runpy
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import stt_ru
from stt_ru import Segment, Transcript, cli
from stt_ru.batch import WatchReport
from stt_ru.cli import main
from stt_ru.exceptions import AudioError, ModelError
from stt_ru.vad import VadOptions

CALL = Path(__file__).parent / "data" / "call.wav"


class FakeTranscriber:
    instances: list["FakeTranscriber"] = []  # noqa: RUF012

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        FakeTranscriber.instances.append(self)

    def transcribe(self, path):
        path = Path(path)
        if "broken" in path.name:
            raise AudioError(f"ffmpeg не прочитал файл {path}")
        if "interrupt" in path.name:
            raise KeyboardInterrupt
        segments = (Segment(0.5, 1.0, f"Текст {path.stem}.", 0), Segment(1.0, 2.0, "Ответ.", 1))
        return Transcript(segments, 2.0, 2, self.model, path.name)


@pytest.fixture
def fake(monkeypatch):
    FakeTranscriber.instances.clear()
    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    return FakeTranscriber.instances


@pytest.fixture(autouse=True)
def exits(monkeypatch):
    """os._exit завершил бы и pytest: заменяется на SystemExit"""

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(cli.os, "_exit", fake_exit)


@pytest.fixture
def files(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.wav", "b.wav"):
        (folder / name).write_bytes(b"audio")
    return folder


def run(argv):
    with pytest.raises(SystemExit) as info:
        main(argv)
    return info.value.code


def test_version(capsys):
    assert run(["--version"]) == 0
    assert (
        capsys.readouterr().out == f"stt-ru {stt_ru.__version__}, onnx-asr {version('onnx-asr')}\n"
    )


def test_no_command(capsys):
    assert run([]) == 2
    assert "transcribe" in capsys.readouterr().err


def test_help(capsys):
    assert run(["transcribe", "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("использование: stt-ru transcribe")
    assert "--output-dir" in out and "--watch" in out


class TestPrint:
    def test_one_file(self, fake, files, capsys):
        main(["transcribe", str(files / "a.wav")])
        captured = capsys.readouterr()
        assert captured.out == "Канал 0: Текст a.\nКанал 1: Ответ.\n"
        assert "Загрузка модели gigaam-v3-e2e-rnnt..." in captured.err
        assert "Готово" not in captured.err

    def test_quiet(self, fake, files, capsys):
        main(["transcribe", str(files / "a.wav"), "-q"])
        assert capsys.readouterr().err == ""

    def test_quiet_shows_errors(self, fake, files, capsys):
        (files / "broken.wav").write_bytes(b"x")
        assert run(["transcribe", str(files / "broken.wav"), "-q"]) == 1
        assert "broken.wav: ошибка - ffmpeg не прочитал файл" in capsys.readouterr().err

    def test_wrong_format_before_model(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "-f", "docx", "--delete-source"]) == 2
        assert "Форматы не поддерживаются: docx" in capsys.readouterr().err
        assert fake == []
        assert (files / "a.wav").exists()

    def test_delete_source_needs_results(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "--delete-source"]) == 2
        assert "для --delete-source нужен --output-dir или --summary" in capsys.readouterr().err
        assert fake == []

    def test_delete_source_with_summary(self, fake, files, tmp_path, capsys):
        main(
            [
                "transcribe",
                str(files / "a.wav"),
                "--delete-source",
                "--summary",
                str(tmp_path / "all.csv"),
            ]
        )
        assert capsys.readouterr().out.startswith("Канал 0: Текст a.")
        assert not (files / "a.wav").exists()

    def test_no_files(self, fake, tmp_path, capsys):
        main(["transcribe", str(tmp_path)])
        assert "Файлов для распознавания нет" in capsys.readouterr().err
        assert fake == []

    def test_several_txt(self, fake, files, capsys):
        main(["transcribe", str(files)])
        captured = capsys.readouterr()
        assert captured.out.count("==> ") == 2
        assert f"==> {files / 'a.wav'} <==\nКанал 0: Текст a." in captured.out
        assert "Готово: распознано 2 из 2" in captured.err

    def test_several_csv(self, fake, files, capsys):
        main(["transcribe", str(files), "-f", "csv"])
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "Audio File,Start,End,Name,Text"
        assert len(lines) == 5
        assert lines.count("Audio File,Start,End,Name,Text") == 1

    def test_several_jsonl(self, fake, files, capsys):
        main(["transcribe", str(files), "-f", "JSONL"])
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert sorted({row["source"] for row in rows}) == ["a.wav", "b.wav"]

    def test_json_needs_output_dir_for_several(self, fake, files, capsys):
        assert run(["transcribe", str(files), "-f", "json"]) == 2
        assert "формат json печатается только для одного файла" in capsys.readouterr().err

    def test_json_one_file(self, fake, files, capsys):
        main(["transcribe", str(files / "a.wav"), "-f", "json"])
        assert json.loads(capsys.readouterr().out)["source"] == "a.wav"

    def test_several_formats_need_output_dir(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "-f", "txt,srt"]) == 2
        assert "stt-ru transcribe: ошибка: без --output-dir печатается один формат" in (
            capsys.readouterr().err
        )


class TestOutputDir:
    def test_default_formats(self, fake, files, tmp_path, capsys):
        main(["transcribe", str(files), "-o", str(tmp_path / "out")])
        names = sorted(path.name for path in (tmp_path / "out").iterdir())
        assert names == ["a.json", "a.srt", "a.txt", "b.json", "b.srt", "b.txt"]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Готово: распознано 2 из 2, пропущено 0, с ошибкой 0" in captured.err

    def test_formats_and_summary(self, fake, files, tmp_path):
        out = tmp_path / "out"
        main(
            [
                "transcribe",
                str(files),
                "-o",
                str(out),
                "-f",
                "ass, CSV",
                "--summary",
                str(out / "all.csv"),
            ]
        )
        assert sorted(path.name for path in out.iterdir()) == [
            "a.ass",
            "a.csv",
            "all.csv",
            "b.ass",
            "b.csv",
        ]
        assert (out / "all.csv").read_bytes().count(b"Audio File") == 1

    def test_unknown_format(self, fake, files, tmp_path, capsys):
        assert run(["transcribe", str(files), "-o", str(tmp_path), "-f", "srt,docx"]) == 2
        assert "Форматы не поддерживаются: docx" in capsys.readouterr().err

    def test_failure_exit_code(self, fake, files, tmp_path, capsys):
        (files / "broken.wav").write_bytes(b"x")
        assert run(["transcribe", str(files), "-o", str(tmp_path / "out")]) == 1
        err = capsys.readouterr().err
        assert "broken.wav: ошибка - ffmpeg не прочитал файл" in err
        assert "Готово: распознано 2 из 3, пропущено 0, с ошибкой 1" in err

    def test_print_failure_exit_code(self, fake, files, capsys):
        (files / "broken.wav").write_bytes(b"x")
        assert run(["transcribe", str(files / "broken.wav")]) == 1

    def test_options_passed(self, fake, files, tmp_path):
        out = tmp_path / "out"
        main(
            [
                "transcribe",
                str(files),
                "-o",
                str(out),
                "-r",
                "--overwrite",
                "--delete-source",
                "-j",
                "2",
            ]
        )
        assert list(files.iterdir()) == []

    def test_wrong_workers(self, fake, files, tmp_path, capsys):
        assert run(["transcribe", str(files), "-o", str(tmp_path), "-j", "0"]) == 2
        assert "Число потоков должно быть положительным: 0" in capsys.readouterr().err
        assert fake == []

    def test_wrong_summary(self, fake, files, tmp_path, capsys):
        assert run(["transcribe", str(files), "-o", str(tmp_path), "--summary", "all.txt"]) == 2
        assert "Общий файл должен быть CSV или JSONL" in capsys.readouterr().err
        assert fake == []

    def test_missing_path(self, fake, tmp_path, capsys):
        assert run(["transcribe", str(tmp_path / "missing.wav")]) == 2
        assert "Файл или каталог не найден" in capsys.readouterr().err


class TestModelOptions:
    def test_defaults(self, fake, files):
        main(["transcribe", str(files / "a.wav"), "-q"])
        (transcriber,) = fake
        assert transcriber.model == "gigaam-v3-e2e-rnnt"
        assert transcriber.kwargs == {
            "quantization": "int8",
            "model_dir": None,
            "providers": None,
            "vad": VadOptions(),
            "split_channels": True,
            "ffmpeg": None,
        }

    def test_arguments(self, fake, files, tmp_path):
        main(
            [
                "transcribe", str(files / "a.wav"), "-q", "-m", "t-tech/t-one",
                "--quantization", "None", "--model-dir", str(tmp_path), "--device", "cpu",
                "--mono", "--ffmpeg", "/opt/ffmpeg", "--vad-threshold", "0.4",
                "--min-speech-ms", "100", "--max-speech-s", "15", "--min-silence-ms", "300",
                "--speech-pad-ms", "50",
            ]
        )  # fmt: skip
        (transcriber,) = fake
        assert transcriber.model == "t-tech/t-one"
        assert transcriber.kwargs == {
            "quantization": None,
            "model_dir": str(tmp_path),
            "providers": ["CPUExecutionProvider"],
            "vad": VadOptions(0.4, None, 100, 15, 300, 50),
            "split_channels": False,
            "ffmpeg": "/opt/ffmpeg",
        }

    def test_available_device(self, fake, files, monkeypatch):
        monkeypatch.setattr(
            cli.rt,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        main(["transcribe", str(files / "a.wav"), "-q", "--device", "cuda"])
        assert fake[0].kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @pytest.mark.parametrize(
        ("device", "hint"),
        [("cuda", "onnxruntime-gpu"), ("coreml", "macOS"), ("directml", "onnxruntime-directml")],
    )
    def test_unavailable_device(self, fake, files, monkeypatch, capsys, device, hint):
        monkeypatch.setattr(cli.rt, "get_available_providers", lambda: ["CPUExecutionProvider"])
        assert run(["transcribe", str(files / "a.wav"), "--device", device]) == 2
        assert f"устройство {device} недоступно" in capsys.readouterr().err
        assert fake == []

    def test_wrong_vad(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "--vad-threshold", "2"]) == 2
        assert "threshold должен быть от 0 до 1" in capsys.readouterr().err

    def test_model_error(self, monkeypatch, files, capsys):
        def broken(*args, **kwargs):
            raise ModelError("Не удалось загрузить модель x: нет сети")

        monkeypatch.setattr(cli, "Transcriber", broken)
        assert run(["transcribe", str(files / "a.wav")]) == 1
        assert "stt-ru: ошибка: Не удалось загрузить модель x: нет сети" in capsys.readouterr().err


class TestWatch:
    def test_needs_output_dir(self, fake, files, capsys):
        assert run(["transcribe", str(files), "--watch"]) == 2
        assert "для --watch нужен --output-dir" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "message"),
        [("--interval=-1", "Интервал просмотра должен быть положительным"),
         ("--min-age=-5", "Время без изменений не может быть отрицательным")],
    )  # fmt: skip
    def test_wrong_timing(self, fake, files, tmp_path, capsys, option, message):
        assert run(["transcribe", str(files), "-o", str(tmp_path), "--watch", option]) == 2
        assert message in capsys.readouterr().err
        assert fake == []

    def test_needs_directories(self, fake, files, tmp_path, capsys):
        assert run(["transcribe", str(files / "a.wav"), "-o", str(tmp_path), "--watch"]) == 2
        assert "--watch следит только за каталогами" in capsys.readouterr().err

    def test_runs_watch(self, fake, files, tmp_path, monkeypatch):
        seen = {}

        def watch(batch, directories, **kwargs):
            seen.update(kwargs, directories=directories, formats=batch.formats)
            return WatchReport(failed={Path("x"): "ошибка"})

        monkeypatch.setattr(cli, "watch", watch)
        code = run(
            [
                "transcribe",
                str(files),
                "-o",
                str(tmp_path),
                "--watch",
                "--interval",
                "2",
                "--min-age",
                "30",
                "-r",
            ]
        )
        assert code == 1
        assert seen == {
            "recursive": True,
            "interval": 2.0,
            "min_age": 30.0,
            "directories": [str(files)],
            "formats": ["txt", "srt", "json"],
        }


class TestArgparseMessages:
    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["transcribe", "x.wav", "--bogus"], "неизвестные аргументы: --bogus"),
            (
                ["transcribe", "x.wav", "--device", "tpu"],
                "аргумент --device: неизвестное значение 'tpu'",
            ),
            (
                ["transcribe", "x.wav", "-j", "many"],
                "аргумент -j/--workers: неверное число 'many'",
            ),
            (["transcribe", "x.wav", "-o"], "аргумент -o/--output-dir: нужно значение"),
            (["transcribe"], "нужны аргументы: ПУТЬ"),
            (
                ["transcribe", "x.wav", "--o", "y"],
                "неоднозначный параметр --o: подходят --output-dir, --overwrite",
            ),
        ],
    )
    def test_russian(self, capsys, argv, message):
        assert run(argv) == 2
        err = capsys.readouterr().err
        assert message in err
        assert "использование: stt-ru" in err


def test_interrupt(fake, files):
    (files / "interrupt.wav").write_bytes(b"x")
    assert run(["transcribe", str(files / "interrupt.wav"), "-q"]) == 130


class Console(io.TextIOWrapper):
    def isatty(self):
        return True


def test_redirected_output_is_utf8(monkeypatch, fake, files):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    main(["transcribe", str(files / "a.wav"), "-q"])
    stream.flush()
    assert buffer.getvalue().decode("utf-8").startswith("Канал 0: Текст a.")


def test_ascii_console(monkeypatch, fake, files):
    buffer = io.BytesIO()
    stream = Console(buffer, encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    main(["transcribe", str(files / "a.wav"), "-q"])
    stream.flush()
    assert buffer.getvalue().startswith(b"????? 0: ????? a.")


def test_broken_pipe(monkeypatch, fake, files):
    class Closed(io.StringIO):
        def write(self, text):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", Closed())
    assert run(["transcribe", str(files), "-q"]) == 1


def test_module_import(fake):
    importlib.import_module("stt_ru.__main__")
    assert fake == []


def test_module(monkeypatch, fake, files, capsys):
    monkeypatch.setattr(sys, "argv", ["stt_ru", "transcribe", str(files / "a.wav"), "-q"])
    monkeypatch.delitem(sys.modules, "stt_ru.__main__", raising=False)
    runpy.run_module("stt_ru", run_name="__main__")
    assert capsys.readouterr().out.startswith("Канал 0: Текст a.")


@pytest.mark.network
def test_real_model(capsys):
    main(["transcribe", str(CALL), "-q"])
    out = capsys.readouterr().out
    assert "Канал 1: Наталья, благодарю вас за ожидание." in out
    assert "до свидания" in out.lower()
