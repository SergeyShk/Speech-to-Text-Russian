import importlib
import io
import json
import os
import runpy
import socket
import sys
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn

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
        self.recognizer = SimpleNamespace(model=model)
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
        assert sorted({row["source"] for row in rows}) == [
            str(files / "a.wav"),
            str(files / "b.wav"),
        ]

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
         ("--min-age=-5", "Время без изменений должно быть неотрицательным числом")],
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


def test_exit_with_closed_stderr(monkeypatch, fake, files):
    class Closed(io.StringIO):
        def write(self, text):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stderr", Closed())
    assert run(["transcribe", str(files)]) == 1


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


@pytest.fixture
def manifest(files):
    path = files / "eval.csv"
    path.write_text(
        "audio,text,channel\na.wav,Текст a,0\nb.wav,Текст b. Вопрос.,\n", encoding="utf-8"
    )
    return path


class TestEvaluate:
    def test_table(self, fake, manifest, capsys):
        main(["evaluate", str(manifest)])
        captured = capsys.readouterr()
        header, row = captured.out.splitlines()
        assert header.split() == [
            "Модель",
            "WER,",
            "%",
            "CER,",
            "%",
            "Скорость",
            "Записей",
            "Ошибок",
        ]
        model, wer, cer, speed, count, failed = row.split()
        assert (model, wer, cer, count, failed) == ("gigaam-v3-e2e-rnnt", "20.0", "23.8", "2", "0")
        assert speed.endswith("x")
        assert "Записей в разметке: 2" in captured.err
        assert "Загрузка модели gigaam-v3-e2e-rnnt..." in captured.err
        assert f"[gigaam-v3-e2e-rnnt 2/2] {manifest.parent / 'b.wav'}: WER 33.3 %" in captured.err

    def test_several_models(self, fake, manifest, capsys):
        main(["evaluate", str(manifest), "-q", "-m", "org/a", "--model", "org/b", "-m", "org/a"])
        assert [transcriber.model for transcriber in fake] == ["org/a", "org/b"]
        lines = capsys.readouterr().out.splitlines()
        assert [line.split()[0] for line in lines] == ["Модель", "org/a", "org/b"]

    def test_model_options(self, fake, manifest, tmp_path):
        main(
            [
                "evaluate", str(manifest), "-q", "-m", "t-tech/t-one", "--quantization", "none",
                "--model-dir", str(tmp_path), "--device", "cpu", "--mono", "--ffmpeg", "ffmpeg",
                "--max-speech-s", "15",
            ]
        )  # fmt: skip
        assert fake[0].kwargs == {
            "quantization": None,
            "model_dir": str(tmp_path),
            "providers": ["CPUExecutionProvider"],
            "vad": VadOptions(max_speech_duration_s=15),
            "split_channels": False,
            "ffmpeg": "ffmpeg",
        }

    def test_report(self, fake, manifest, tmp_path, capsys):
        output = tmp_path / "reports" / "eval.json"
        main(["evaluate", str(manifest), "-o", str(output)])
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["manifest"] == str(manifest)
        (model,) = document["models"]
        assert (model["model"], model["wer"], model["files"]) == ("gigaam-v3-e2e-rnnt", 0.2, 2)
        assert [result["hypothesis"] for result in model["results"]] == [
            "Текст a.",
            "Текст b. Ответ.",
        ]
        assert f"Отчет: {output}" in capsys.readouterr().err

    def test_report_not_saved(self, fake, manifest, tmp_path, monkeypatch, capsys):
        def fail(path, content):
            raise OSError("диск заполнен")

        monkeypatch.setattr(cli, "write_atomic", fail)
        output = tmp_path / "eval.json"
        assert run(["evaluate", str(manifest), "-o", str(output), "-q"]) == 1
        captured = capsys.readouterr()
        assert f"stt-ru: ошибка: отчет не сохранен в {output}: диск заполнен" in captured.err
        assert "gigaam-v3-e2e-rnnt" in captured.out

    def test_report_path_checked_first(self, fake, manifest, tmp_path, capsys):
        output = tmp_path / "eval.json"
        output.mkdir()
        assert run(["evaluate", str(manifest), "-o", str(output), "-q"]) == 2
        assert f"{output} - каталог, а нужен файл" in capsys.readouterr().err
        assert fake == []

    def test_interrupt_keeps_finished_models(self, monkeypatch, manifest, tmp_path, capsys):
        def transcriber(model, **kwargs):
            if model == "org/slow":
                raise KeyboardInterrupt
            return FakeTranscriber(model, **kwargs)

        monkeypatch.setattr(cli, "Transcriber", transcriber)
        output = tmp_path / "eval.json"
        args = [
            "evaluate",
            str(manifest),
            "-q",
            "-m",
            "org/a",
            "-m",
            "org/slow",
            "-o",
            str(output),
        ]
        assert run(args) == 130
        captured = capsys.readouterr()
        assert [line.split()[0] for line in captured.out.splitlines()] == ["Модель", "org/a"]
        assert "оценка прервана, в таблице модели, оцененные целиком: 1" in captured.err
        document = json.loads(output.read_text(encoding="utf-8"))
        assert [model["model"] for model in document["models"]] == ["org/a"]

    def test_interrupt_before_any_model(self, monkeypatch, manifest, capsys):
        def transcriber(model, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "Transcriber", transcriber)
        assert run(["evaluate", str(manifest), "-q"]) == 130
        assert capsys.readouterr().out == ""

    def test_known_models(self, monkeypatch):
        assert cli._known_model("gigaam-v3-e2e-rnnt")
        assert cli._known_model("nemo-conformer-ctc")
        assert cli._known_model("org/any")
        assert not cli._known_model("gigaam-v3-e2e-rnn")
        monkeypatch.setitem(sys.modules, "onnx_asr.loader", None)
        assert cli._known_model("gigaam-v3-e2e-rnn")

    def test_unknown_model_checked_first(self, fake, manifest, capsys):
        args = [
            "evaluate",
            str(manifest),
            "-q",
            "-m",
            "gigaam-v3-e2e-rnnt",
            "-m",
            "gigaam-v3-e2e-rnn",
        ]
        assert run(args) == 2
        assert "неизвестные модели onnx-asr: gigaam-v3-e2e-rnn" in capsys.readouterr().err
        assert fake == []

    def test_directory(self, fake, files, capsys):
        (files / "a.txt").write_text("текст a ответ", encoding="utf-8")
        (files / "b.txt").write_text("текст b ответ", encoding="utf-8")
        main(["evaluate", str(files), "-q"])
        assert capsys.readouterr().out.splitlines()[1].split()[1:3] == ["0.0", "0.0"]

    def test_failed_file(self, fake, manifest, capsys):
        with manifest.open("a", encoding="utf-8") as file:
            file.write("broken.wav,да,\n")
        (manifest.parent / "broken.wav").write_bytes(b"x")
        assert run(["evaluate", str(manifest), "-q"]) == 1
        captured = capsys.readouterr()
        assert "broken.wav: ошибка - ffmpeg не прочитал файл" in captured.err
        assert captured.out.splitlines()[1].split()[-2:] == ["3", "1"]

    def test_model_error(self, monkeypatch, manifest, capsys):
        def transcriber(model, **kwargs):
            if model == "org/x":
                raise ModelError("Не удалось загрузить модель org/x: нет сети")
            return FakeTranscriber(model, **kwargs)

        monkeypatch.setattr(cli, "Transcriber", transcriber)
        assert run(["evaluate", str(manifest), "-q", "-m", "org/x", "-m", "org/y"]) == 1
        captured = capsys.readouterr()
        assert "stt-ru: ошибка: Не удалось загрузить модель org/x: нет сети" in captured.err
        lines = captured.out.splitlines()
        assert lines[1].startswith("org/x (не загружена)")
        assert lines[2].startswith("org/y ")

    def test_digits(self, fake, files, capsys):
        (files / "eval.jsonl").write_text(
            '{"audio": "a.wav", "text": "текст а ответ"}\n', encoding="utf-8"
        )

        def transcribe(self, path):
            return Transcript((Segment(0.0, 1.0, "В 15 часов"),), 1.0, 1, self.model, path.name)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(FakeTranscriber, "transcribe", transcribe)
            main(["evaluate", str(files / "eval.jsonl"), "-q"])
        assert (
            "gigaam-v3-e2e-rnnt: в 1 записях цифры есть только в эталоне или только в распознанном "
            "тексте - если числа записаны по-разному (цифрами и словами), WER завышен"
            in capsys.readouterr().err
        )

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["missing.csv"], "Разметка не найдена"),
            (["{manifest}", "-o", "eval.csv"], "отчет сохраняется в JSON: eval.csv"),
            (["{manifest}", "-m", "a", "-m", "b", "--model-dir", "m"], "только для одной модели"),
            (["{manifest}", "--min-speech-ms", "-1"], "min_speech_duration_ms"),
        ],
    )
    def test_wrong_arguments(self, fake, manifest, capsys, arguments, message):
        argv = ["evaluate", *(item.format(manifest=manifest) for item in arguments)]
        assert run(argv) == 2
        assert message in capsys.readouterr().err
        assert fake == []

    def test_unavailable_device(self, fake, manifest, monkeypatch, capsys):
        monkeypatch.setattr(cli.rt, "get_available_providers", lambda: ["CPUExecutionProvider"])
        assert run(["evaluate", str(manifest), "--device", "cuda"]) == 2
        assert "устройство cuda недоступно" in capsys.readouterr().err
        assert fake == []

    def test_missing_jiwer(self, fake, manifest, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "jiwer", None)
        assert run(["evaluate", str(manifest)]) == 1
        assert "pip install 'stt-ru[eval]'" in capsys.readouterr().err
        assert fake == []

    def test_help(self, capsys):
        assert run(["evaluate", "--help"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("использование: stt-ru evaluate")
        assert "РАЗМЕТКА" in out and "--output" in out and "--vad-threshold" in out


bind_error = cli._bind_error


class TestServe:
    @pytest.fixture
    def server(self, monkeypatch):
        runs = []

        def run(app, **kwargs):
            jobs = app.state.settings.jobs
            runs.append((app.state.settings, kwargs, jobs.directory.is_dir(), tempfile.tempdir))

        monkeypatch.setattr(uvicorn, "run", run)
        monkeypatch.setattr(cli, "_bind_error", lambda host, port: None)
        monkeypatch.delenv("STT_RU_TOKEN", raising=False)
        return runs

    def test_defaults(self, fake, server, capsys):
        before = tempfile.tempdir
        main(["serve"])
        ((settings, kwargs, existed, temporary),) = server
        assert temporary == before
        assert kwargs == {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
            "access_log": False,
            "timeout_graceful_shutdown": 3,
        }
        assert (settings.token, settings.max_size) == (None, 500 << 20)
        jobs = settings.jobs
        assert (jobs.max_queue, jobs.max_duration, jobs.keep) == (100, 10800, 86400)
        assert len(jobs._threads) == 1
        assert existed and not jobs.directory.exists()
        assert fake[0].kwargs == {
            "quantization": "int8",
            "model_dir": None,
            "providers": None,
            "vad": VadOptions(),
            "split_channels": True,
            "ffmpeg": None,
        }
        err = capsys.readouterr().err
        assert "Загрузка модели gigaam-v3-e2e-rnnt..." in err
        assert (
            "Веб-интерфейс: http://127.0.0.1:8000, описание API: http://127.0.0.1:8000/docs" in err
        )
        assert "без токена" not in err

    def test_options(self, fake, server, tmp_path, capsys):
        before = tempfile.tempdir
        main(
            [
                "serve", "--host", "0.0.0.0", "--port", "9000", "-j", "2", "--max-size", "1.5",
                "--max-duration", "30", "--max-queue", "5", "--keep", "0.5", "--data-dir",
                str(tmp_path), "-q", "-m", "t-tech/t-one", "--mono", "--device", "cpu",
            ]
        )  # fmt: skip
        ((settings, kwargs, _, temporary),) = server
        assert temporary == str(settings.jobs.directory)
        assert tempfile.tempdir == before
        assert (kwargs["host"], kwargs["port"], kwargs["log_level"]) == (
            "0.0.0.0",
            9000,
            "warning",
        )
        jobs = settings.jobs
        assert settings.max_size == int(1.5 * (1 << 20))
        assert (jobs.max_queue, jobs.max_duration, jobs.keep) == (5, 1800, 1800)
        assert len(jobs._threads) == 2
        assert jobs.directory.parent == tmp_path
        assert fake[0].model == "t-tech/t-one"
        assert fake[0].kwargs["split_channels"] is False
        assert fake[0].kwargs["providers"] == ["CPUExecutionProvider"]
        assert capsys.readouterr().err == (
            "stt-ru: предупреждение: сервис доступен по сети без токена; задайте токен "
            "в переменной окружения STT_RU_TOKEN\n"
        )

    def test_token(self, fake, server, monkeypatch, capsys):
        monkeypatch.setenv("STT_RU_TOKEN", "секрет")
        main(["serve", "--host", "0.0.0.0", "-q"])
        assert server[0][0].token == "секрет"
        assert capsys.readouterr().err == ""

    def test_token_without_line_break(self, fake, server, monkeypatch):
        monkeypatch.setenv("STT_RU_TOKEN", "секрет\r\n")
        main(["serve", "-q"])
        assert server[0][0].token == "секрет"

    def test_job_errors_logged_when_quiet(self, fake, server, capsys):
        main(["serve", "-q"])
        jobs = server[0][0].jobs
        jobs.errors("[1] a.wav: ошибка - файл не читается")
        assert capsys.readouterr().err == "[1] a.wav: ошибка - файл не читается\n"

    @pytest.mark.parametrize("value", ["nan", "inf"])
    @pytest.mark.parametrize("option", ["--max-size", "--max-duration", "--keep"])
    def test_not_finite(self, fake, server, capsys, option, value):
        assert run(["serve", option, value]) == 2
        assert f"{option} должен быть положительным числом: {value}" in capsys.readouterr().err
        assert fake == []

    def test_empty_data_dir(self, fake, server, capsys):
        assert run(["serve", "--data-dir", ""]) == 2
        assert "каталога нет" in capsys.readouterr().err

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="права каталогов POSIX"
    )
    def test_unwritable_data_dir(self, fake, server, tmp_path, capsys):
        tmp_path.chmod(0o500)
        try:
            assert run(["serve", "--data-dir", str(tmp_path)]) == 2
        finally:
            tmp_path.chmod(0o755)
        assert f"нет права записи в каталог {tmp_path}" in capsys.readouterr().err
        assert fake == []

    def test_missing_multipart(self, fake, server, monkeypatch, capsys):
        def broken(name, *args, **kwargs):
            if name == "service" or name.endswith("stt_ru.service"):
                raise RuntimeError('Form data requires "python-multipart" to be installed.')
            return importlib.__import__(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "stt_ru.service", raising=False)
        monkeypatch.setattr("builtins.__import__", broken)
        assert run(["serve"]) == 1
        assert (
            "stt-ru: ошибка: Для сервиса нужен пакет python-multipart: pip install 'stt-ru[service]'"
            in capsys.readouterr().err
        )
        assert fake == []

    def test_closed_stderr(self, fake, server, monkeypatch):
        class Closed(io.StringIO):
            def write(self, text):
                raise BrokenPipeError

        monkeypatch.setattr(sys, "stderr", Closed())
        main(["serve", "--host", "0.0.0.0"])
        assert len(server) == 1

    def test_ipv6(self, fake, server, capsys):
        main(["serve", "--host", "::1"])
        assert "Веб-интерфейс: http://[::1]:8000," in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (["--port", "0"], "порт должен быть от 1 до 65535: 0"),
            (["--port", "65536"], "порт должен быть от 1 до 65535: 65536"),
            (["-j", "0"], "--workers должен быть положительным числом: 0"),
            (["--max-size", "0"], "--max-size должен быть положительным числом: 0"),
            (["--max-duration", "-1"], "--max-duration должен быть положительным числом: -1"),
            (["--max-queue", "0"], "--max-queue должен быть положительным числом: 0"),
            (["--keep", "0"], "--keep должен быть положительным числом: 0"),
            (["--data-dir", "missing-dir"], "каталога нет: missing-dir"),
            (["--vad-threshold", "2"], "threshold должен быть от 0 до 1"),
        ],
    )
    def test_wrong_arguments(self, fake, server, capsys, arguments, message):
        assert run(["serve", *arguments]) == 2
        assert message in capsys.readouterr().err
        assert fake == [] and server == []

    def test_unavailable_device(self, fake, server, monkeypatch, capsys):
        monkeypatch.setattr(cli.rt, "get_available_providers", lambda: ["CPUExecutionProvider"])
        assert run(["serve", "--device", "cuda"]) == 2
        assert "устройство cuda недоступно" in capsys.readouterr().err

    def test_model_error(self, server, monkeypatch, capsys):
        def broken(*args, **kwargs):
            raise ModelError("Не удалось загрузить модель x: нет сети")

        monkeypatch.setattr(cli, "Transcriber", broken)
        assert run(["serve"]) == 1
        assert "stt-ru: ошибка: Не удалось загрузить модель x: нет сети" in capsys.readouterr().err
        assert server == []

    @pytest.mark.parametrize("module", ["uvicorn", "fastapi"])
    def test_missing_dependency(self, fake, server, monkeypatch, capsys, module):
        monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.delitem(sys.modules, "stt_ru.service", raising=False)
        monkeypatch.delitem(sys.modules, "stt_ru.service.api", raising=False)
        assert run(["serve"]) == 1
        assert (
            f"stt-ru: ошибка: Для сервиса нужен пакет {module}: pip install 'stt-ru[service]'"
            in capsys.readouterr().err
        )
        assert fake == []

    def test_busy_port(self, fake, server, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_bind_error", bind_error)
        with socket.create_server(("127.0.0.1", 0)) as busy:
            port = busy.getsockname()[1]
            assert run(["serve", "--port", str(port)]) == 1
        err = capsys.readouterr().err
        assert f"stt-ru: ошибка: не удалось занять адрес 127.0.0.1:{port}: " in err
        assert fake == [] and server == []

    def test_unknown_host(self, fake, server, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_bind_error", bind_error)
        assert run(["serve", "--host", "no-such-host.invalid"]) == 1
        assert "не удалось занять адрес no-such-host.invalid:8000" in capsys.readouterr().err
        assert fake == []

    def test_free_address(self):
        assert bind_error("127.0.0.1", 0) is None

    def test_closes_queue_on_error(self, fake, server, monkeypatch, tmp_path):
        directories = []

        def run_server(app, **kwargs):
            directories.append(app.state.settings.jobs.directory)
            raise SystemExit(1)

        monkeypatch.setattr(uvicorn, "run", run_server)
        assert run(["serve", "--data-dir", str(tmp_path)]) == 1
        assert not directories[0].exists()

    def test_help(self, capsys):
        assert run(["serve", "--help"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("использование: stt-ru serve")
        assert "STT_RU_TOKEN" in out and "--max-duration" in out and "--model" in out


@pytest.mark.network
def test_real_model(capsys):
    main(["transcribe", str(CALL), "-q"])
    out = capsys.readouterr().out
    assert "Канал 1: Наталья, благодарю вас за ожидание." in out
    assert "до свидания" in out.lower()


@pytest.mark.network
def test_real_model_evaluate(tmp_path, capsys):
    manifest = tmp_path / "eval.jsonl"
    rows = [
        {"audio": str(CALL), "channel": 0, "text": "Угу. Угу. До свидания."},
        {
            "audio": str(CALL),
            "channel": 1,
            "text": "Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок "
            "переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, "
            "на линии. Я, в свою очередь, с вами прощаюсь. Всего доброго, до свидания.",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    main(["evaluate", str(manifest), "-q", "-o", str(tmp_path / "eval.json")])
    (model,) = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))["models"]
    assert model["failed"] == 0
    assert model["wer"] <= 0.1
    assert model["speed"] > 1
    assert capsys.readouterr().out.startswith("Модель")


class TestTranscribeChecks:
    def test_paths_after_options(self, fake, files, tmp_path):
        out = tmp_path / "out"
        main(
            ["transcribe", str(files / "a.wav"), "-o", str(out), "-f", "txt", str(files / "b.wav")]
        )
        assert sorted(path.name for path in out.iterdir()) == ["a.txt", "b.txt"]

    def test_unknown_option_after_paths(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "--bogus"]) == 2
        assert "неизвестные аргументы: --bogus" in capsys.readouterr().err

    def test_extra_argument_of_other_command(self, fake, files, capsys):
        assert run(["serve", "лишнее"]) == 2
        assert "неизвестные аргументы: лишнее" in capsys.readouterr().err

    def test_quantization_any_case(self, fake, files):
        main(["transcribe", str(files / "a.wav"), "-q", "--quantization", "INT8"])
        main(["transcribe", str(files / "a.wav"), "-q", "--quantization", "None"])
        assert [transcriber.kwargs["quantization"] for transcriber in fake] == ["int8", None]

    def test_unknown_quantization(self, fake, files, capsys):
        assert run(["transcribe", str(files / "a.wav"), "--quantization", "int4"]) == 2
        assert "аргумент --quantization: неизвестное значение 'int4'" in capsys.readouterr().err
        assert fake == []

    def test_warning_in_one_line(self, capsys):
        cli._show_warning(UserWarning("У модели m нет весов int8"), UserWarning, "x.py", 70)
        assert capsys.readouterr().err == "stt-ru: предупреждение: У модели m нет весов int8\n"

    def test_warning_with_closed_stderr(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stderr", None)
        cli._show_warning(UserWarning("текст"), UserWarning, "x.py", 70)
        assert capsys.readouterr().out == ""

    def test_warnings_shown_by_command(self, fake, files):
        with warnings.catch_warnings():
            main(["transcribe", str(files / "a.wav"), "-q"])
            assert warnings.showwarning is cli._show_warning

    def test_empty_formats(self, fake, files, tmp_path, capsys):
        args = ["transcribe", str(files), "-o", str(tmp_path), "-f", ",", "--delete-source"]
        assert run(args) == 2
        assert "не задано ни одного формата" in capsys.readouterr().err
        assert (files / "a.wav").exists() and fake == []

    def test_output_dir_is_file(self, fake, files, tmp_path, capsys):
        target = tmp_path / "out"
        target.write_text("", encoding="utf-8")
        assert run(["transcribe", str(files), "-o", str(target)]) == 2
        assert f"{target} - файл, а нужен каталог" in capsys.readouterr().err
        assert fake == []

    def test_summary_is_directory(self, fake, files, tmp_path, capsys):
        (tmp_path / "all.csv").mkdir()
        assert run(["transcribe", str(files), "--summary", str(tmp_path / "all.csv")]) == 2
        assert "- каталог, а нужен файл" in capsys.readouterr().err

    def test_empty_output_dir(self, fake, files, capsys):
        assert run(["transcribe", str(files), "-o", ""]) == 2
        assert "Путь результата не задан" in capsys.readouterr().err

    def test_closed_stdout(self, fake, files, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdout", None)
        assert run(["transcribe", str(files / "a.wav")]) == 1
        assert "stt-ru: ошибка: вывод закрыт" in capsys.readouterr().err
        assert fake == []

    def test_same_names_printed_with_paths(self, fake, tmp_path, capsys):
        first, second = tmp_path / "d1" / "a.wav", tmp_path / "d2" / "a.wav"
        for path in (first, second):
            path.parent.mkdir()
            path.write_bytes(b"audio")
        main(["transcribe", str(first), str(second), "-f", "jsonl", "-q"])
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert sorted({row["source"] for row in rows}) == [str(first), str(second)]
        assert (
            run(["transcribe", str(first), str(second), "--summary", str(tmp_path / "a.csv")]) == 2
        )
        assert "совпадают имена результатов" in capsys.readouterr().err

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="права каталогов POSIX"
    )
    def test_unreadable_subdirectory(self, fake, files, tmp_path, capsys):
        locked = files / "locked"
        locked.mkdir()
        locked.chmod(0)
        out = tmp_path / "out"
        try:
            assert run(["transcribe", str(files), "-r", "-o", str(out), "-f", "txt", "-q"]) == 1
        finally:
            locked.chmod(0o755)
        assert f"stt-ru: ошибка: Нет доступа к каталогу: {locked}" in capsys.readouterr().err
        assert sorted(path.name for path in out.iterdir()) == ["a.txt", "b.txt"]

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="права каталогов POSIX"
    )
    def test_unreadable_subdirectory_only(self, fake, tmp_path, capsys):
        locked = tmp_path / "in" / "locked"
        locked.mkdir(parents=True)
        locked.chmod(0)
        try:
            assert run(["transcribe", str(tmp_path / "in"), "-r"]) == 1
        finally:
            locked.chmod(0o755)
        assert "Файлов для распознавания нет" in capsys.readouterr().err
        assert fake == []

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="права каталогов POSIX"
    )
    def test_unreadable_argument(self, fake, files, capsys):
        files.chmod(0)
        try:
            assert run(["transcribe", str(files)]) == 2
        finally:
            files.chmod(0o755)
        assert f"Нет доступа к каталогу: {files}" in capsys.readouterr().err
        assert fake == []

    def test_watch_starts_with_name_conflicts(self, fake, files, tmp_path, monkeypatch):
        (files / "a.mp3").write_bytes(b"audio")
        calls = []
        monkeypatch.setattr(
            cli, "watch", lambda batch, paths, **kwargs: calls.append(paths) or WatchReport()
        )
        main(["transcribe", str(files), "-o", str(tmp_path / "out"), "--watch", "-q"])
        assert calls == [[str(files)]]
