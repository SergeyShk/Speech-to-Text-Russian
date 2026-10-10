import threading
import time
from pathlib import Path

import pytest

from stt_ru import Segment, Transcript
from stt_ru.batch import (
    Batch,
    Job,
    Report,
    WatchReport,
    check_options,
    check_watch,
    find_jobs,
    summary_line,
    watch,
    write_atomic,
)
from stt_ru.exceptions import AudioError, ParameterError
from stt_ru.export import dumps


class FakeTranscriber:
    def __init__(self, delay=0.0):
        self.calls = []
        self.delay = delay
        self._lock = threading.Lock()

    def transcribe(self, path):
        path = Path(path)
        with self._lock:
            self.calls.append(path)
        time.sleep(self.delay)
        if "broken" in path.name:
            raise AudioError(f"ffmpeg не прочитал файл {path}")
        segment = Segment(0.5, 1.0, f"Текст {path.stem}.", 0)
        return Transcript((segment,), 1.0, 2, "fake", path.name)


def touch(path, data=b"audio"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def quiet(**kwargs):
    return {"log": lambda m: None, "errors": lambda m: None, **kwargs}


@pytest.fixture
def calls(tmp_path):
    root = tmp_path / "calls"
    touch(root / "a.wav")
    touch(root / "b.MP3")
    touch(root / "notes.txt")
    touch(root / "sub" / "c.ogg")
    touch(root / "._a.wav")
    touch(root / ".Trashes" / "d.wav")
    return root


class TestFindJobs:
    def test_directory(self, calls):
        assert find_jobs([calls]) == [
            Job(calls / "a.wav", Path("a"), "a.wav"),
            Job(calls / "b.MP3", Path("b"), "b.MP3"),
        ]

    def test_recursive(self, calls):
        jobs = find_jobs([calls], recursive=True)
        assert [(job.name, job.label) for job in jobs] == [
            (Path("a"), "a.wav"),
            (Path("b"), "b.MP3"),
            (Path("sub/c"), "sub/c.ogg"),
        ]

    def test_file_with_any_suffix(self, calls):
        assert find_jobs([calls / "notes.txt"]) == [
            Job(calls / "notes.txt", Path("notes"), "notes.txt")
        ]

    def test_hidden_file_given_explicitly(self, calls):
        assert [job.name for job in find_jobs([calls / "._a.wav"])] == [Path("._a")]

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Файл или каталог не найден"):
            find_jobs([tmp_path / "missing.wav"])

    def test_missing_not_strict(self, calls, tmp_path):
        messages = []
        jobs = find_jobs([tmp_path / "missing", calls], strict=False, log=messages.append)
        assert len(jobs) == 2
        assert messages == [f"Файл или каталог не найден: {tmp_path / 'missing'}"]

    def test_same_names(self, tmp_path):
        first = touch(tmp_path / "x" / "call.wav")
        second = touch(tmp_path / "y" / "Call.mp3")
        with pytest.raises(ParameterError, match="совпадают имена результатов"):
            find_jobs([first, second])

    def test_same_names_not_strict(self, tmp_path):
        folder = tmp_path / "in"
        touch(folder / "call.mp3")
        touch(folder / "call.wav")
        messages = []
        jobs = find_jobs([folder], strict=False, log=messages.append)
        assert [job.source.name for job in jobs] == ["call.mp3"]
        assert messages == [
            f"У файлов {folder / 'call.mp3'} и {folder / 'call.wav'} совпадают имена "
            f"результатов: {folder / 'call.wav'} пропущен"
        ]

    def test_same_file_by_different_paths(self, calls, monkeypatch):
        monkeypatch.chdir(calls)
        link = calls / "link.wav"
        link.symlink_to(calls / "a.wav")
        jobs = find_jobs(["a.wav", calls / "a.wav", link])
        assert [job.source for job in jobs] == [Path("a.wav")]

    def test_empty_directory(self, tmp_path):
        assert find_jobs([tmp_path]) == []


class TestOptions:
    def test_wrong_format(self):
        with pytest.raises(ParameterError, match="Форматы не поддерживаются: docx"):
            check_options(["srt", "docx"])

    def test_wrong_summary(self, tmp_path):
        with pytest.raises(ParameterError, match="Общий файл должен быть CSV или JSONL"):
            check_options(["srt"], tmp_path / "all.srt")

    def test_wrong_workers(self):
        with pytest.raises(ParameterError, match="Число потоков должно быть положительным: 0"):
            check_options(["srt"], workers=0)

    def test_batch_checks(self, tmp_path):
        with pytest.raises(ParameterError, match="docx"):
            Batch(FakeTranscriber(), tmp_path, ["docx"])

    def test_delete_without_results(self):
        with pytest.raises(ParameterError, match="только если результаты сохраняются"):
            Batch(FakeTranscriber(), None, [], delete_source=True)

    @pytest.mark.parametrize(
        ("interval", "min_age", "message"),
        [
            (0, 0, "Интервал просмотра должен быть положительным"),
            (1, -1, "не может быть отрицательным"),
        ],
    )
    def test_watch(self, interval, min_age, message):
        with pytest.raises(ParameterError, match=message):
            check_watch(interval, min_age)


class TestWriteAtomic:
    def test_writes(self, tmp_path):
        write_atomic(tmp_path / "sub" / "a.txt", b"data")
        assert (tmp_path / "sub" / "a.txt").read_bytes() == b"data"
        assert list((tmp_path / "sub").iterdir()) == [tmp_path / "sub" / "a.txt"]

    def test_failure_leaves_nothing(self, tmp_path, monkeypatch):
        def fail(self, target):
            raise OSError("диск переполнен")

        monkeypatch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="диск переполнен"):
            write_atomic(tmp_path / "a.txt", b"data")
        assert list(tmp_path.iterdir()) == []

    def test_keeps_old_file_on_failure(self, tmp_path, monkeypatch):
        touch(tmp_path / "a.txt", b"old")
        monkeypatch.setattr(
            Path, "replace", lambda self, target: (_ for _ in ()).throw(KeyboardInterrupt)
        )
        with pytest.raises(KeyboardInterrupt):
            write_atomic(tmp_path / "a.txt", b"new")
        assert (tmp_path / "a.txt").read_bytes() == b"old"
        assert list(tmp_path.iterdir()) == [tmp_path / "a.txt"]


class TestBatch:
    def test_outputs(self, tmp_path):
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["srt", "json"])
        job = Job(Path("x.wav"), Path("sub/call.part1"), "sub/call.part1.wav")
        assert batch.outputs(job) == [
            tmp_path / "out/sub/call.part1.srt",
            tmp_path / "out/sub/call.part1.json",
        ]
        assert Batch(FakeTranscriber(), None, ["srt"]).outputs(job) == []

    def test_run(self, calls, tmp_path):
        messages = []
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["srt", "csv"], **quiet(log=messages.append))
        report = batch.run(find_jobs([calls], recursive=True))
        assert sorted(report.done) == sorted(
            [calls / "a.wav", calls / "b.MP3", calls / "sub/c.ogg"]
        )
        transcript = transcriber.transcribe(calls / "sub/c.ogg")
        assert (tmp_path / "out/sub/c.srt").read_bytes() == dumps(transcript, "srt")
        assert (tmp_path / "out/sub/c.csv").read_bytes() == dumps(transcript, "csv")
        assert any("фрагментов 1, 1.0 с записи за" in message for message in messages)

    def test_skip_existing(self, calls, tmp_path):
        out = tmp_path / "out"
        touch(out / "a.srt")
        touch(out / "b.srt")
        touch(out / "b.json")
        messages = []
        transcriber = FakeTranscriber()
        report = Batch(transcriber, out, ["srt", "json"], **quiet(log=messages.append)).run(
            find_jobs([calls])
        )
        assert (report.skipped, report.done) == ([calls / "b.MP3"], [calls / "a.wav"])
        assert transcriber.calls == [calls / "a.wav"]
        assert f"{calls / 'b.MP3'}: пропущен, результаты уже есть" in messages

    def test_skipped_source_deleted(self, calls, tmp_path):
        touch(tmp_path / "out" / "a.txt")
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True, **quiet())
        assert batch.run(find_jobs([calls / "a.wav"])).skipped == [calls / "a.wav"]
        assert not (calls / "a.wav").exists()

    def test_overwrite(self, calls, tmp_path):
        out = tmp_path / "out"
        touch(out / "a.srt", b"old")
        batch = Batch(FakeTranscriber(), out, ["srt"], overwrite=True, **quiet())
        assert batch.run(find_jobs([calls / "a.wav"])).done == [calls / "a.wav"]
        assert (out / "a.srt").read_bytes() != b"old"

    def test_delete_source(self, calls, tmp_path):
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True, **quiet())
        batch.run(find_jobs([calls / "a.wav"]))
        assert not (calls / "a.wav").exists()
        assert (tmp_path / "out/a.txt").exists()

    def test_source_removed_during_work(self, calls, tmp_path):
        class Removing(FakeTranscriber):
            def transcribe(self, path):
                result = super().transcribe(path)
                Path(path).unlink()
                return result

        batch = Batch(Removing(), tmp_path / "out", ["txt"], delete_source=True, **quiet())
        report = batch.run(find_jobs([calls / "a.wav"]))
        assert (report.done, report.failed) == ([calls / "a.wav"], {})

    def test_delete_failure(self, calls, tmp_path, monkeypatch):
        errors = []

        def refuse(self, missing_ok=False):
            raise PermissionError("занят")

        monkeypatch.setattr(Path, "unlink", refuse)
        batch = Batch(
            FakeTranscriber(),
            tmp_path / "out",
            ["txt"],
            delete_source=True,
            **quiet(errors=errors.append),
        )
        report = batch.run(find_jobs([calls / "a.wav"]))
        assert report.done == [calls / "a.wav"]
        assert report.failed == {calls / "a.wav": "не удалось удалить исходник: занят"}
        assert errors == [f"{calls / 'a.wav'}: не удалось удалить исходник - занят"]

    def test_failure_does_not_stop(self, calls, tmp_path):
        broken = touch(calls / "broken.wav")
        errors, messages = [], []
        batch = Batch(
            FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True,
            **quiet(log=messages.append, errors=errors.append),
        )  # fmt: skip
        report = batch.run(find_jobs([calls]))
        assert report.failed == {broken: f"ffmpeg не прочитал файл {broken}"}
        assert len(report.done) == 2
        assert broken.exists()
        assert not (tmp_path / "out/broken.txt").exists()
        assert len(errors) == 1 and f"{broken}: ошибка - ffmpeg не прочитал файл" in errors[0]
        assert not any("ошибка" in message for message in messages)

    def test_summary_failure_retried(self, calls, tmp_path, monkeypatch):
        out = tmp_path / "out"
        summary = tmp_path / "all.csv"
        batch = Batch(
            FakeTranscriber(), out, ["txt"], summary=summary, delete_source=True, **quiet()
        )

        def fail(self, transcript):
            raise PermissionError("файл открыт в Excel")

        monkeypatch.setattr(Batch, "_append_summary", fail)
        report = batch.run(find_jobs([calls / "a.wav"]))
        assert report.failed == {calls / "a.wav": "файл открыт в Excel"}
        assert not (out / "a.txt").exists()
        assert (calls / "a.wav").exists()
        monkeypatch.undo()
        assert batch.run(find_jobs([calls / "a.wav"])).done == [calls / "a.wav"]
        assert "Текст a." in summary.read_text(encoding="utf-8-sig")

    def test_order(self, calls, tmp_path):
        events = []
        out = tmp_path / "out"
        summary = tmp_path / "all.jsonl"

        def on_done(job, transcript):
            events.append(
                ("print", summary.exists(), (out / "a.txt").exists(), job.source.exists())
            )

        batch = Batch(
            FakeTranscriber(),
            out,
            ["txt"],
            summary=summary,
            delete_source=True,
            **quiet(on_done=on_done),
        )
        batch.run(find_jobs([calls / "a.wav"]))
        assert events == [("print", True, True, True)]
        assert not (calls / "a.wav").exists()

    def test_on_done(self, calls):
        done = []
        batch = Batch(
            FakeTranscriber(),
            None,
            [],
            **quiet(on_done=lambda job, t: done.append((job.name, t.text))),
        )
        batch.run(find_jobs([calls]))
        assert sorted(done) == [(Path("a"), "Текст a."), (Path("b"), "Текст b.")]

    def test_workers(self, tmp_path):
        files = [touch(tmp_path / "in" / f"f{i}.wav") for i in range(8)]
        transcriber = FakeTranscriber()
        report = Batch(transcriber, tmp_path / "out", ["txt"], workers=4, **quiet()).run(
            find_jobs([tmp_path / "in"])
        )
        assert sorted(report.done) == files
        assert sorted(transcriber.calls) == files

    def test_summary_csv(self, calls, tmp_path):
        summary = tmp_path / "reports" / "all.csv"
        for _ in range(2):
            Batch(FakeTranscriber(), None, [], summary=summary, **quiet()).run(
                find_jobs([calls], recursive=True)
            )
        data = summary.read_bytes()
        assert data.startswith(b"\xef\xbb\xbfAudio File,Start,End,Name,Text\n")
        assert data.count(b"Audio File") == 1
        assert data.count(b"\xef\xbb\xbf") == 1
        text = data.decode("utf-8-sig")
        assert text.count("Канал 0") == 6
        assert "sub/c.ogg,500,1000" in text

    def test_summary_jsonl(self, calls, tmp_path):
        summary = tmp_path / "all.jsonl"
        Batch(FakeTranscriber(), None, [], summary=summary, **quiet()).run(find_jobs([calls]))
        lines = summary.read_text(encoding="utf-8").splitlines()
        assert ['"source": "a.wav"' in line or '"source": "b.MP3"' in line for line in lines] == [
            True,
            True,
        ]

    def test_interrupt_finishes_running(self, tmp_path):
        for i in range(6):
            touch(tmp_path / "in" / f"f{i}.wav")
        messages = []
        transcriber = FakeTranscriber(delay=0.2)
        out = tmp_path / "out"
        interrupts = iter([True])

        def on_done(job, transcript):
            if next(interrupts, False):
                raise KeyboardInterrupt

        batch = Batch(
            transcriber, out, ["txt"], workers=2, **quiet(log=messages.append, on_done=on_done)
        )
        with pytest.raises(KeyboardInterrupt):
            batch.run(find_jobs([tmp_path / "in"]))
        assert 2 <= len(transcriber.calls) < 6
        assert sorted(path.name for path in out.iterdir()) == sorted(
            f"{p.stem}.txt" for p in transcriber.calls
        )
        assert any(
            message.startswith("Прерывание: дожидаюсь файлов в работе (") for message in messages
        )

    def test_broken_pipe_stops(self, tmp_path):
        for i in range(5):
            touch(tmp_path / "in" / f"f{i}.wav")
        transcriber = FakeTranscriber(delay=0.05)

        def on_done(job, transcript):
            raise BrokenPipeError

        batch = Batch(transcriber, None, [], **quiet(on_done=on_done))
        with pytest.raises(BrokenPipeError):
            batch.run(find_jobs([tmp_path / "in"]))
        time.sleep(0.2)
        assert len(transcriber.calls) <= 2


class TestWatch:
    def test_waits_for_stable_files(self, tmp_path):
        folder = tmp_path / "in"
        audio = touch(folder / "a.wav")
        mtime = audio.stat().st_mtime
        times = iter([mtime + 1, mtime + 20, mtime + 40])
        pauses = []
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], **quiet())
        report = watch(
            batch,
            [folder],
            interval=5,
            min_age=10,
            rounds=3,
            sleep=pauses.append,
            clock=lambda: next(times),
        )
        assert transcriber.calls == [audio]
        assert report == WatchReport(done=1, skipped=0, failed={})
        assert pauses == [5, 5]

    def test_failed_file_retried_only_after_change(self, tmp_path):
        folder = tmp_path / "in"
        broken = touch(folder / "broken.wav")
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], **quiet())
        changes = iter([lambda: None, lambda: broken.write_bytes(b"new audio")])
        report = watch(batch, [folder], min_age=0, rounds=3, sleep=lambda s: next(changes)())
        assert transcriber.calls == [broken, broken]
        assert list(report.failed) == [broken]

    def test_forgets_removed_files(self, tmp_path):
        folder = tmp_path / "in"
        broken = touch(folder / "broken.wav")
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], **quiet())
        report = watch(batch, [folder], min_age=0, rounds=2, sleep=lambda s: broken.unlink())
        assert report.failed == {}

    def test_missing_directory_and_conflicts(self, tmp_path):
        folder = tmp_path / "in"
        touch(folder / "call.mp3")
        touch(folder / "call.wav")
        errors = []
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], **quiet(errors=errors.append))
        report = watch(
            batch, [folder, tmp_path / "gone"], min_age=0, rounds=3, sleep=lambda s: None
        )
        assert transcriber.calls == [folder / "call.mp3"]
        assert report.done == 1
        assert len(errors) == 2
        assert errors[0] == f"Файл или каталог не найден: {tmp_path / 'gone'}"
        assert "совпадают имена результатов" in errors[1]

    def test_scan_error(self, tmp_path, monkeypatch):
        from stt_ru import batch as module

        def fail(*args, **kwargs):
            raise PermissionError("нет доступа")

        monkeypatch.setattr(module, "find_jobs", fail)
        errors = []
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], **quiet(errors=errors.append))
        assert watch(batch, [tmp_path], rounds=2, sleep=lambda s: None) == WatchReport()
        assert errors == ["Не удалось просмотреть каталоги: нет доступа"]

    def test_file_removed_while_scanning(self, tmp_path, monkeypatch):
        from stt_ru import batch as module

        gone = tmp_path / "in" / "gone.wav"
        monkeypatch.setattr(
            module, "find_jobs", lambda *args, **kwargs: [Job(gone, Path("gone"), "gone.wav")]
        )
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], **quiet())
        assert watch(batch, [tmp_path], rounds=1, sleep=lambda s: None).done == 0
        assert transcriber.calls == []

    def test_new_files_and_deleted_sources(self, tmp_path):
        folder = tmp_path / "in"
        folder.mkdir()
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True, **quiet())
        arrivals = iter([lambda: touch(folder / "a.wav"), lambda: touch(folder / "b.wav")])
        report = watch(batch, [folder], min_age=0, rounds=3, sleep=lambda s: next(arrivals)())
        assert report.done == 2
        assert list(folder.iterdir()) == []

    def test_wrong_options(self, tmp_path):
        batch = Batch(FakeTranscriber(), tmp_path, ["txt"])
        with pytest.raises(ParameterError):
            watch(batch, [tmp_path], interval=0)


def test_summary_line():
    report = Report(done=[Path("a")], skipped=[Path("b"), Path("c")], failed={Path("d"): "x"})
    assert summary_line(report) == "Готово: распознано 1 из 4, пропущено 2, с ошибкой 1"
    assert (
        summary_line(WatchReport(3, 1, {}))
        == "Готово: распознано 3 из 4, пропущено 1, с ошибкой 0"
    )
