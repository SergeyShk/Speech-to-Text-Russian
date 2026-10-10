import threading
from pathlib import Path

import pytest

from stt_ru import Segment, Transcript
from stt_ru.batch import Batch, Job, Report, find_jobs, summary_line, watch
from stt_ru.exceptions import AudioError, ParameterError
from stt_ru.export import dumps


class FakeTranscriber:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def transcribe(self, path):
        path = Path(path)
        with self._lock:
            self.calls.append(path)
        if "broken" in path.name:
            raise AudioError(f"ffmpeg не прочитал файл {path}")
        segment = Segment(0.5, 1.0, f"Текст {path.stem}.", 0)
        return Transcript((segment,), 1.0, 2, "fake", path.name)


def touch(path, data=b"audio"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def calls(tmp_path):
    root = tmp_path / "calls"
    touch(root / "a.wav")
    touch(root / "b.MP3")
    touch(root / "notes.txt")
    touch(root / "sub" / "c.ogg")
    return root


class TestFindJobs:
    def test_directory(self, calls):
        jobs = find_jobs([calls])
        assert jobs == [Job(calls / "a.wav", Path("a")), Job(calls / "b.MP3", Path("b"))]

    def test_recursive(self, calls):
        names = [job.name for job in find_jobs([calls], recursive=True)]
        assert names == [Path("a"), Path("b"), Path("sub/c")]

    def test_file_with_any_suffix(self, calls):
        assert find_jobs([calls / "notes.txt"]) == [Job(calls / "notes.txt", Path("notes"))]

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Файл или каталог не найден"):
            find_jobs([tmp_path / "missing.wav"])

    def test_same_names(self, tmp_path):
        first = touch(tmp_path / "x" / "call.wav")
        second = touch(tmp_path / "y" / "call.wav")
        with pytest.raises(ParameterError, match="совпадают имена результатов"):
            find_jobs([first, second])

    def test_same_file_twice(self, calls):
        assert len(find_jobs([calls / "a.wav", calls / "a.wav"])) == 1

    def test_empty_directory(self, tmp_path):
        assert find_jobs([tmp_path]) == []


class TestBatch:
    def test_wrong_format(self, tmp_path):
        with pytest.raises(ParameterError, match="Форматы не поддерживаются: docx"):
            Batch(FakeTranscriber(), tmp_path, ["srt", "docx"])

    def test_wrong_summary(self, tmp_path):
        with pytest.raises(ParameterError, match="Общий файл должен быть CSV или JSONL"):
            Batch(FakeTranscriber(), tmp_path, ["srt"], summary=tmp_path / "all.srt")

    def test_wrong_workers(self, tmp_path):
        with pytest.raises(ParameterError, match="Число потоков должно быть положительным: 0"):
            Batch(FakeTranscriber(), tmp_path, ["srt"], workers=0)

    def test_outputs(self, tmp_path):
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["srt", "json"])
        job = Job(Path("x.wav"), Path("sub/call.part1"))
        assert batch.outputs(job) == [
            tmp_path / "out/sub/call.part1.srt",
            tmp_path / "out/sub/call.part1.json",
        ]
        assert Batch(FakeTranscriber(), None, []).outputs(job) == []

    def test_run(self, calls, tmp_path):
        messages = []
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["srt", "csv"], log=messages.append)
        report = batch.run(find_jobs([calls], recursive=True))
        assert sorted(report.done) == sorted(
            [calls / "a.wav", calls / "b.MP3", calls / "sub/c.ogg"]
        )
        transcript = transcriber.transcribe(calls / "sub/c.ogg")
        assert (tmp_path / "out/sub/c.srt").read_bytes() == dumps(transcript, "srt")
        assert (tmp_path / "out/sub/c.csv").read_bytes() == dumps(transcript, "csv")
        assert any("фрагментов 1, 1.0 с записи за" in message for message in messages)
        assert all(message.startswith("[") for message in messages)

    def test_skip_existing(self, calls, tmp_path):
        out = tmp_path / "out"
        touch(out / "a.srt")
        touch(out / "b.srt")
        touch(out / "b.json")
        messages = []
        transcriber = FakeTranscriber()
        report = Batch(transcriber, out, ["srt", "json"], log=messages.append).run(
            find_jobs([calls])
        )
        assert report.skipped == [calls / "b.MP3"]
        assert report.done == [calls / "a.wav"]
        assert transcriber.calls == [calls / "a.wav"]
        assert f"{calls / 'b.MP3'}: пропущен, результаты уже есть" in messages

    def test_overwrite(self, calls, tmp_path):
        out = tmp_path / "out"
        touch(out / "a.srt", b"old")
        batch = Batch(FakeTranscriber(), out, ["srt"], overwrite=True, log=lambda m: None)
        assert batch.run(find_jobs([calls / "a.wav"])).done == [calls / "a.wav"]
        assert (out / "a.srt").read_bytes() != b"old"

    def test_delete_source(self, calls, tmp_path):
        batch = Batch(FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True, log=print)
        batch.run(find_jobs([calls / "a.wav"]))
        assert not (calls / "a.wav").exists()
        assert (tmp_path / "out/a.txt").exists()

    def test_failure_does_not_stop(self, calls, tmp_path):
        broken = touch(calls / "broken.wav")
        messages = []
        batch = Batch(
            FakeTranscriber(), tmp_path / "out", ["txt"], delete_source=True, log=messages.append
        )
        report = batch.run(find_jobs([calls]))
        assert report.failed == {broken: f"ffmpeg не прочитал файл {broken}"}
        assert len(report.done) == 2
        assert broken.exists()
        assert not (tmp_path / "out/broken.txt").exists()
        assert any(f"{broken}: ошибка - ffmpeg не прочитал файл" in m for m in messages)

    def test_on_done(self, calls, tmp_path):
        done = []
        batch = Batch(
            FakeTranscriber(),
            None,
            [],
            on_done=lambda job, t: done.append((job.name, t.text)),
            log=print,
        )
        batch.run(find_jobs([calls]))
        assert sorted(done) == [(Path("a"), "Текст a."), (Path("b"), "Текст b.")]

    def test_workers(self, tmp_path):
        files = [touch(tmp_path / "in" / f"f{i}.wav") for i in range(8)]
        transcriber = FakeTranscriber()
        report = Batch(transcriber, tmp_path / "out", ["txt"], workers=4, log=lambda m: None).run(
            find_jobs([tmp_path / "in"])
        )
        assert sorted(report.done) == files
        assert sorted(transcriber.calls) == files

    def test_summary_csv(self, calls, tmp_path):
        summary = tmp_path / "reports" / "all.csv"
        for _ in range(2):
            Batch(FakeTranscriber(), None, [], summary=summary, log=lambda m: None).run(
                find_jobs([calls])
            )
        data = summary.read_bytes()
        assert data.startswith(b"\xef\xbb\xbfAudio File,Start,End,Name,Text\n")
        assert data.count(b"Audio File") == 1
        assert data.count(b"\xef\xbb\xbf") == 1
        assert data.decode("utf-8-sig").count("Канал 0") == 4

    def test_summary_jsonl(self, calls, tmp_path):
        summary = tmp_path / "all.jsonl"
        Batch(FakeTranscriber(), None, [], summary=summary, log=lambda m: None).run(
            find_jobs([calls])
        )
        lines = summary.read_text(encoding="utf-8").splitlines()
        assert [line.count('"source"') for line in lines] == [1, 1]

    def test_interrupt_cancels(self, tmp_path):
        touch(tmp_path / "in" / "a.wav")

        class Interrupted(FakeTranscriber):
            def transcribe(self, path):
                raise KeyboardInterrupt

        batch = Batch(Interrupted(), tmp_path / "out", ["txt"], log=lambda m: None)
        with pytest.raises(KeyboardInterrupt):
            batch.run(find_jobs([tmp_path / "in"]))


class TestWatch:
    def test_waits_for_stable_files(self, tmp_path):
        folder = tmp_path / "in"
        audio = touch(folder / "a.wav")
        mtime = audio.stat().st_mtime
        times = iter([mtime + 1, mtime + 20, mtime + 40])
        pauses = []
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], log=lambda m: None)
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
        assert report.done == [audio]
        assert pauses == [5, 5]

    def test_failed_file_retried_only_after_change(self, tmp_path):
        folder = tmp_path / "in"
        broken = touch(folder / "broken.wav")
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], log=lambda m: None)
        changes = iter([lambda: None, lambda: broken.write_bytes(b"new audio")])

        def sleep(seconds):
            next(changes)()

        report = watch(batch, [folder], min_age=0, rounds=3, sleep=sleep)
        assert transcriber.calls == [broken, broken]
        assert list(report.failed) == [broken]

    def test_file_removed_while_scanning(self, tmp_path, monkeypatch):
        from stt_ru import batch as module

        gone = tmp_path / "in" / "gone.wav"
        monkeypatch.setattr(
            module, "find_jobs", lambda paths, recursive: [Job(gone, Path("gone"))]
        )
        transcriber = FakeTranscriber()
        batch = Batch(transcriber, tmp_path / "out", ["txt"], log=lambda m: None)
        assert watch(batch, [tmp_path], rounds=1, sleep=lambda s: None).done == []
        assert transcriber.calls == []

    def test_new_files_and_deleted_sources(self, tmp_path):
        folder = tmp_path / "in"
        transcriber = FakeTranscriber()
        batch = Batch(
            transcriber, tmp_path / "out", ["txt"], delete_source=True, log=lambda m: None
        )
        arrivals = iter([lambda: touch(folder / "a.wav"), lambda: touch(folder / "b.wav")])

        def sleep(seconds):
            next(arrivals)()

        folder.mkdir()
        report = watch(batch, [folder], min_age=0, rounds=3, sleep=sleep)
        assert sorted(path.name for path in report.done) == ["a.wav", "b.wav"]
        assert list(folder.iterdir()) == []


def test_summary_line():
    report = Report(done=[Path("a")], skipped=[Path("b"), Path("c")], failed={Path("d"): "x"})
    assert summary_line(report) == "Готово: распознано 1 из 4, пропущено 2, с ошибкой 1"
