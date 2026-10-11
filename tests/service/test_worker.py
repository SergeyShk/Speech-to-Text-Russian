import io
import os
import sys
import threading

import pytest

from stt_ru.exceptions import ParameterError
from stt_ru.service import JobQueue, QueueFullError, worker

from .conftest import job_folders, wait, wav_bytes


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def submit(jobs, name="call.wav", content=None):
    return jobs.submit(name, io.BytesIO(wav_bytes() if content is None else content))


class TestJobQueue:
    def test_done(self, jobs, transcriber):
        messages = []
        jobs.log = messages.append
        job = submit(jobs, "звонок.wav")
        assert (job.status, job.size, job.path.name) == ("queued", len(wav_bytes()), "source.wav")
        assert job.path.parent.parent == jobs.directory
        done = wait(jobs, job.id)
        assert done.status == "done"
        assert (done.duration, done.channels) == (1.0, 2)
        assert [len(row) for row in done.peaks] == [2000, 2000]
        assert max(done.peaks[0]) == 0.5 and max(done.peaks[1]) == 0.25
        assert done.transcript.source == "звонок.wav"
        assert done.started <= done.finished
        assert transcriber.sources == ["звонок.wav"]
        assert messages[0] == f"[{job.id}] звонок.wav: в очереди"
        assert messages[1].startswith(f"[{job.id}] звонок.wav: фрагментов 2, 1.0 с записи за ")

    def test_ids_are_random(self, jobs):
        first, second = submit(jobs), submit(jobs)
        assert first.id != second.id and len(first.id) >= 20

    @pytest.mark.parametrize(
        ("name", "filename", "stored"),
        [
            ("C:\\Users\\..\\Звонок 1.WAV", "Звонок 1.WAV", "source.wav"),
            ("../../etc/passwd", "passwd", "source"),
            ("a\x00b\nc.mp3", "abc.mp3", "source.mp3"),
            ("  ", "audio", "source"),
            ("запись.очень-длинное-расширение", "запись.очень-длинное-расширение", "source"),
            ("x" * 300 + ".ogg", "x" * 196 + ".ogg", "source.ogg"),
        ],
    )
    def test_filename(self, jobs, name, filename, stored):
        job = submit(jobs, name)
        assert (job.filename, job.path.name) == (filename, stored)

    def test_failed(self, jobs):
        job = jobs.submit("broken.wav", io.BytesIO(b"not audio"))
        failed = wait(jobs, job.id)
        assert failed.status == "failed"
        assert failed.error.startswith("ffmpeg не прочитал файл broken.wav: ")
        assert str(jobs.directory.name) not in failed.error and "source" not in failed.error
        assert failed.duration is None

    def test_broken_log(self, jobs):
        def log(message):
            raise BrokenPipeError(32, "Broken pipe")

        jobs.log = log
        first = submit(jobs)
        assert wait(jobs, first.id).status == "done"
        second = submit(jobs)
        assert wait(jobs, second.id).status == "done"
        assert jobs.delete(first.id)
        assert all(thread.is_alive() for thread in jobs._threads)

    def test_transcription_error(self, jobs):
        failed = wait(jobs, submit(jobs, "broken.wav").id)
        assert (failed.status, failed.error, failed.duration) == (
            "failed",
            "модель не приняла запись",
            1.0,
        )

    def test_error_without_message(self, jobs):
        assert wait(jobs, submit(jobs, "silent.wav").id).error == "RuntimeError"

    def test_max_duration(self, transcriber, tmp_path):
        jobs = JobQueue(transcriber, tmp_path, max_duration=0.5)
        try:
            failed = wait(jobs, submit(jobs).id)
        finally:
            jobs.close()
        assert (failed.status, failed.error) == ("failed", "Запись длиннее 0.5 с")
        assert transcriber.sources == []

    def test_position_and_full_queue(self, transcriber, tmp_path):
        transcriber.gate.clear()
        jobs = JobQueue(transcriber, tmp_path, max_queue=2)
        try:
            running = submit(jobs)
            wait(jobs, running.id, ("running",))
            first, second = submit(jobs), submit(jobs)
            assert [jobs.position(job.id) for job in (running, first, second)] == [None, 0, 1]
            assert jobs.position("missing") is None
            with pytest.raises(QueueFullError, match="Очередь заполнена: задач в ней 2"):
                submit(jobs)
            assert len(job_folders(jobs)) == 3
            assert jobs.delete(first.id)
            assert jobs.position(second.id) == 0
            transcriber.gate.set()
            assert wait(jobs, second.id).status == "done"
        finally:
            jobs.close()
        assert transcriber.sources == ["call.wav", "call.wav"]

    def test_get_returns_copy(self, jobs):
        job = submit(jobs)
        jobs.get(job.id).status = "failed"
        assert jobs.get(job.id).status != "failed"

    def test_delete_done(self, jobs):
        job = wait(jobs, submit(jobs).id)
        assert jobs.delete(job.id)
        assert jobs.get(job.id) is None
        assert not job.path.parent.exists()
        assert not jobs.delete(job.id)
        assert not jobs.delete("missing")

    def test_delete_running(self, jobs, transcriber):
        transcriber.gate.clear()
        job = submit(jobs)
        wait(jobs, job.id, ("running",))
        assert jobs.delete(job.id)
        assert jobs.get(job.id) is None
        assert job.path.exists()
        assert not jobs.delete(job.id)
        transcriber.gate.set()
        for _ in range(500):
            if not job.path.parent.exists():
                break
            threading.Event().wait(0.01)
        assert not job.path.parent.exists()

    def test_cleanup(self, transcriber, tmp_path):
        clock = Clock()
        jobs = JobQueue(transcriber, tmp_path, keep=60, clock=clock)
        try:
            transcriber.gate.clear()
            done, waiting = submit(jobs), submit(jobs)
            transcriber.gate.set()
            done = wait(jobs, done.id)
            wait(jobs, waiting.id)
            assert jobs.expires(done) == done.finished + 60
            clock.now += 59
            assert jobs.cleanup() == 0
            clock.now += 1
            assert jobs.cleanup() == 2
            assert jobs.get(done.id) is None
            assert job_folders(jobs) == []
        finally:
            jobs.close()

    def test_expires_unfinished(self, jobs, transcriber):
        transcriber.gate.clear()
        assert jobs.expires(submit(jobs)) is None

    def test_copy_failure_removes_files(self, jobs):
        class Broken(io.BytesIO):
            def read(self, size=-1):
                raise OSError("диск заполнен")

        with pytest.raises(OSError, match="диск заполнен"):
            jobs.submit("call.wav", Broken())
        assert job_folders(jobs) == []

    def test_close(self, transcriber, tmp_path):
        jobs = JobQueue(transcriber, tmp_path, workers=2)
        job = wait(jobs, submit(jobs).id)
        jobs.close()
        assert not jobs.directory.exists()
        assert jobs.get(job.id) is None
        for thread in jobs._threads:
            thread.join(5)
            assert not thread.is_alive()

    def test_system_temp_directory(self, transcriber):
        jobs = JobQueue(transcriber)
        try:
            assert jobs.directory.name.startswith("stt-ru-")
        finally:
            jobs.close()

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"workers": 0}, "workers должен быть положительным числом: 0"),
            ({"max_queue": 0}, "max_queue"),
            ({"keep": 0}, "keep"),
            ({"max_duration": 0}, "max_duration"),
            ({"keep": float("inf")}, "keep должен быть положительным числом: inf"),
            ({"max_duration": float("nan")}, "max_duration должен быть положительным числом"),
        ],
    )
    def test_wrong_options(self, transcriber, tmp_path, options, message):
        with pytest.raises(ParameterError, match=message):
            JobQueue(transcriber, tmp_path, **options)
        assert list(tmp_path.iterdir()) == []


class SlowContent(io.BytesIO):
    """Загрузка, которая не кончается, пока не открыт gate"""

    def __init__(self, gate):
        super().__init__(wav_bytes())
        self.gate = gate

    def read(self, size=-1):
        self.gate.wait(5)
        return super().read(size)


def test_errors_logged_separately(transcriber, tmp_path):
    messages, errors = [], []
    jobs = JobQueue(transcriber, tmp_path, log=messages.append, errors=errors.append)
    try:
        failed = wait(jobs, submit(jobs, "broken.wav").id)
    finally:
        jobs.close()
    assert errors == [f"[{failed.id}] broken.wav: ошибка - модель не приняла запись"]
    assert not any("ошибка" in message for message in messages)


def test_place_reserved_while_uploading(transcriber, tmp_path):
    transcriber.gate.clear()
    jobs = JobQueue(transcriber, tmp_path, max_queue=1)
    gate = threading.Event()
    try:
        running = submit(jobs)
        wait(jobs, running.id, ("running",))
        uploading = threading.Thread(target=jobs.submit, args=("slow.wav", SlowContent(gate)))
        uploading.start()
        while not jobs.full():
            pass
        with pytest.raises(QueueFullError):
            submit(jobs)
        gate.set()
        uploading.join(5)
        assert jobs.full()
        assert len(job_folders(jobs)) == 2
    finally:
        gate.set()
        transcriber.gate.set()
        jobs.close()


def test_failed_upload_frees_place(transcriber, tmp_path):
    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise OSError("соединение оборвалось")

    jobs = JobQueue(transcriber, tmp_path, max_queue=1)
    try:
        with pytest.raises(OSError, match="соединение оборвалось"):
            jobs.submit("a.wav", Broken())
        assert not jobs.full()
        assert job_folders(jobs) == []
    finally:
        jobs.close()


@pytest.mark.skipif(sys.platform == "win32", reason="блокировки flock в Windows нет")
class TestStaleDirectories:
    def test_crashed_run_removed(self, transcriber, tmp_path):
        crashed = tmp_path / "stt-ru-crashed"
        (crashed / "job").mkdir(parents=True)
        (crashed / worker.LOCK).write_text("", encoding="utf-8")
        (crashed / "job" / "source.wav").write_bytes(b"audio")
        messages = []
        jobs = JobQueue(transcriber, tmp_path, log=messages.append)
        jobs.close()
        assert not crashed.exists()
        assert messages == ["Удалены файлы прошлых запусков, завершившихся аварийно: 1"]

    @pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root читает все")
    def test_unreadable_foreign_directory(self, transcriber, tmp_path):
        foreign = tmp_path / "stt-ru-other"
        foreign.mkdir()
        foreign.chmod(0)
        try:
            jobs = JobQueue(transcriber, tmp_path)
            jobs.close()
        finally:
            foreign.chmod(0o755)
        assert foreign.is_dir()

    def test_running_and_foreign_kept(self, transcriber, tmp_path):
        foreign = tmp_path / "stt-ru-foreign"
        foreign.mkdir()
        alive = JobQueue(transcriber, tmp_path)
        try:
            second = JobQueue(transcriber, tmp_path)
            second.close()
            assert alive.directory.is_dir() and foreign.is_dir()
        finally:
            alive.close()


def test_without_flock(transcriber, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", None)
    jobs = JobQueue(transcriber, tmp_path)
    try:
        assert wait(jobs, submit(jobs).id).status == "done"
    finally:
        jobs.close()
    assert list(tmp_path.iterdir()) == []
