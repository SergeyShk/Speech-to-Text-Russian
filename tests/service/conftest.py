import io
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from stt_ru import Segment, Transcript
from stt_ru.audio import load_audio
from stt_ru.exceptions import AudioError
from stt_ru.service import JobQueue
from stt_ru.service.worker import LOCK


class FakeTranscriber:
    """Читает файл по-настоящему, распознает подменой; gate задерживает распознавание"""

    def __init__(self):
        self.recognizer = SimpleNamespace(model="fake")
        self.gate = threading.Event()
        self.gate.set()
        self.sources = []

    def load(self, path, *, max_duration=None):
        return load_audio(path, max_duration=max_duration)

    def transcribe_audio(self, audio, source=None):
        self.sources.append(source)
        self.gate.wait(10)
        if "silent" in source:
            raise RuntimeError
        if "broken" in source:
            raise AudioError("модель не приняла запись")
        segments = (Segment(0.1, 0.4, "Алло.", 0), Segment(0.5, 0.9, "Да, слушаю.", 1))
        return Transcript(segments, audio.duration, audio.num_channels, "fake", source)


def wav_bytes(seconds=1.0, rate=8000):
    samples = np.zeros((int(rate * seconds), 2), dtype=np.float32)
    samples[: rate // 2, 0] = 0.5
    samples[rate // 2 :, 1] = -0.25
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def job_folders(jobs):
    """Каталоги задач без служебного файла блокировки"""
    return [path for path in jobs.directory.iterdir() if path.name != LOCK]


def wait(jobs, job_id, statuses=("done", "failed"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = jobs.get(job_id)
        if job is None or job.status in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"задача {job_id} не дошла до {statuses}")


@pytest.fixture
def transcriber():
    fake = FakeTranscriber()
    yield fake
    fake.gate.set()


@pytest.fixture
def jobs(transcriber, tmp_path):
    queue = JobQueue(transcriber, tmp_path)
    yield queue
    queue.close()
