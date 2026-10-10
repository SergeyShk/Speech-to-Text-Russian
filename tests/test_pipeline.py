from pathlib import Path
from typing import ClassVar

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

import stt_ru
from stt_ru import Segment, Transcriber, pipeline
from stt_ru.audio import Audio
from stt_ru.vad import VadOptions

CALL = Path(__file__).parent / "data" / "call.wav"


class FakeRecognizer:
    instances: ClassVar[list["FakeRecognizer"]] = []

    def __init__(self, model="fake", **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.calls = []
        FakeRecognizer.instances.append(self)

    def recognize(self, waveforms, sample_rate):
        self.calls.append(([w.copy() for w in waveforms], sample_rate))
        found = [[(1.0, 2.0, "Второй."), (0.2, 0.8, "Первый."), (3.0, 3.5, "")]]
        found += [[(0.5, 1.5, "Ответ.")] for _ in waveforms[1:]]
        return found


@pytest.fixture
def fake(monkeypatch):
    FakeRecognizer.instances.clear()
    monkeypatch.setattr(pipeline, "Recognizer", FakeRecognizer)
    return FakeRecognizer.instances


def stereo(rate=8000):
    samples = np.zeros((rate, 2), dtype=np.float32)
    samples[:, 0] = 0.25
    samples[:, 1] = 0.75
    return samples


def test_arguments(fake, tmp_path):
    vad = VadOptions(threshold=0.4)
    Transcriber("t-tech/t-one", quantization=None, model_dir=tmp_path, providers=["CPU"], vad=vad)
    (recognizer,) = fake
    assert recognizer.model == "t-tech/t-one"
    assert recognizer.kwargs == {
        "quantization": None,
        "model_dir": tmp_path,
        "providers": ["CPU"],
        "vad": vad,
    }


def test_channels(fake):
    transcript = Transcriber().transcribe_audio(Audio(stereo(), 8000), source="call.wav")
    ((waveforms, rate),) = fake[0].calls
    assert rate == 8000
    assert [float(w[0]) for w in waveforms] == [0.25, 0.75]
    assert transcript.segments == (
        Segment(0.2, 0.8, "Первый.", 0),
        Segment(0.5, 1.5, "Ответ.", 1),
        Segment(1.0, 2.0, "Второй.", 0),
    )
    assert (transcript.channels, transcript.duration) == (2, 1.0)
    assert (transcript.model, transcript.source) == ("gigaam-v3-e2e-rnnt", "call.wav")


def test_same_start_ordered_by_channel(monkeypatch, fake):
    transcriber = Transcriber()
    monkeypatch.setattr(
        transcriber.recognizer, "recognize", lambda w, r: [[(1.0, 2.0, "Б")], [(1.0, 1.5, "А")]]
    )
    segments = transcriber.transcribe_audio(Audio(stereo(), 8000)).segments
    assert [s.channel for s in segments] == [0, 1]


def test_mixed_channels(fake):
    transcript = Transcriber(split_channels=False).transcribe_audio(Audio(stereo(), 8000))
    ((waveforms, _),) = fake[0].calls
    assert len(waveforms) == 1
    assert float(waveforms[0][0]) == 0.5
    assert {segment.channel for segment in transcript.segments} == {None}
    assert transcript.channels == 1


def test_mono(fake):
    transcript = Transcriber().transcribe_audio(Audio(stereo()[:, :1], 8000))
    assert transcript.channels == 1
    assert transcript.segments[0] == Segment(0.2, 0.8, "Первый.")


def test_empty_audio(fake):
    transcript = Transcriber().transcribe_audio(Audio(np.zeros((0, 2), dtype=np.float32), 8000))
    assert fake[0].calls == []
    assert (transcript.segments, transcript.channels) == ((), 2)


def test_transcribe_file(fake, tmp_path):
    path = tmp_path / "звонок.wav"
    sf.write(path, stereo(), 8000, subtype="PCM_16")
    transcript = Transcriber().transcribe(path)
    assert transcript.source == "звонок.wav"
    assert fake[0].calls[0][1] == 8000


def test_resample_unsupported_rate(fake, tmp_path):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=12000), 12000, subtype="PCM_16")
    transcript = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()).transcribe(path)
    assert fake[0].calls[0][1] == 16000
    assert transcript.duration == pytest.approx(1.0, abs=0.01)


def test_package_exports():
    assert stt_ru.Transcriber is Transcriber


@pytest.mark.network
def test_real_model():
    transcript = Transcriber().transcribe(CALL)
    assert transcript.channels == 2
    assert transcript.model == "gigaam-v3-e2e-rnnt"
    operator = " ".join(s.text for s in transcript.segments if s.channel == 1).lower()
    client = " ".join(s.text for s in transcript.segments if s.channel == 0).lower()
    assert "благодарю вас за ожидание" in operator
    assert "до свидания" in operator
    assert "," in operator
    assert "до свидания" in client
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert all(0 <= s.start < s.end <= transcript.duration for s in transcript.segments)
