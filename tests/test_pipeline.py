from pathlib import Path
from typing import ClassVar

import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

import stt_ru
from stt_ru import Segment, Transcriber, pipeline
from stt_ru.audio import Audio
from stt_ru.pipeline import cut_points
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


def stereo(rate=8000, seconds=1.0):
    samples = np.zeros((int(rate * seconds), 2), dtype=np.float32)
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


def test_identical_channels(fake):
    samples = np.repeat(stereo()[:, :1], 2, axis=1)
    transcript = Transcriber().transcribe_audio(Audio(samples, 8000))
    ((waveforms, _),) = fake[0].calls
    assert len(waveforms) == 1
    assert transcript.channels == 1
    assert {segment.channel for segment in transcript.segments} == {None}


def test_nearly_identical_channels(fake):
    rng = np.random.default_rng(0)
    left = rng.normal(0, 0.1, 8000).astype(np.float32)
    right = left + rng.normal(0, 0.001, 8000).astype(np.float32)
    transcript = Transcriber().transcribe_audio(Audio(np.stack([left, right], 1), 8000))
    assert transcript.channels == 1


def test_different_channels_kept(fake):
    rng = np.random.default_rng(0)
    left = rng.normal(0, 0.1, 8000).astype(np.float32)
    right = left + rng.normal(0, 0.01, 8000).astype(np.float32)
    transcript = Transcriber().transcribe_audio(Audio(np.stack([left, right], 1), 8000))
    assert transcript.channels == 2


def test_tracks_are_views(fake):
    samples = stereo()
    transcriber = Transcriber()
    tracks = transcriber._tracks(Audio(samples, 8000))
    assert all(np.shares_memory(track, samples) for _, track in tracks)


def test_duplicate_channel_among_others(fake):
    samples = np.concatenate([stereo(), stereo()[:, :1]], axis=1)
    transcript = Transcriber().transcribe_audio(Audio(samples, 8000))
    ((waveforms, _),) = fake[0].calls
    assert [float(w[0]) for w in waveforms] == [0.25, 0.75]
    assert transcript.channels == 2
    assert {segment.channel for segment in transcript.segments} == {0, 1}


def test_empty_audio(fake):
    transcript = Transcriber().transcribe_audio(Audio(np.zeros((0, 2), dtype=np.float32), 8000))
    assert fake[0].calls == []
    assert transcript.segments == ()


def test_windows(fake, monkeypatch):
    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 1.0)
    monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 0.25)
    rate = 8000
    samples = stereo(rate=rate, seconds=2.2)
    samples[int(1.0 * rate) : int(1.4 * rate)] = 0
    transcript = Transcriber().transcribe_audio(Audio(samples, rate))
    lengths = [[len(w) for w in waveforms] for waveforms, _ in fake[0].calls]
    cut = lengths[0][0]
    assert int(1.0 * rate) <= cut < int(1.4 * rate)
    assert lengths == [[cut, cut], [len(samples) - cut, len(samples) - cut]]
    first = [s for s in transcript.segments if s.text == "Первый."]
    offset = cut / rate
    assert [(s.start, s.end) for s in first] == [
        (0.2, 0.8),
        (round(offset + 0.2, 3), round(offset + 0.8, 3)),
    ]
    assert transcript.duration == 2.2


class TestCutPoints:
    @pytest.fixture(autouse=True)
    def short_windows(self, monkeypatch):
        monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 1.0)
        monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 0.5)

    def test_short(self):
        assert cut_points([np.ones(12000, dtype=np.float32)], 8000) == [0, 12000]

    def test_pauses(self):
        track = np.ones(8000 * 4, dtype=np.float32)
        track[7000:10200] = 0
        track[15000:18200] = 0
        cuts = cut_points([track], 8000)
        assert cuts[0] == 0 and cuts[-1] == len(track)
        assert 7000 <= cuts[1] < 10200
        assert 15000 <= cuts[2] < 18200

    def test_pause_beats_short_gap_inside_phrase(self):
        track = np.ones(8000 * 2, dtype=np.float32)
        track[5000:5320] = 0
        track[9000:11400] = 0.05
        assert 9000 <= cut_points([track], 8000)[1] < 11400

    def test_noisy_channel_does_not_outweigh(self):
        speech = np.ones(8000 * 2, dtype=np.float32)
        speech[9000:11400] = 0
        noise = np.full(8000 * 2, 3, dtype=np.float32)
        noise[5000:5320] = 0
        assert 9000 <= cut_points([speech, noise], 8000)[1] < 11400

    def test_digital_silence_channel_does_not_outweigh(self):
        speech = np.ones(8000 * 2, dtype=np.float32)
        speech[9000:11400] = 0
        clicks = np.zeros(8000 * 2, dtype=np.float32)
        for start in range(0, len(clicks), 800):
            if not 5000 <= start < 7400:
                clicks[start : start + 160] = 0.001
        assert 9000 <= cut_points([speech, clicks], 8000)[1] < 11400

    def test_last_window_absorbs_tail(self):
        assert cut_points([np.ones(8000 + 4000, dtype=np.float32)], 8000) == [0, 12000]
        assert len(cut_points([np.ones(8000 + 4001, dtype=np.float32)], 8000)) == 3


def test_transcribe_file(fake, tmp_path):
    path = tmp_path / "звонок.wav"
    sf.write(path, stereo(), 8000, subtype="PCM_16")
    transcript = Transcriber().transcribe(path)
    assert transcript.source == "звонок.wav"
    assert fake[0].calls[0][1] == 8000


def test_resample_unsupported_rate_once(fake, tmp_path, monkeypatch):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=12000), 12000, subtype="PCM_16")
    loads = []
    load = pipeline.load_audio

    def spy(*args, **kwargs):
        loads.append(kwargs["sample_rate"])
        return load(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_audio", spy)
    transcript = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()).transcribe(path)
    assert loads == [16000]
    assert fake[0].calls[0][1] == 16000
    assert transcript.duration == pytest.approx(1.0, abs=0.01)


def test_resample_after_ffmpeg_decode(fake, tmp_path, monkeypatch):
    path = tmp_path / "call.wav"
    sf.write(path, stereo(rate=12000), 12000, subtype="PCM_16")
    monkeypatch.setattr(
        pipeline.sf, "info", lambda path: (_ for _ in ()).throw(sf.LibsndfileError(1))
    )
    transcript = Transcriber(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()).transcribe(path)
    assert fake[0].calls[0][1] == 16000
    assert transcript.duration == pytest.approx(1.0, abs=0.01)


def test_missing_file(fake, tmp_path):
    with pytest.raises(FileNotFoundError):
        Transcriber().transcribe(tmp_path / "missing.wav")


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


@pytest.mark.network
def test_real_model_windows(monkeypatch):
    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", 9.0)
    monkeypatch.setattr(pipeline, "CUT_SEARCH_SECONDS", 1.0)
    transcript = Transcriber().transcribe(CALL)
    operator = " ".join(s.text for s in transcript.segments if s.channel == 1).lower()
    assert "благодарю вас за ожидание" in operator
    assert "до свидания" in operator
