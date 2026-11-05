from itertools import groupby

import numpy as np
import pytest

from stt_ru.diarization import Diarizer, diarizer
from stt_ru.exceptions import AudioError, ModelError, ParameterError

RATE = 16000


class FakeEmbedder:
    """Голос определяется громкостью: 0.5 и 0.25 - разные люди"""

    def __init__(self):
        self.lengths = []

    def embedding(self, chunk):
        self.lengths.append(len(chunk))
        return np.array([2.0, 0.0] if np.abs(chunk).mean() > 0.4 else [0.0, 3.0], dtype=np.float32)

    def resampler(self, waveforms, lengths, rate):
        return np.repeat(waveforms, RATE // rate, axis=1), lengths * (RATE // rate)


class FakeSession:
    """Вероятности наборов говорящих по громкости кадра; набор 4 - перекрытие"""

    def __init__(self, overlap=None):
        self.overlap = overlap

    def run(self, names, inputs):
        window = inputs["input_values"][0, 0]
        assert len(window) == diarizer.WINDOW
        frames = (len(window) - diarizer.FRAME_FIELD) // diarizer.FRAME_STEP + 1
        logits = np.full((1, frames, 7), -10.0, dtype=np.float32)
        for frame in range(frames):
            begin = frame * diarizer.FRAME_STEP
            level = np.abs(window[begin : begin + diarizer.FRAME_STEP]).mean()
            kind = 1 if level > 0.4 else 2 if level > 0.1 else 0
            if self.overlap is not None and level == pytest.approx(self.overlap, abs=0.01):
                kind = 4
            logits[0, frame, kind] = 0.0
        return [logits]


@pytest.fixture
def embedder(monkeypatch):
    fake = FakeEmbedder()

    class FakeManager:
        def __init__(self, options, providers):
            self.providers = providers

        def create_se(self, model):
            assert model == diarizer.EMBEDDING_MODEL
            return fake

    class FakeResolver:
        def resolve_model(self):
            return {"model": "segmentation.onnx"}

    monkeypatch.setattr(diarizer, "Manager", FakeManager)
    monkeypatch.setattr(diarizer, "create_vad_resolver", lambda model: FakeResolver())
    monkeypatch.setattr(diarizer.rt, "InferenceSession", lambda *args, **kwargs: FakeSession())
    return fake


def signal(*parts, rate=RATE):
    return np.concatenate(
        [np.full(int(seconds * rate), level, dtype=np.float32) for level, seconds in parts]
    )


def test_two_speakers(embedder):
    turns = Diarizer().diarize(signal((0, 1), (0.5, 8), (0, 3), (0.25, 8), (0, 5)), RATE)
    assert [turn.speaker for turn in turns] == [1, 2]
    first, second = turns
    assert first.start == pytest.approx(1.0, abs=0.1) and first.end == pytest.approx(9.0, abs=0.1)
    assert second.start == pytest.approx(12.0, abs=0.1)
    assert second.end == pytest.approx(20.0, abs=0.1)


def test_numbered_by_appearance(embedder):
    turns = Diarizer().diarize(signal((0.25, 6), (0, 3), (0.5, 6), (0, 3), (0.25, 6)), RATE)
    speakers = [turn.speaker for turn in turns]
    assert [speaker for speaker, _ in groupby(speakers)] == [1, 2, 1]


def test_exact_speakers(embedder):
    found = Diarizer().diarize(signal((0.5, 6), (0, 3), (0.25, 6)), RATE, speakers=1)
    assert {turn.speaker for turn in found} == {1}


@pytest.mark.parametrize("speakers", [0, -1])
def test_wrong_speakers(embedder, speakers):
    with pytest.raises(ParameterError, match="Число говорящих"):
        Diarizer().diarize(signal((0.5, 1)), RATE, speakers=speakers)


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_wrong_threshold(embedder, threshold):
    with pytest.raises(ParameterError, match="threshold"):
        Diarizer(threshold=threshold)


def test_threshold_used(embedder):
    found = Diarizer(threshold=5.0).diarize(signal((0.5, 6), (0, 3), (0.25, 6)), RATE)
    assert {turn.speaker for turn in found} == {1}


def test_short_recording(embedder):
    (turn,) = Diarizer().diarize(signal((0.5, 2)), RATE)
    assert (turn.start, turn.speaker) == (pytest.approx(0.02, abs=0.03), 1)
    assert turn.end == pytest.approx(2.0, abs=0.05)


def test_last_window_reaches_end(embedder):
    turns = Diarizer().diarize(signal((0, 11.3), (0.5, 1.5)), RATE)
    assert turns[-1].end == pytest.approx(12.8, abs=0.1)


def test_ends_within_recording(embedder):
    assert Diarizer().diarize(signal((0.5, 13)), RATE)[-1].end <= 13.0


@pytest.mark.parametrize("parts", [(), ((0, 3),), ((0, 12),)])
def test_no_speech(embedder, parts):
    samples = signal(*parts) if parts else np.zeros(0, dtype=np.float32)
    assert Diarizer().diarize(samples, RATE) == []


def test_too_short_speech_skipped(embedder):
    assert Diarizer().diarize(signal((0, 2), (0.5, 0.05), (0, 2)), RATE) == []


def test_short_chunk_stretched(embedder):
    Diarizer().diarize(signal((0, 2), (0.5, 0.3), (0, 2)), RATE)
    assert set(embedder.lengths) == {diarizer.MIN_CHUNK}


def test_each_chunk_embedded_alone(embedder):
    Diarizer().diarize(signal((0.5, 3), (0, 1), (0.25, 2)), RATE)
    assert sorted(embedder.lengths)[0] != sorted(embedder.lengths)[-1]


def test_overlap(embedder, monkeypatch):
    session = FakeSession(overlap=0.75)
    monkeypatch.setattr(diarizer.rt, "InferenceSession", lambda *args, **kwargs: session)
    turns = Diarizer().diarize(signal((0.5, 4), (0.75, 2), (0.25, 4)), RATE)
    assert len(turns) >= 2
    assert {turn.speaker for turn in turns} == {1, 2}
    spans = sorted((turn.start, turn.end) for turn in turns)
    assert spans[1][0] < spans[0][1]


def test_only_overlap_uses_all_voiced(embedder, monkeypatch):
    session = FakeSession(overlap=0.75)
    monkeypatch.setattr(diarizer.rt, "InferenceSession", lambda *args, **kwargs: session)
    assert Diarizer().diarize(signal((0.75, 3)), RATE)


def test_resampled(embedder):
    samples = signal((0.5, 2), rate=8000)
    (turn,) = Diarizer().diarize(samples, 8000)
    assert turn.end == pytest.approx(2.0, abs=0.05)


def test_resampled_by_blocks(embedder, monkeypatch):
    monkeypatch.setattr(diarizer, "RESAMPLE_BLOCK", 1)
    (turn,) = Diarizer().diarize(signal((0.5, 2.5), rate=8000), 8000)
    assert turn.end == pytest.approx(2.5, abs=0.05)


def test_empty_resampled(embedder):
    assert Diarizer().diarize(np.zeros(0, dtype=np.float32), 8000) == []


def test_unknown_rate(embedder):
    with pytest.raises(AudioError, match="Частота 7000 Гц"):
        Diarizer().diarize(signal((0.5, 1)), 7000)


def test_nan_replaced(embedder):
    samples = signal((0.5, 2))
    samples[10] = np.nan
    samples[20] = np.inf
    assert Diarizer().diarize(samples, RATE)
    assert np.isnan(samples[10])


def test_speech_only_in_padding(embedder, monkeypatch):
    class Hallucinating(FakeSession):
        def run(self, names, inputs):
            logits = super().run(names, inputs)[0]
            logits[0, :, :] = -10.0
            logits[0, :, 0] = 0.0
            logits[0, 400:, 0] = -10.0
            logits[0, 400:, 1] = 0.0
            return [logits]

    monkeypatch.setattr(diarizer.rt, "InferenceSession", lambda *args, **kwargs: Hallucinating())
    assert Diarizer().diarize(signal((0.5, 2)), RATE) == []
    assert embedder.lengths == []


def test_providers_passed(monkeypatch, embedder):
    seen = {}

    def session(*args, **kwargs):
        seen.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(diarizer.rt, "InferenceSession", session)
    Diarizer(providers=["CPUExecutionProvider"])
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_load_failure(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("нет сети")

    monkeypatch.setattr(diarizer, "Manager", broken)
    with pytest.raises(ModelError, match="Не удалось загрузить модели диаризации: нет сети"):
        Diarizer()
