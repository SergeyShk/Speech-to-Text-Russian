import numpy as np
import pytest
from onnx_asr.utils import ModelNotSupportedError
from onnx_asr.vad import SegmentResult

from stt_ru.asr import DEFAULT_MODEL, DEFAULT_QUANTIZATION, SAMPLE_RATES, Recognizer, recognizer
from stt_ru.exceptions import AudioError, ModelError
from stt_ru.vad import VadOptions


class FakeAdapter:
    def __init__(self):
        self.vad = None
        self.vad_options = None
        self.calls = []

    def with_vad(self, vad, **options):
        self.vad = vad
        self.vad_options = options
        return self

    def recognize(self, waveforms, *, sample_rate):
        self.calls.append((waveforms, sample_rate))
        return [
            iter([SegmentResult(0.5, 1.25, "Алло."), SegmentResult(2.0, 2.5, "")])
            for _ in waveforms
        ]


@pytest.fixture
def fake(monkeypatch):
    adapter = FakeAdapter()
    loads = {}

    def load_model(model, path=None, **kwargs):
        loads["model"] = (model, path, kwargs)
        return adapter

    def load_vad(model="silero", **kwargs):
        loads["vad"] = (model, kwargs)
        return "silero"

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    monkeypatch.setattr(recognizer.onnx_asr, "load_vad", load_vad)
    return adapter, loads


def test_defaults(fake):
    adapter, loads = fake
    asr = Recognizer()
    model, path, kwargs = loads["model"]
    assert (asr.model, model, path) == (DEFAULT_MODEL, "gigaam-v3-e2e-rnnt", None)
    assert kwargs["quantization"] == DEFAULT_QUANTIZATION == "int8"
    assert kwargs["providers"] is None
    assert kwargs["sess_options"].log_severity_level == 3
    assert loads["vad"][0] == "silero"
    assert adapter.vad == "silero"
    assert adapter.vad_options == VadOptions().as_kwargs()


def test_arguments(fake, tmp_path):
    adapter, loads = fake
    vad = VadOptions(min_silence_duration_ms=300)
    Recognizer(
        "t-tech/t-one",
        quantization=None,
        model_dir=tmp_path,
        providers=["CPUExecutionProvider"],
        vad=vad,
    )
    model, path, kwargs = loads["model"]
    assert (model, path, kwargs["quantization"]) == ("t-tech/t-one", tmp_path, None)
    assert isinstance(path, type(tmp_path))
    assert kwargs["providers"] == ["CPUExecutionProvider"]
    assert loads["vad"][1]["providers"] == ["CPUExecutionProvider"]
    assert adapter.vad_options["min_silence_duration_ms"] == 300


def test_recognize(fake):
    adapter, _ = fake
    waveforms = [np.zeros(8000, dtype=np.float32), np.ones(8000, dtype=np.float32)]
    found = Recognizer().recognize(waveforms, 8000)
    assert found == [[(0.5, 1.25, "Алло."), (2.0, 2.5, "")]] * 2
    ((passed, rate),) = adapter.calls
    assert rate == 8000
    assert [w.tolist() for w in passed] == [w.tolist() for w in waveforms]


def test_recognize_nothing(fake):
    adapter, _ = fake
    assert Recognizer().recognize([], 16000) == []
    assert adapter.calls == []


@pytest.mark.parametrize("rate", [12000, 96000])
def test_unsupported_sample_rate(fake, rate):
    with pytest.raises(AudioError, match=f"Частота {rate} Гц не поддерживается"):
        Recognizer().recognize([np.zeros(10, dtype=np.float32)], rate)


def test_sample_rates():
    assert 8000 in SAMPLE_RATES
    assert 16000 in SAMPLE_RATES


def test_unsupported_model(monkeypatch):
    def load_model(model, path=None, **kwargs):
        raise ModelNotSupportedError(model)

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(
        ModelError, match="Не удалось загрузить модель unknown с квантизацией int8"
    ):
        Recognizer("unknown")


def test_download_failure(monkeypatch):
    def load_model(model, path=None, **kwargs):
        raise OSError("сеть недоступна")

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(ModelError, match=r"модель gigaam-v3-e2e-rnnt: сеть недоступна"):
        Recognizer(quantization=None)
