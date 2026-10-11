import numpy as np
import pytest
from onnx_asr.utils import ModelFileNotFoundError, ModelNotSupportedError
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
            iter([SegmentResult(0.5, 1.25, f"Алло {len(w)}."), SegmentResult(2.0, 2.5, "")])
            for w in waveforms
        ]


@pytest.fixture
def fake(monkeypatch):
    adapter = FakeAdapter()
    loads = {"model": []}

    def load_model(model, path=None, **kwargs):
        loads["model"].append((model, path, kwargs))
        return adapter

    def load_vad(model="silero", **kwargs):
        loads["vad"] = (model, kwargs)
        return "silero"

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    monkeypatch.setattr(recognizer.onnx_asr, "load_vad", load_vad)
    monkeypatch.setattr(
        recognizer.rt,
        "get_available_providers",
        lambda: [
            "TensorrtExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
            "CoreMLExecutionProvider",
            "AzureExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    return adapter, loads


def test_defaults(fake):
    adapter, loads = fake
    asr = Recognizer()
    ((model, path, kwargs),) = loads["model"]
    assert (asr.model, model, path) == (DEFAULT_MODEL, "gigaam-v3-e2e-rnnt", None)
    assert asr.quantization == kwargs["quantization"] == DEFAULT_QUANTIZATION == "int8"
    assert kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert kwargs["sess_options"].log_severity_level == 3
    assert loads["vad"][0] == "silero"
    assert loads["vad"][1]["providers"] == kwargs["providers"]
    assert adapter.vad == "silero"
    assert adapter.vad_options == VadOptions().as_kwargs()


def test_default_providers(fake):
    assert recognizer.default_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_load_output_goes_to_stderr(fake, monkeypatch, capsys):
    adapter, _ = fake

    def load_model(model, path=None, **kwargs):
        print("EP Error: libnvinfer.so.10: cannot open shared object file")
        return adapter

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    Recognizer()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "EP Error" in captured.err


def test_arguments(fake, tmp_path):
    adapter, loads = fake
    vad = VadOptions(min_silence_duration_ms=300)
    Recognizer(
        "t-tech/t-one",
        quantization=None,
        model_dir=str(tmp_path),
        providers=["CoreMLExecutionProvider"],
        vad=vad,
    )
    ((model, path, kwargs),) = loads["model"]
    assert (model, path, kwargs["quantization"]) == ("t-tech/t-one", tmp_path, None)
    assert kwargs["providers"] == ["CoreMLExecutionProvider"]
    assert loads["vad"][1]["providers"] == ["CoreMLExecutionProvider"]
    assert adapter.vad_options["min_silence_duration_ms"] == 300


def test_quantization_fallback(fake, monkeypatch):
    _, loads = fake
    load = recognizer.onnx_asr.load_model

    def load_model(model, path=None, **kwargs):
        if kwargs["quantization"] == "int8":
            raise ModelFileNotFoundError("model?int8.onnx", "/cache")
        return load(model, path, **kwargs)

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.warns(UserWarning, match="У модели org/model нет весов int8"):
        asr = Recognizer("org/model")
    assert asr.quantization is None
    assert [kwargs["quantization"] for _, _, kwargs in loads["model"]] == [None]


def test_model_without_int8_not_searched(fake):
    _, loads = fake
    with pytest.warns(UserWarning, match="У модели t-tech/t-one нет весов int8"):
        asr = Recognizer("t-tech/t-one")
    assert asr.quantization is None
    assert [kwargs["quantization"] for _, _, kwargs in loads["model"]] == [None]


def test_explicit_quantization_without_fallback(fake, monkeypatch):
    def load_model(model, path=None, **kwargs):
        raise ModelFileNotFoundError("model?fp16.onnx", "/cache")

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(ModelError, match="модель m с квантизацией fp16"):
        Recognizer("m", quantization="fp16")


def test_model_dir_without_fallback(fake, monkeypatch, tmp_path, recwarn):
    def load_model(model, path=None, **kwargs):
        raise ModelFileNotFoundError("model?int8.onnx", path)

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(ModelError, match=r"с квантизацией int8: File 'model\?int8.onnx'"):
        Recognizer("m", model_dir=tmp_path)
    assert not recwarn.list


def test_no_fallback_without_quantization(monkeypatch):
    def load_model(model, path=None, **kwargs):
        raise ModelFileNotFoundError("model.onnx", "/models")

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(ModelError, match=r"модель m: File 'model.onnx' not found"):
        Recognizer("m", quantization=None)


def test_recognize(fake):
    adapter, _ = fake
    waveforms = [np.zeros(8000, dtype=np.float32), np.ones(4000, dtype=np.float32)]
    found = Recognizer().recognize(waveforms, 8000)
    assert found == [
        [(0.5, 1.25, "Алло 8000."), (2.0, 2.5, "")],
        [(0.5, 1.25, "Алло 4000."), (2.0, 2.5, "")],
    ]
    ((passed, rate),) = adapter.calls
    assert rate == 8000
    assert [w.tolist() for w in passed] == [w.tolist() for w in waveforms]


def test_recognize_replaces_non_finite_samples(fake):
    adapter, _ = fake
    waveform = np.zeros(8000, dtype=np.float32)
    waveform[1:4] = [np.nan, np.inf, -np.inf]
    Recognizer().recognize([waveform], 8000)
    ((passed, _),) = adapter.calls
    assert passed[0][:5].tolist() == [0.0, 0.0, 1.0, -1.0, 0.0]


def test_recognize_nothing(fake):
    adapter, _ = fake
    assert Recognizer().recognize([], 16000) == []
    assert adapter.calls == []


def test_short_waveforms(fake):
    adapter, _ = fake
    short = np.zeros(399, dtype=np.float32)
    long = np.zeros(400, dtype=np.float32)
    found = Recognizer().recognize([short, long, short], 8000)
    assert found == [[], [(0.5, 1.25, "Алло 400."), (2.0, 2.5, "")], []]
    ((passed, _),) = adapter.calls
    assert [len(w) for w in passed] == [400]


def test_only_short_waveforms(fake):
    adapter, _ = fake
    assert Recognizer().recognize([np.zeros(1, dtype=np.float32)], 16000) == [[]]
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


class BrokenModelFileError(Exception):
    pass


@pytest.mark.parametrize(
    "error",
    [OSError("сеть недоступна"), ValueError("bad repo id"), BrokenModelFileError("protobuf")],
)
def test_load_errors(monkeypatch, error):
    def load_model(model, path=None, **kwargs):
        raise error

    monkeypatch.setattr(recognizer.onnx_asr, "load_model", load_model)
    with pytest.raises(ModelError, match=f"модель gigaam-v3-e2e-rnnt: {error}") as raised:
        Recognizer(quantization=None)
    assert raised.value.__cause__ is error


def test_vad_load_error(fake, monkeypatch):
    def load_vad(model="silero", **kwargs):
        raise OSError("нет сети")

    monkeypatch.setattr(recognizer.onnx_asr, "load_vad", load_vad)
    with pytest.raises(ModelError, match="нет сети"):
        Recognizer()
