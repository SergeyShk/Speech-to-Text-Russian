import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

import onnx_asr
import onnxruntime as rt
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.utils import ModelFileNotFoundError, SampleRates

from ..audio.loader import Samples
from ..exceptions import AudioError, ModelError
from ..vad import VadOptions

DEFAULT_MODEL = "gigaam-v3-e2e-rnnt"
DEFAULT_QUANTIZATION = "int8"
SAMPLE_RATES: tuple[int, ...] = get_args(SampleRates)
# Silero VAD нужно 36 мс; более короткие записи распознаются как пустые
MIN_DURATION = 0.05
# CoreML на моделях onnx-asr медленнее CPU и держит в памяти в разы больше
EXCLUDED_PROVIDERS = ("CoreMLExecutionProvider", "AzureExecutionProvider")

Found = list[tuple[float, float, str]]


def default_providers() -> list[str]:
    """
    Провайдеры ONNX Runtime по умолчанию

    Вывод:
        list[str]: Доступные провайдеры по приоритету, кроме CoreML и Azure
    """
    return [name for name in rt.get_available_providers() if name not in EXCLUDED_PROVIDERS]


class Recognizer:
    """
    Распознавание речи моделью onnx-asr с поиском речи Silero VAD

    Описание:
        Модель задается именем onnx-asr: gigaam-v3-e2e-rnnt (по умолчанию), gigaam-v3-rnnt,
        t-tech/t-one, alphacep/vosk-model-ru, onnx-community/whisper-large-v3-turbo и другие.
        При первом использовании модель скачивается с Hugging Face в его кэш. Если у модели
        нет весов с заданной квантизацией, загружаются исходные веса с предупреждением

    Аргументы:
        model (str): Имя модели onnx-asr
        quantization (str): Квантизация весов: "int8" или None - исходные веса
        model_dir (str): Каталог с файлами модели; если каталога нет, модель скачивается в него
        providers (list[str]): Провайдеры ONNX Runtime по приоритету,
            например ["CUDAExecutionProvider"]; None - default_providers()
        vad (VadOptions): Параметры поиска речи; None - значения по умолчанию

    Исключения:
        ModelError: Если модель не поддерживается, ее файлы повреждены
            или ее не удалось скачать

    Атрибуты:
        model (str): Имя модели
        quantization (str): Квантизация загруженных весов
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        quantization: str | None = DEFAULT_QUANTIZATION,
        model_dir: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        vad: VadOptions | None = None,
    ) -> None:
        self.model = model
        self.quantization = quantization
        self._model_dir = None if model_dir is None else Path(model_dir)
        self._providers = default_providers() if providers is None else list(providers)
        self._options = rt.SessionOptions()
        # Без этого ONNX Runtime пишет в stderr предупреждения о разбиении графа по провайдерам
        self._options.log_severity_level = 3
        try:
            try:
                asr = self._load(quantization)
            except ModelFileNotFoundError:
                if quantization is None:
                    raise
                warnings.warn(
                    f"У модели {model} нет весов {quantization}, загружаются исходные веса",
                    stacklevel=2,
                )
                self.quantization = None
                asr = self._load(None)
            silero = onnx_asr.load_vad(
                "silero", sess_options=self._options, providers=self._providers
            )
        except Exception as error:
            variant = f" с квантизацией {self.quantization}" if self.quantization else ""
            raise ModelError(f"Не удалось загрузить модель {model}{variant}: {error}") from error
        self._asr = asr.with_vad(silero, **(vad or VadOptions()).as_kwargs())

    def _load(self, quantization: str | None) -> TextResultsAsrAdapter:
        return onnx_asr.load_model(
            self.model,
            self._model_dir,
            quantization=quantization,
            sess_options=self._options,
            providers=self._providers,
        )

    def recognize(self, waveforms: Sequence[Samples], sample_rate: int) -> list[Found]:
        """
        Распознавание одноканальных записей одной частоты

        Аргументы:
            waveforms (list[numpy.ndarray]): Одномерные массивы отсчетов float32
            sample_rate (int): Частота дискретизации, Гц, одна из SAMPLE_RATES

        Вывод:
            list[list[tuple[float, float, str]]]: Для каждой записи фрагменты речи:
                начало и конец в секундах и текст; записи короче MIN_DURATION - без фрагментов

        Исключения:
            AudioError: Если модель не принимает такую частоту дискретизации
        """
        if sample_rate not in SAMPLE_RATES:
            raise AudioError(
                f"Частота {sample_rate} Гц не поддерживается, допустимые: "
                + ", ".join(map(str, SAMPLE_RATES))
            )
        found: list[Found] = [[] for _ in waveforms]
        audible = [
            index
            for index, waveform in enumerate(waveforms)
            if len(waveform) >= MIN_DURATION * sample_rate
        ]
        if audible:
            results = self._asr.recognize(
                [waveforms[index] for index in audible], sample_rate=cast(SampleRates, sample_rate)
            )
            for index, result in zip(audible, results, strict=True):
                found[index] = [(segment.start, segment.end, segment.text) for segment in result]
        return found
