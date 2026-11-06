import contextlib
import os
import sys
import threading
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast, get_args

import numpy as np
import onnx_asr
import onnxruntime as rt
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.utils import ModelFileNotFoundError, SampleRates
from onnx_asr.vad import TimestampedSegmentResult, Vad

from ..audio.loader import Samples
from ..exceptions import AudioError, ModelError
from ..vad import VadOptions

DEFAULT_MODEL = "gigaam-v3-e2e-rnnt"
DEFAULT_QUANTIZATION = "int8"
SAMPLE_RATES: tuple[int, ...] = get_args(SampleRates)
# Silero VAD нужно 36 мс; более короткие записи распознаются как пустые
MIN_DURATION = 0.05
# CoreML на моделях onnx-asr медленнее CPU и держит в памяти в разы больше; TensorRT из
# onnxruntime-gpu требует отдельно установленный TensorRT и долго собирает движки
EXCLUDED_PROVIDERS = (
    "CoreMLExecutionProvider",
    "AzureExecutionProvider",
    "TensorrtExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
)
# У этих моделей нет весов int8: попытка их найти ходила бы в сеть при каждой загрузке
WITHOUT_INT8 = frozenset({"t-tech/t-one"})

Found = list[tuple[float, float, str]]
Word = tuple[float, float, str]


_STDOUT_LOCK = threading.Lock()


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """
    Вывод stdout в stderr на время загрузки моделей

    Описание:
        redirect_stdout меняет sys.stdout на весь процесс, поэтому одновременные загрузки ждут
        друг друга: иначе последняя оставила бы его подмененным. Вывод других потоков на это
        время тоже уходит в stderr
    """
    with _STDOUT_LOCK, contextlib.redirect_stdout(sys.stderr):
        yield


def default_providers() -> list[str]:
    """
    Провайдеры ONNX Runtime по умолчанию

    Вывод:
        list[str]: Доступные провайдеры по приоритету, кроме CoreML, Azure и TensorRT
    """
    return [name for name in rt.get_available_providers() if name not in EXCLUDED_PROVIDERS]


class Recognizer:
    """
    Распознавание речи моделью onnx-asr с поиском речи Silero VAD

    Описание:
        Модель задается именем onnx-asr: gigaam-v3-e2e-rnnt (по умолчанию), gigaam-v3-rnnt,
        t-tech/t-one, alphacep/vosk-model-ru, onnx-community/whisper-large-v3-turbo и другие.
        При первом использовании модель скачивается с Hugging Face в его кэш, дальше
        загружается из кэша. Если у модели нет весов int8, загружаются исходные веса
        с предупреждением

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
        # ONNX Runtime печатает в stdout ошибку провайдера, если переходит на запасной,
        # а stdout - для результата распознавания
        with stdout_to_stderr():
            asr, silero = self._load_all(quantization)
        self._asr = asr.with_vad(silero, **(vad or VadOptions()).as_kwargs())

    def _load_all(self, quantization: str | None) -> tuple[TextResultsAsrAdapter, Vad]:
        try:
            asr = self._load_weights(quantization)
            silero = onnx_asr.load_vad(
                "silero", sess_options=self._options, providers=self._providers
            )
        except Exception as error:
            variant = f" с квантизацией {self.quantization}" if self.quantization else ""
            raise ModelError(
                f"Не удалось загрузить модель {self.model}{variant}: {error}"
            ) from error
        return asr, silero

    def _load_weights(self, quantization: str | None) -> TextResultsAsrAdapter:
        if quantization != DEFAULT_QUANTIZATION:
            return self._load(quantization)
        if self.model in WITHOUT_INT8:
            return self._fall_back(quantization)
        try:
            return self._load(quantization)
        except ModelFileNotFoundError:
            if self._model_dir is not None and self._model_dir.is_dir():
                raise
            return self._fall_back(quantization)

    def _fall_back(self, quantization: str) -> TextResultsAsrAdapter:
        warnings.warn(
            f"У модели {self.model} нет весов {quantization}, загружаются исходные веса",
            stacklevel=5,
        )
        self.quantization = None
        return self._load(None)

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
        return [
            [(segment.start, segment.end, segment.text) for segment in result]
            for result in self._run(self._asr, waveforms, sample_rate)
        ]

    def recognize_words(
        self, waveforms: Sequence[Samples], sample_rate: int
    ) -> list[list[tuple[float, float, str, list[Word] | None]]]:
        """
        Распознавание со временем слов

        Описание:
            Как recognize, но у каждого фрагмента есть еще его слова. Слова не отличаются от текста
            фрагмента, кроме лишних пробелов. Конец слова - начало следующего, у последнего -
            конец фрагмента

        Аргументы:
            waveforms (list[numpy.ndarray]): Одномерные массивы отсчетов float32
            sample_rate (int): Частота дискретизации, Гц, одна из SAMPLE_RATES

        Вывод:
            list[list[tuple]]: Для каждой записи фрагменты речи: начало и конец в секундах, текст
                и список слов (начало, конец, слово); вместо списка None, если модель
                не сообщает время слов

        Исключения:
            AudioError: Если модель не принимает такую частоту дискретизации
        """
        return [
            [(segment.start, segment.end, segment.text, _words(segment)) for segment in result]
            for result in self._run(self._asr.with_timestamps(), waveforms, sample_rate)
        ]

    def _run(
        self, adapter: Any, waveforms: Sequence[Samples], sample_rate: int
    ) -> list[list[Any]]:
        if sample_rate not in SAMPLE_RATES:
            raise AudioError(
                f"Частота {sample_rate} Гц не поддерживается, допустимые: "
                + ", ".join(map(str, SAMPLE_RATES))
            )
        found: list[list[Any]] = [[] for _ in waveforms]
        # Один NaN или inf превращает в <unk> текст всей пачки фрагментов
        waveforms = [
            waveform
            if np.isfinite(waveform).all()
            else np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
            for waveform in waveforms
        ]
        audible = [
            index
            for index, waveform in enumerate(waveforms)
            if len(waveform) >= MIN_DURATION * sample_rate
        ]
        if audible:
            results = adapter.recognize(
                [waveforms[index] for index in audible], sample_rate=cast(SampleRates, sample_rate)
            )
            for index, result in zip(audible, results, strict=True):
                found[index] = list(result)
        return found


def _words(segment: TimestampedSegmentResult) -> list[Word] | None:
    if not segment.tokens or not segment.timestamps:
        return None
    letters = [
        (letter, moment)
        for token, moment in zip(segment.tokens, segment.timestamps, strict=True)
        for letter in token
        if not letter.isspace()
    ]
    # onnx-asr убирает пробел перед знаками препинания: границы слов берутся из текста
    words = segment.text.split()
    if not words or "".join(letter for letter, _ in letters) != "".join(words):
        return None
    moments = []
    position = 0
    for word in words:
        moments.append(segment.start + letters[position][1])
        position += len(word)
    ends = [*moments[1:], segment.end]
    return [
        (min(begin, segment.end), min(finish, segment.end), word)
        for begin, finish, word in zip(moments, ends, words, strict=True)
    ]
