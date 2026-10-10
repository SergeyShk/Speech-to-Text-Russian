import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

import onnx_asr
import onnxruntime as rt
from onnx_asr.utils import ModelLoadingError, SampleRates

from ..audio.loader import Samples
from ..exceptions import AudioError, ModelError
from ..vad import VadOptions

DEFAULT_MODEL = "gigaam-v3-e2e-rnnt"
DEFAULT_QUANTIZATION = "int8"
SAMPLE_RATES: tuple[int, ...] = get_args(SampleRates)

Found = list[tuple[float, float, str]]


class Recognizer:
    """
    Распознавание речи моделью onnx-asr с поиском речи Silero VAD

    Описание:
        Модель задается именем onnx-asr: gigaam-v3-e2e-rnnt (по умолчанию), gigaam-v3-rnnt,
        t-tech/t-one, alphacep/vosk-model-ru, onnx-community/whisper-large-v3-turbo и другие.
        При первом использовании модель скачивается с Hugging Face в его кэш

    Аргументы:
        model (str): Имя модели onnx-asr
        quantization (str): Квантизация весов: "int8" или None - исходные веса
        model_dir (str): Каталог с файлами модели; если каталога нет, модель скачивается в него
        providers (list[str]): Провайдеры ONNX Runtime по приоритету,
            например ["CUDAExecutionProvider"]; None - все доступные
        vad (VadOptions): Параметры поиска речи; None - значения по умолчанию

    Исключения:
        ModelError: Если модель не поддерживается, у нее нет такой квантизации
            или ее не удалось скачать

    Атрибуты:
        model (str): Имя модели
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
        options = rt.SessionOptions()
        # Без этого ONNX Runtime пишет в stderr предупреждения о разбиении графа по провайдерам
        options.log_severity_level = 3
        try:
            asr = onnx_asr.load_model(
                model,
                None if model_dir is None else Path(model_dir),
                quantization=quantization,
                sess_options=options,
                providers=providers,
            )
            silero = onnx_asr.load_vad("silero", sess_options=options, providers=providers)
        except (ModelLoadingError, OSError) as error:
            variant = f" с квантизацией {quantization}" if quantization else ""
            raise ModelError(f"Не удалось загрузить модель {model}{variant}: {error}") from error
        self._asr = asr.with_vad(silero, **(vad or VadOptions()).as_kwargs())

    def recognize(self, waveforms: Sequence[Samples], sample_rate: int) -> list[Found]:
        """
        Распознавание одноканальных записей одной частоты

        Аргументы:
            waveforms (list[numpy.ndarray]): Одномерные массивы отсчетов float32
            sample_rate (int): Частота дискретизации, Гц, одна из SAMPLE_RATES

        Вывод:
            list[list[tuple[float, float, str]]]: Для каждой записи фрагменты речи:
                начало и конец в секундах и текст

        Исключения:
            AudioError: Если модель не принимает такую частоту дискретизации
        """
        if sample_rate not in SAMPLE_RATES:
            raise AudioError(
                f"Частота {sample_rate} Гц не поддерживается, допустимые: "
                + ", ".join(map(str, SAMPLE_RATES))
            )
        if not waveforms:
            return []
        results = self._asr.recognize(list(waveforms), sample_rate=cast(SampleRates, sample_rate))
        return [[(found.start, found.end, found.text) for found in result] for result in results]
