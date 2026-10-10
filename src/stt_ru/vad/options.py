from dataclasses import dataclass

from onnx_asr.adapters import VadOptions as OnnxVadOptions

from ..exceptions import ParameterError


@dataclass(frozen=True)
class VadOptions:
    """
    Параметры поиска речи Silero VAD

    Описание:
        Перед распознаванием запись режется на фрагменты речи: модели принимают фрагменты
        не длиннее 20-30 с. Значения по умолчанию - значения Silero VAD и onnx-asr

    Аргументы:
        threshold (float): Вероятность речи, с которой начинается фрагмент, от 0 до 1
        neg_threshold (float): Вероятность, ниже которой фрагмент заканчивается;
            None - threshold - 0.15
        min_speech_duration_ms (float): Наименьшая длительность фрагмента, мс: короткие отбрасываются
        max_speech_duration_s (float): Наибольшая длительность фрагмента, с: длинные режутся
        min_silence_duration_ms (float): Наименьшая пауза, которая разделяет фрагменты, мс
        speech_pad_ms (float): Запас по краям фрагмента, мс

    Исключения:
        ParameterError: Если вероятность вне интервала от 0 до 1, neg_threshold больше threshold,
            длительность отрицательна или max_speech_duration_s не положительна

    Пример использования:
        >>> from stt_ru.vad import VadOptions
        >>> VadOptions(min_silence_duration_ms=300).as_kwargs()
        {'threshold': 0.5, 'min_speech_duration_ms': 250, 'max_speech_duration_s': 20,
         'min_silence_duration_ms': 300, 'speech_pad_ms': 30}
    """

    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: float = 250
    max_speech_duration_s: float = 20
    min_silence_duration_ms: float = 100
    speech_pad_ms: float = 30

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ParameterError(f"threshold должен быть от 0 до 1: {self.threshold}")
        if self.neg_threshold is not None and not 0 < self.neg_threshold <= self.threshold:
            raise ParameterError(
                f"neg_threshold должен быть от 0 до threshold ({self.threshold}): "
                f"{self.neg_threshold}"
            )
        for name in ("min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"):
            if getattr(self, name) < 0:
                raise ParameterError(f"{name} не может быть отрицательным: {getattr(self, name)}")
        if self.max_speech_duration_s <= 0:
            raise ParameterError(
                f"max_speech_duration_s должен быть положительным: {self.max_speech_duration_s}"
            )

    def as_kwargs(self) -> OnnxVadOptions:
        """
        Параметры для with_vad в onnx-asr

        Вывод:
            dict[str, float]: Параметры без neg_threshold, если он не задан
        """
        kwargs: OnnxVadOptions = {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_s": self.max_speech_duration_s,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }
        if self.neg_threshold is not None:
            kwargs["neg_threshold"] = self.neg_threshold
        return kwargs
