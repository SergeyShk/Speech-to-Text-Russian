class SttRuError(Exception):
    """Базовое исключение stt-ru"""


class AudioError(SttRuError):
    """Аудио не читается или имеет неподходящий формат"""


class ModelError(SttRuError):
    """Модель распознавания не поддерживается или не загружается"""


class ParameterError(SttRuError, ValueError):
    """Недопустимое значение параметра"""
