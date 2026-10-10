class SttRuError(Exception):
    """Базовое исключение stt-ru"""


class AudioError(SttRuError):
    """Аудио не читается или имеет неподходящий формат"""
