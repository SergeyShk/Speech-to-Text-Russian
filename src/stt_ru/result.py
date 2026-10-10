from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """
    Фрагмент распознанной речи

    Аргументы:
        start (float): Начало, с
        end (float): Конец, с
        text (str): Распознанный текст
        channel (int): Номер канала записи, с нуля; None - каналы сведены в один

    Пример использования:
        >>> from stt_ru import Segment
        >>> segment = Segment(0.99, 9.05, "Наталья, благодарю вас за ожидание.", channel=1)
        >>> segment.speaker, segment.duration
        ('Канал 1', 8.06)
    """

    start: float
    end: float
    text: str
    channel: int | None = None

    @property
    def speaker(self) -> str | None:
        """Говорящий: «Канал N» для канала записи, None для сведенных каналов"""
        return None if self.channel is None else f"Канал {self.channel}"

    @property
    def duration(self) -> float:
        """Длительность, с, с точностью до миллисекунды"""
        return round(self.end - self.start, 3)


@dataclass(frozen=True)
class Transcript:
    """
    Результат распознавания записи

    Аргументы:
        segments (tuple[Segment, ...]): Фрагменты по времени начала
        duration (float): Длительность записи, с
        channels (int): Число распознанных каналов: 1, если каналы сведены в один
        model (str): Модель распознавания
        source (str): Имя файла записи; None - запись не из файла

    Пример использования:
        >>> from stt_ru import Segment, Transcript
        >>> segments = (Segment(0.5, 1.2, "Алло.", 0), Segment(1.0, 3.0, "Здравствуйте!", 1))
        >>> Transcript(segments, duration=3.0, channels=2, model="gigaam-v3-e2e-rnnt").text
        'Алло. Здравствуйте!'
    """

    segments: tuple[Segment, ...]
    duration: float
    channels: int
    model: str
    source: str | None = None

    @property
    def text(self) -> str:
        """Текст всех фрагментов по времени начала через пробел"""
        return " ".join(segment.text for segment in self.segments)
