import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    """
    Фрагмент распознанной речи

    Аргументы:
        start (float): Начало, с
        end (float): Конец, с
        text (str): Распознанный текст
        channel (int): Номер канала записи, с нуля; None - каналы сведены в один
        speaker_id (int): Номер говорящего при диаризации, с единицы, по порядку появления
            в записи; None - диаризации не было

    Пример использования:
        >>> from stt_ru import Segment
        >>> segment = Segment(0.99, 9.05, "Наталья, благодарю вас за ожидание.", channel=1)
        >>> segment.speaker, segment.duration
        ('Канал 1', 8.06)
        >>> Segment(0.99, 9.05, "Наталья, благодарю вас за ожидание.", speaker_id=2).speaker
        'Говорящий 2'
    """

    start: float
    end: float
    text: str
    channel: int | None = None
    speaker_id: int | None = None

    @property
    def speaker(self) -> str | None:
        """Говорящий: «Говорящий N» при диаризации, «Канал N» для канала записи, иначе None"""
        if self.speaker_id is not None:
            return f"Говорящий {self.speaker_id}"
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
        channel_map (tuple[int | None, ...]): Для каждого канала записи - номер канала,
            под которым распознана его речь: свой номер, номер совпавшего с ним канала
            или None, если дорожка одна; пусто - неизвестно

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
    channel_map: tuple[int | None, ...] = ()

    @property
    def text(self) -> str:
        """Текст всех фрагментов по времени начала через пробел"""
        return " ".join(segment.text for segment in self.segments)

    def save(self, path: str | os.PathLike[str], fmt: str | None = None) -> Path:
        """
        Сохранение в файл (см. stt_ru.export.save)

        Аргументы:
            path (str): Путь к файлу
            fmt (str): Формат: txt, json, jsonl, srt, vtt, ass, csv; None - по расширению файла

        Вывод:
            Path: Путь к файлу
        """
        from .export import save

        return save(self, path, fmt)
