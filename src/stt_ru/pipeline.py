import os
from collections.abc import Sequence
from pathlib import Path

from .asr import DEFAULT_MODEL, DEFAULT_QUANTIZATION, SAMPLE_RATES, Recognizer
from .audio import Audio, load_audio
from .audio.loader import Samples
from .result import Segment, Transcript
from .vad import VadOptions

RESAMPLE_RATE = 16000


class Transcriber:
    """
    Распознавание аудиофайлов

    Описание:
        Каналы стереозаписи распознаются отдельно: в записях разговоров каждый канал - свой
        говорящий. Файл с частотой, которую модель не принимает, передискретизируется
        в 16 кГц через ffmpeg (см. load_audio)

    Аргументы:
        model (str): Имя модели onnx-asr (см. Recognizer)
        quantization (str): Квантизация весов: "int8" или None - исходные веса
        model_dir (str): Каталог с файлами модели
        providers (list[str]): Провайдеры ONNX Runtime; None - все доступные
        vad (VadOptions): Параметры поиска речи
        split_channels (bool): Распознавать каналы отдельно; False - сводить в один
        ffmpeg (str): Путь к ffmpeg; None - поиск через find_ffmpeg

    Исключения:
        ModelError: Если модель не загружается

    Пример использования:
        >>> from stt_ru import Transcriber
        >>> transcript = Transcriber().transcribe("call.wav")  # doctest: +SKIP
        >>> for segment in transcript.segments:  # doctest: +SKIP
        ...     print(segment.speaker, segment.start, segment.end, segment.text)
        Канал 1 0.994 9.054 Наталья, благодарю вас за ожидание. ...
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        quantization: str | None = DEFAULT_QUANTIZATION,
        model_dir: str | os.PathLike[str] | None = None,
        providers: Sequence[str] | None = None,
        vad: VadOptions | None = None,
        split_channels: bool = True,
        ffmpeg: str | os.PathLike[str] | None = None,
    ) -> None:
        self.recognizer = Recognizer(
            model, quantization=quantization, model_dir=model_dir, providers=providers, vad=vad
        )
        self.split_channels = split_channels
        self.ffmpeg = ffmpeg

    def transcribe(self, path: str | os.PathLike[str]) -> Transcript:
        """
        Распознавание аудиофайла

        Аргументы:
            path (str): Путь к файлу (см. load_audio)

        Вывод:
            Transcript: Фрагменты речи по времени начала

        Исключения:
            FileNotFoundError: Если файла нет
            AudioError: Если файл не читается
        """
        audio = load_audio(path, ffmpeg=self.ffmpeg)
        if audio.sample_rate not in SAMPLE_RATES:
            audio = load_audio(path, sample_rate=RESAMPLE_RATE, ffmpeg=self.ffmpeg)
        return self.transcribe_audio(audio, source=Path(path).name)

    def transcribe_audio(self, audio: Audio, source: str | None = None) -> Transcript:
        """
        Распознавание записи в памяти

        Аргументы:
            audio (Audio): Запись с частотой из SAMPLE_RATES
            source (str): Имя файла записи для результата

        Вывод:
            Transcript: Фрагменты речи по времени начала; фрагменты без текста отбрасываются

        Исключения:
            AudioError: Если модель не принимает частоту записи
        """
        tracks: list[tuple[int | None, Samples]]
        if self.split_channels and audio.num_channels > 1:
            tracks = [(index, audio.channel(index)) for index in range(audio.num_channels)]
        else:
            tracks = [(None, audio.mono())]
        if audio.samples.shape[0] == 0:
            found: list[list[tuple[float, float, str]]] = [[] for _ in tracks]
        else:
            found = self.recognizer.recognize(
                [samples for _, samples in tracks], audio.sample_rate
            )
        segments = [
            Segment(start, end, text, channel)
            for (channel, _), track in zip(tracks, found, strict=True)
            for start, end, text in track
            if text
        ]
        segments.sort(key=lambda segment: (segment.start, segment.channel or 0))
        return Transcript(
            tuple(segments), audio.duration, len(tracks), self.recognizer.model, source
        )
