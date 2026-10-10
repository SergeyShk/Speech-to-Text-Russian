import argparse
import contextlib
import io
import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn

import onnxruntime as rt

from . import __version__
from .asr import DEFAULT_MODEL, DEFAULT_QUANTIZATION
from .batch import (
    Batch,
    Job,
    Log,
    check_options,
    check_watch,
    find_jobs,
    ignore,
    log_to_stderr,
    summary_line,
    watch,
)
from .exceptions import ModelError, ParameterError
from .export import FORMATS, render, to_csv
from .pipeline import Transcriber
from .result import Transcript
from .vad import VadOptions

INTERRUPTED = 130
DEVICES: dict[str, list[str] | None] = {
    "auto": None,
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}
DEVICE_HINTS = {
    "cuda": "нужны NVIDIA GPU и пакет onnxruntime-gpu вместо onnxruntime",
    "coreml": "CoreML есть только в сборке ONNX Runtime для macOS",
    "directml": "нужен пакет onnxruntime-directml вместо onnxruntime",
}
# Форматы, которые можно печатать подряд для нескольких файлов
STREAM_FORMATS = ("txt", "jsonl", "csv")
ARGPARSE_ERRORS = (
    (re.compile(r"unrecognized arguments: (.*)"), "неизвестные аргументы: {0}"),
    (
        re.compile(r"argument (\S+): invalid choice: (.*?) \(choose from (.*)\)"),
        "аргумент {0}: неизвестное значение {1} (допустимые: {2})",
    ),
    (re.compile(r"argument (\S+): invalid (\w+) value: (.*)"), "аргумент {0}: неверное число {2}"),
    (re.compile(r"argument (\S+): expected one argument"), "аргумент {0}: нужно значение"),
    (re.compile(r"the following arguments are required: (.*)"), "нужны аргументы: {0}"),
    (
        re.compile(r"ambiguous option: (\S+) could match (.*)"),
        "неоднозначный параметр {0}: подходят {1}",
    ),
)


class HelpFormatter(argparse.HelpFormatter):
    """Справка с русской строкой использования"""

    def add_usage(
        self,
        usage: str | None,
        actions: Iterable[argparse.Action],
        groups: Iterable[Any],
        prefix: str | None = None,
    ) -> None:
        super().add_usage(usage, actions, groups, "использование: " if prefix is None else prefix)


class ArgumentParser(argparse.ArgumentParser):
    """Разбор аргументов с русскими справкой и сообщениями об ошибках"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(formatter_class=HelpFormatter, add_help=False, **kwargs)
        self.commands: dict[str, ArgumentParser] = {}
        self._positionals.title = "аргументы"
        self.options = self.add_argument_group("параметры")
        self.options.add_argument(
            "-h", "--help", action="help", help="показать эту справку и выйти"
        )

    def error(self, message: str) -> NoReturn:
        for pattern, template in ARGPARSE_ERRORS:
            if match := pattern.fullmatch(message):
                message = template.format(*match.groups())
                break
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка: {message}\n")


def build_parser() -> ArgumentParser:
    """
    Разбор аргументов команды stt-ru

    Вывод:
        ArgumentParser: Разбор с командой transcribe
    """
    parser = ArgumentParser(
        prog="stt-ru",
        usage="%(prog)s [-h] [--version] КОМАНДА ...",
        description="Распознавание русской речи моделью GigaAM-v3 и другими моделями onnx-asr",
    )
    parser.options.add_argument(
        "--version",
        action="version",
        version=f"stt-ru {__version__}, onnx-asr {version('onnx-asr')}",
        help="показать версии stt-ru и onnx-asr и выйти",
    )
    commands = parser.add_subparsers(dest="command", title="команды", metavar="КОМАНДА")
    transcribe = commands.add_parser(
        "transcribe",
        help="распознать речь в аудиофайлах",
        prog="stt-ru transcribe",
        description="Распознает речь в файлах и каталогах. Без --output-dir печатает результат; "
        "с ним сохраняет результат каждого файла в выбранных форматах. Каналы стереозаписи "
        "распознаются отдельно: канал - говорящий",
    )
    parser.commands["transcribe"] = transcribe
    transcribe.add_argument("paths", nargs="+", metavar="ПУТЬ", help="аудиофайлы и каталоги")
    output = transcribe.add_argument_group("результат")
    output.add_argument(
        "-o", "--output-dir", metavar="КАТАЛОГ", help="каталог для результатов каждого файла"
    )
    output.add_argument(
        "-f",
        "--formats",
        metavar="ФОРМАТЫ",
        help="форматы через запятую: " + ", ".join(FORMATS) + "; по умолчанию txt без "
        "--output-dir и txt,srt,json с ним",
    )
    output.add_argument(
        "--summary",
        metavar="ФАЙЛ",
        help="общий файл CSV или JSONL с фрагментами файлов, распознанных в этом запуске; "
        "дописывается, если есть",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="распознавать файлы, у которых результаты уже есть",
    )
    files = transcribe.add_argument_group("файлы")
    files.add_argument(
        "-r", "--recursive", action="store_true", help="искать файлы и во вложенных каталогах"
    )
    files.add_argument(
        "--delete-source",
        action="store_true",
        help="удалять файл после распознавания и записи результатов; нужен --output-dir "
        "или --summary; файлы, у которых результаты уже есть, не удаляются",
    )
    files.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="сколько файлов распознавать одновременно (по умолчанию 1)",
    )
    files.add_argument("--ffmpeg", metavar="ПУТЬ", help="путь к программе ffmpeg")
    monitor = transcribe.add_argument_group("мониторинг")
    monitor.add_argument(
        "--watch",
        action="store_true",
        help="следить за каталогами и распознавать новые файлы до прерывания (Ctrl+C)",
    )
    monitor.add_argument(
        "--interval",
        type=float,
        default=10.0,
        metavar="СЕК",
        help="пауза между просмотрами каталогов (по умолчанию 10)",
    )
    monitor.add_argument(
        "--min-age",
        type=float,
        default=10.0,
        metavar="СЕК",
        help="сколько секунд файл не должен меняться, чтобы его взять (по умолчанию 10)",
    )
    model = transcribe.add_argument_group("модель")
    model.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="ИМЯ",
        help=f"модель onnx-asr (по умолчанию {DEFAULT_MODEL})",
    )
    model.add_argument(
        "--quantization",
        default=DEFAULT_QUANTIZATION,
        metavar="ТИП",
        help=f"квантизация весов: {DEFAULT_QUANTIZATION} (по умолчанию) или none",
    )
    model.add_argument("--model-dir", metavar="КАТАЛОГ", help="каталог с файлами модели")
    model.add_argument(
        "--device",
        choices=list(DEVICES),
        default="auto",
        help="устройство: auto (по умолчанию), cpu, cuda, coreml, directml",
    )
    model.add_argument("--mono", action="store_true", help="сводить каналы в один")
    vad = transcribe.add_argument_group("поиск речи")
    vad.add_argument("--vad-threshold", type=float, metavar="P", help="порог вероятности речи")
    vad.add_argument("--min-speech-ms", type=float, metavar="МС", help="наименьший фрагмент")
    vad.add_argument("--max-speech-s", type=float, metavar="С", help="наибольший фрагмент")
    vad.add_argument("--min-silence-ms", type=float, metavar="МС", help="наименьшая пауза")
    vad.add_argument("--speech-pad-ms", type=float, metavar="МС", help="запас по краям")
    output.add_argument(
        "-q", "--quiet", action="store_true", help="не выводить ход работы (ошибки выводятся)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Запуск команды stt-ru

    Описание:
        Коды выхода: 0 - успех, 1 - ошибка распознавания, загрузки модели или закрытый
        вывод, 2 - неверные аргументы, 130 - прерывание (Ctrl+C). Вывод в файл
        или конвейер идет в UTF-8

    Аргументы:
        argv (list[str]): Аргументы командной строки; None - sys.argv
    """
    if isinstance(sys.stdout, io.TextIOWrapper) and not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    # Справка и ошибки по-русски не должны падать на консоли с однобайтной кодировкой
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(2)
    try:
        transcribe(parser.commands[args.command], args)
    except KeyboardInterrupt:
        _exit(INTERRUPTED)
    except BrokenPipeError:
        _exit(1)


def _exit(code: int) -> NoReturn:
    # Без ожидания потоков распознавания: файлы пишутся целиком или никак
    with contextlib.suppress(OSError, ValueError):
        sys.stderr.flush()
    os._exit(code)


def _vad_options(args: argparse.Namespace) -> VadOptions:
    options = {
        "threshold": args.vad_threshold,
        "min_speech_duration_ms": args.min_speech_ms,
        "max_speech_duration_s": args.max_speech_s,
        "min_silence_duration_ms": args.min_silence_ms,
        "speech_pad_ms": args.speech_pad_ms,
    }
    return VadOptions(**{name: value for name, value in options.items() if value is not None})


def _providers(parser: ArgumentParser, device: str) -> list[str] | None:
    providers = DEVICES[device]
    if providers is not None and providers[0] not in rt.get_available_providers():
        parser.error(f"устройство {device} недоступно: {DEVICE_HINTS[device]}")
    return providers


def transcribe(parser: ArgumentParser, args: argparse.Namespace) -> None:
    """
    Команда transcribe

    Описание:
        Аргументы и файлы проверяются до загрузки модели; без файлов модель не загружается

    Аргументы:
        parser (ArgumentParser): Разбор аргументов для сообщений об ошибках
        args (argparse.Namespace): Разобранные аргументы
    """
    default = "txt,srt,json" if args.output_dir else "txt"
    formats = [fmt.strip().lower() for fmt in (args.formats or default).split(",") if fmt.strip()]
    log: Log = ignore if args.quiet else log_to_stderr
    try:
        check_options(formats, args.summary, args.workers)
        if args.watch:
            check_watch(args.interval, args.min_age)
        vad = _vad_options(args)
        if args.output_dir is None:
            if len(formats) != 1:
                parser.error("без --output-dir печатается один формат")
            if args.watch:
                parser.error("для --watch нужен --output-dir")
            if args.delete_source and args.summary is None:
                parser.error("для --delete-source нужен --output-dir или --summary")
        if args.watch and not all(Path(path).is_dir() for path in args.paths):
            parser.error("--watch следит только за каталогами")
        jobs = find_jobs(args.paths, args.recursive)
        if args.output_dir is None and len(jobs) > 1 and formats[0] not in STREAM_FORMATS:
            parser.error(
                f"формат {formats[0]} печатается только для одного файла: "
                "для нескольких укажите --output-dir"
            )
        providers = _providers(parser, args.device)
    except (FileNotFoundError, ParameterError) as error:
        parser.error(str(error))
    if not jobs and not args.watch:
        log("Файлов для распознавания нет")
        return
    log(f"Загрузка модели {args.model}...")
    try:
        transcriber = Transcriber(
            args.model,
            quantization=None if args.quantization.lower() == "none" else args.quantization,
            model_dir=args.model_dir,
            providers=providers,
            vad=vad,
            split_channels=not args.mono,
            ffmpeg=args.ffmpeg,
        )
    except ModelError as error:
        print(f"stt-ru: ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    batch = Batch(
        transcriber,
        args.output_dir,
        formats,
        summary=args.summary,
        overwrite=args.overwrite,
        delete_source=args.delete_source,
        workers=args.workers,
        log=log,
        on_done=None if args.output_dir else _printer(formats[0], several=len(jobs) > 1),
    )
    if args.watch:
        log("Мониторинг каталогов, Ctrl+C - выход")
        failed = bool(
            watch(
                batch,
                args.paths,
                recursive=args.recursive,
                interval=args.interval,
                min_age=args.min_age,
            ).failed
        )
    else:
        report = batch.run(jobs)
        if args.output_dir or len(jobs) > 1:
            log(summary_line(report))
        failed = bool(report.failed)
    if failed:
        sys.exit(1)


def _printer(fmt: str, several: bool) -> Callable[[Job, Transcript], None]:
    printed = 0

    def show(job: Job, transcript: Transcript) -> None:
        nonlocal printed
        if fmt == "txt" and several:
            print(f"==> {job.source} <==")
        if fmt == "csv":
            sys.stdout.write(to_csv(transcript, header=printed == 0))
        else:
            sys.stdout.write(render(transcript, fmt))
        sys.stdout.flush()
        printed += 1

    return show
