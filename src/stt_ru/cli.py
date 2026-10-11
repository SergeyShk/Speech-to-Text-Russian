import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
import re
import socket
import sys
import tempfile
import typing
import warnings
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
    check_target,
    check_watch,
    find_jobs,
    ignore,
    log_to_stderr,
    summary_line,
    watch,
    write_atomic,
)
from .evaluation import (
    ModelReport,
    evaluate as evaluate_model,
    format_table,
    load_manifest,
    require_jiwer,
)
from .exceptions import MissingDependencyError, ModelError, ParameterError
from .export import FORMATS, render, to_csv
from .pipeline import Transcriber
from .result import Transcript
from .vad import VadOptions

INTERRUPTED = 130
TOKEN_VARIABLE = "STT_RU_TOKEN"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
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
        description="Распознавание русской речи моделью GigaAM-v3 и другими моделями onnx-asr: "
        "файлы и каталоги, веб-интерфейс и REST API, оценка качества моделей на своей разметке",
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
    _add_model_options(transcribe, several=False)
    output.add_argument(
        "-q", "--quiet", action="store_true", help="не выводить ход работы (ошибки выводятся)"
    )
    evaluate_parser = commands.add_parser(
        "evaluate",
        help="оценить качество моделей на своей разметке",
        prog="stt-ru evaluate",
        description="Распознает записи из разметки каждой моделью и считает долю ошибок "
        "слов (WER) и символов (CER) после нормализации: регистр, «ё», пунктуация. Числа "
        "не переводятся: если модель пишет цифры, а эталон - слова, WER завышен. Скорость - "
        "во сколько раз распознавание быстрее реального времени",
    )
    parser.commands["evaluate"] = evaluate_parser
    evaluate_parser.add_argument(
        "manifest",
        metavar="РАЗМЕТКА",
        help="CSV или JSONL в UTF-8 с полями audio, text и необязательным channel или "
        "каталог, где у каждого аудиофайла есть эталон .txt с тем же именем",
    )
    report = evaluate_parser.add_argument_group("отчет")
    report.add_argument("-o", "--output", metavar="ФАЙЛ", help="подробный отчет в JSON")
    report.add_argument(
        "-q", "--quiet", action="store_true", help="не выводить ход работы (ошибки выводятся)"
    )
    _add_model_options(evaluate_parser, several=True)
    serve_parser = commands.add_parser(
        "serve",
        help="запустить веб-интерфейс и REST API",
        prog="stt-ru serve",
        description="Веб-интерфейс и REST API: файлы ставятся в очередь и распознаются моделью, "
        "загруженной один раз; результат скачивается в любом формате. Токен доступа "
        f"задается переменной окружения {TOKEN_VARIABLE}. Задачи хранятся в памяти "
        "и пропадают при перезапуске",
    )
    parser.commands["serve"] = serve_parser
    server = serve_parser.add_argument_group("сервис")
    server.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="АДРЕС",
        help="адрес (по умолчанию 127.0.0.1 - только этот компьютер; 0.0.0.0 - все адреса)",
    )
    server.add_argument(
        "--port", type=int, default=8000, metavar="ПОРТ", help="порт (по умолчанию 8000)"
    )
    server.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="сколько файлов распознавать одновременно (по умолчанию 1)",
    )
    server.add_argument(
        "--max-size",
        type=float,
        default=500,
        metavar="МБ",
        help="наибольший размер файла (по умолчанию 500 МБ)",
    )
    server.add_argument(
        "--max-duration",
        type=float,
        default=180,
        metavar="МИН",
        help="наибольшая длительность записи (по умолчанию 180 минут)",
    )
    server.add_argument(
        "--max-queue",
        type=int,
        default=100,
        metavar="N",
        help="сколько файлов может ждать в очереди (по умолчанию 100)",
    )
    server.add_argument(
        "--keep",
        type=float,
        default=24,
        metavar="Ч",
        help="сколько часов хранить файл и результат после распознавания (по умолчанию 24)",
    )
    server.add_argument(
        "--data-dir",
        metavar="КАТАЛОГ",
        help="каталог для загруженных и временных файлов (по умолчанию системный временный)",
    )
    server.add_argument(
        "-q", "--quiet", action="store_true", help="не выводить ход работы (ошибки выводятся)"
    )
    _add_model_options(serve_parser, several=False)
    return parser


def _add_model_options(command: ArgumentParser, several: bool) -> None:
    model = command.add_argument_group("модель")
    if several:
        model.add_argument(
            "-m",
            "--model",
            action="append",
            metavar="ИМЯ",
            help=f"модель onnx-asr; можно повторить (по умолчанию {DEFAULT_MODEL})",
        )
    else:
        model.add_argument(
            "-m",
            "--model",
            default=DEFAULT_MODEL,
            metavar="ИМЯ",
            help=f"модель onnx-asr (по умолчанию {DEFAULT_MODEL})",
        )
    model.add_argument(
        "--quantization",
        type=str.lower,
        choices=[DEFAULT_QUANTIZATION, "none"],
        default=DEFAULT_QUANTIZATION,
        metavar="ТИП",
        help=f"квантизация весов: {DEFAULT_QUANTIZATION} (по умолчанию) или none",
    )
    model.add_argument(
        "--model-dir",
        metavar="КАТАЛОГ",
        help="каталог с файлами модели" + (" (только для одной модели)" if several else ""),
    )
    model.add_argument(
        "--device",
        choices=list(DEVICES),
        default="auto",
        help="устройство: auto (по умолчанию), cpu, cuda, coreml, directml",
    )
    audio = command.add_argument_group("аудио")
    audio.add_argument("--mono", action="store_true", help="сводить каналы в один")
    audio.add_argument("--ffmpeg", metavar="ПУТЬ", help="путь к программе ffmpeg")
    vad = command.add_argument_group("поиск речи")
    vad.add_argument("--vad-threshold", type=float, metavar="P", help="порог вероятности речи")
    vad.add_argument("--min-speech-ms", type=float, metavar="МС", help="наименьший фрагмент")
    vad.add_argument("--max-speech-s", type=float, metavar="С", help="наибольший фрагмент")
    vad.add_argument("--min-silence-ms", type=float, metavar="МС", help="наименьшая пауза")
    vad.add_argument("--speech-pad-ms", type=float, metavar="МС", help="запас по краям")


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
    warnings.showwarning = _show_warning
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    # Пути после параметров: stt-ru transcribe a.wav -o out b.wav
    if args.command == "transcribe" and not any(item.startswith("-") for item in extra):
        args.paths += extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(2)
    command = {"transcribe": transcribe, "evaluate": evaluate, "serve": serve}[args.command]
    try:
        command(parser.commands[args.command], args)
    except KeyboardInterrupt:
        _exit(INTERRUPTED)
    except BrokenPipeError:
        _exit(1)


def _show_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: typing.TextIO | None = None,
    line: str | None = None,
) -> None:
    # Без пути к исходнику и строки кода, как у остальных сообщений команды
    if sys.stderr is not None:
        with contextlib.suppress(OSError, ValueError):
            print(f"stt-ru: предупреждение: {message}", file=sys.stderr)


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


def _quantization(args: argparse.Namespace) -> str | None:
    return None if args.quantization == "none" else args.quantization


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
        if not formats:
            parser.error("не задано ни одного формата")
        check_options(formats, args.summary, args.workers, args.output_dir)
        if args.watch:
            check_watch(args.interval, args.min_age)
        vad = _vad_options(args)
        if args.output_dir is None:
            _check_printing(parser, args, formats)
        if args.watch and not all(Path(path).is_dir() for path in args.paths):
            parser.error("--watch следит только за каталогами")
        # О недоступных каталогах и совпадениях имен сообщает мониторинг: перезапуск не падает
        denied: list[str] = []
        jobs = find_jobs(
            args.paths,
            args.recursive,
            strict=not args.watch,
            names=args.output_dir is not None or args.summary is not None,
            log=denied.append,
        )
        if args.output_dir is None and len(jobs) > 1 and formats[0] not in STREAM_FORMATS:
            parser.error(
                f"формат {formats[0]} печатается только для одного файла: "
                "для нескольких укажите --output-dir"
            )
        providers = _providers(parser, args.device)
    except (OSError, ParameterError) as error:
        parser.error(str(error))
    # В мониторинге о недоступных каталогах сообщает сам мониторинг
    if args.watch:
        denied = []
    for message in denied:
        print(f"stt-ru: ошибка: {message}", file=sys.stderr)
    if not jobs and not args.watch:
        log("Файлов для распознавания нет")
        if denied:
            sys.exit(1)
        return
    log(f"Загрузка модели {args.model}...")
    try:
        transcriber = Transcriber(
            args.model,
            quantization=_quantization(args),
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
        failed = bool(report.failed or denied)
    if failed:
        sys.exit(1)


def _check_printing(parser: ArgumentParser, args: argparse.Namespace, formats: list[str]) -> None:
    if sys.stdout is None:
        print("stt-ru: ошибка: вывод закрыт", file=sys.stderr)
        sys.exit(1)
    if len(formats) != 1:
        parser.error("без --output-dir печатается один формат")
    if args.watch:
        parser.error("для --watch нужен --output-dir")
    if args.delete_source and args.summary is None:
        parser.error("для --delete-source нужен --output-dir или --summary")


def _printer(fmt: str, several: bool) -> Callable[[Job, Transcript], None]:
    printed = 0

    def show(job: Job, transcript: Transcript) -> None:
        nonlocal printed
        if several:
            # Файлы с одним именем из разных каталогов различаются только путем
            transcript = dataclasses.replace(transcript, source=str(job.source))
        if fmt == "txt" and several:
            print(f"==> {job.source} <==")
        if fmt == "csv":
            sys.stdout.write(to_csv(transcript, header=printed == 0))
        else:
            sys.stdout.write(render(transcript, fmt))
        sys.stdout.flush()
        printed += 1

    return show


def evaluate(parser: ArgumentParser, args: argparse.Namespace) -> None:
    """
    Команда evaluate

    Описание:
        Разметка, аргументы и jiwer проверяются до загрузки моделей. Модели загружаются
        по очереди; модель, которая не загрузилась, отмечается в таблице

    Аргументы:
        parser (ArgumentParser): Разбор аргументов для сообщений об ошибках
        args (argparse.Namespace): Разобранные аргументы
    """
    log: Log = ignore if args.quiet else log_to_stderr
    models = list(dict.fromkeys(args.model or [DEFAULT_MODEL]))
    try:
        vad = _vad_options(args)
        if args.model_dir is not None and len(models) > 1:
            parser.error("--model-dir задается только для одной модели")
        if args.output is not None:
            if Path(args.output).suffix.lower() != ".json":
                parser.error(f"отчет сохраняется в JSON: {args.output}")
            check_target(args.output, directory=False)
        unknown = [model for model in models if not _known_model(model)]
        if unknown:
            parser.error(f"неизвестные модели onnx-asr: {', '.join(unknown)}")
        samples = load_manifest(args.manifest)
        providers = _providers(parser, args.device)
    except (FileNotFoundError, ParameterError) as error:
        parser.error(str(error))
    try:
        require_jiwer()
    except MissingDependencyError as error:
        print(f"stt-ru: ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    log(f"Записей в разметке: {len(samples)}")
    reports: list[ModelReport] = []
    interrupted = False
    try:
        for model in models:
            log(f"Загрузка модели {model}...")
            try:
                transcriber = Transcriber(
                    model,
                    quantization=_quantization(args),
                    model_dir=args.model_dir,
                    providers=providers,
                    vad=vad,
                    split_channels=not args.mono,
                    ffmpeg=args.ffmpeg,
                )
            except ModelError as error:
                print(f"stt-ru: ошибка: {error}", file=sys.stderr)
                reports.append(ModelReport(model, error=str(error)))
                continue
            reports.append(evaluate_model(transcriber, samples, log=log))
            del transcriber
    except KeyboardInterrupt:
        # Оценка моделей идет часами: уже оцененные не должны пропадать
        interrupted = True
        print(
            f"stt-ru: оценка прервана, в таблице модели, оцененные целиком: {len(reports)}",
            file=sys.stderr,
        )
    if interrupted and not reports:
        sys.exit(INTERRUPTED)
    failed = any(report.error is not None or report.failed for report in reports)
    if args.output is not None:
        document = {"manifest": str(args.manifest), "models": [r.to_dict() for r in reports]}
        content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        try:
            write_atomic(Path(args.output), content.encode("utf-8"))
        except OSError as error:
            print(f"stt-ru: ошибка: отчет не сохранен в {args.output}: {error}", file=sys.stderr)
            failed = True
        else:
            log(f"Отчет: {args.output}")
    print(format_table(reports), flush=True)
    for report in reports:
        if report.digits_mismatch:
            print(
                f"{report.model}: в {report.digits_mismatch} записях цифры есть только в эталоне "
                "или только в распознанном тексте - если числа записаны по-разному (цифрами "
                "и словами), WER завышен",
                file=sys.stderr,
            )
    if interrupted:
        sys.exit(INTERRUPTED)
    if failed:
        sys.exit(1)


def _known_model(model: str) -> bool:
    # Модель с Hugging Face (имя с «/») проверяется только при загрузке
    if "/" in model:
        return True
    try:
        from onnx_asr.loader import AsrNames, AsrTypeNames
    except ImportError:
        return True
    return model in typing.get_args(AsrNames) + typing.get_args(AsrTypeNames)


def serve(parser: ArgumentParser, args: argparse.Namespace) -> None:
    """
    Команда serve

    Описание:
        Аргументы, зависимости и адрес проверяются до загрузки модели; сервис запускается
        после загрузки модели. Закрытый stderr не останавливает сервис

    Аргументы:
        parser (ArgumentParser): Разбор аргументов для сообщений об ошибках
        args (argparse.Namespace): Разобранные аргументы
    """
    warn = _tolerant(log_to_stderr)
    log = ignore if args.quiet else warn
    try:
        vad = _vad_options(args)
        if not 0 < args.port < 65536:
            parser.error(f"порт должен быть от 1 до 65535: {args.port}")
        for option, value in (
            ("--workers", args.workers),
            ("--max-size", args.max_size),
            ("--max-duration", args.max_duration),
            ("--max-queue", args.max_queue),
            ("--keep", args.keep),
        ):
            if not (math.isfinite(value) and value > 0):
                parser.error(f"{option} должен быть положительным числом: {value:g}")
        if args.data_dir is not None:
            if not args.data_dir or not Path(args.data_dir).is_dir():
                parser.error(f"каталога нет: {args.data_dir}")
            if not os.access(args.data_dir, os.W_OK | os.X_OK):
                parser.error(f"нет права записи в каталог {args.data_dir}")
        providers = _providers(parser, args.device)
    except ParameterError as error:
        parser.error(str(error))
    try:
        import uvicorn

        from .service import JobQueue, create_app
    except (ImportError, RuntimeError) as error:
        # Без python-multipart FastAPI бросает RuntimeError при объявлении загрузки файла
        package = getattr(error, "name", None) or "python-multipart"
        print(
            f"stt-ru: ошибка: Для сервиса нужен пакет {package}: pip install 'stt-ru[service]'",
            file=sys.stderr,
        )
        sys.exit(1)
    reason = _bind_error(args.host, args.port)
    if reason is not None:
        print(
            f"stt-ru: ошибка: не удалось занять адрес {args.host}:{args.port}: {reason}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Перевод строки в конце остается от секрета из файла или .env с CRLF
    token = os.environ.get(TOKEN_VARIABLE, "").strip() or None
    if token is None and args.host not in LOCAL_HOSTS:
        warn(
            f"stt-ru: предупреждение: сервис доступен по сети без токена; задайте токен "
            f"в переменной окружения {TOKEN_VARIABLE}"
        )
    log(f"Загрузка модели {args.model}...")
    try:
        transcriber = Transcriber(
            args.model,
            quantization=_quantization(args),
            model_dir=args.model_dir,
            providers=providers,
            vad=vad,
            split_channels=not args.mono,
            ffmpeg=args.ffmpeg,
        )
    except ModelError as error:
        print(f"stt-ru: ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    jobs = JobQueue(
        transcriber,
        args.data_dir,
        workers=args.workers,
        max_queue=args.max_queue,
        max_duration=args.max_duration * 60,
        keep=args.keep * 3600,
        log=log,
        errors=warn,
    )
    app = create_app(jobs, token=token, max_size=int(args.max_size * (1 << 20)))
    previous = tempfile.tempdir
    if args.data_dir is not None:
        # Туда же - тело загрузки, которое разбирает Starlette, и WAV, который пишет ffmpeg
        tempfile.tempdir = str(jobs.directory)
    address = f"http://{f'[{args.host}]' if ':' in args.host else args.host}:{args.port}"
    log(f"Веб-интерфейс: {address}, описание API: {address}/docs, Ctrl+C - выход")
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning" if args.quiet else "info",
            access_log=False,
            # Короче 10 с, после которых docker stop завершает процесс принудительно
            timeout_graceful_shutdown=3,
        )
    finally:
        tempfile.tempdir = previous
        jobs.close()


def _tolerant(log: Log) -> Log:
    def write(message: str) -> None:
        # Сервис работает и тогда, когда stderr закрыт: конвейер в head, закрытый терминал
        with contextlib.suppress(OSError, ValueError):
            log(message)

    return write


def _bind_error(host: str, port: int) -> str | None:
    # Проверка до загрузки модели: uvicorn сообщил бы о занятом порте только после нее
    try:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.create_server((host, port), family=family):
            return None
    except OSError as error:
        return error.strerror or str(error)
