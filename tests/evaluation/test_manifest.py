import csv
import json

import pytest

from stt_ru.evaluation import Sample, load_manifest
from stt_ru.exceptions import ParameterError


@pytest.fixture
def audio(tmp_path):
    for name in ("a.wav", "b.mp3"):
        (tmp_path / name).write_bytes(b"audio")
    return tmp_path


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    return path


class TestCsv:
    def test_rows(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text(
            "audio,text,channel\na.wav,Привет,\nb.mp3, Ответ ,1\n", encoding="utf-8-sig"
        )
        assert load_manifest(manifest) == [
            Sample(audio / "a.wav", "Привет"),
            Sample(audio / "b.mp3", " Ответ ", 1),
        ]

    def test_without_channel_and_absolute_path(self, audio, tmp_path):
        manifest = tmp_path / "sub" / "eval.csv"
        manifest.parent.mkdir()
        manifest.write_text(f'text,audio\n"да, нет",{audio / "a.wav"}\n', encoding="utf-8")
        assert load_manifest(str(manifest)) == [Sample(audio / "a.wav", "да, нет")]

    def test_missing_columns(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("file,label\na.wav,да\n", encoding="utf-8")
        with pytest.raises(ParameterError, match="нет колонок audio, text"):
            load_manifest(manifest)

    def test_empty_file(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("", encoding="utf-8")
        with pytest.raises(ParameterError, match="нет колонок"):
            load_manifest(manifest)

    def test_header_only(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("audio,text\n", encoding="utf-8")
        with pytest.raises(ParameterError, match="В разметке нет записей"):
            load_manifest(manifest)

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ("a.wav,  ", "строка 3: пустой эталон"),
            (",да", "строка 3: нет пути к аудио"),
            ("a.wav", "строка 3: пустой эталон"),
        ],
    )
    def test_wrong_row(self, audio, row, message):
        manifest = audio / "eval.csv"
        manifest.write_text(f"audio,text\na.wav,да\n{row}\n", encoding="utf-8")
        with pytest.raises(ParameterError, match=message):
            load_manifest(manifest)

    @pytest.mark.parametrize("channel", ["-1", "левый", "1.5", "²"])
    def test_wrong_channel(self, audio, channel):
        manifest = audio / "eval.csv"
        manifest.write_text(f"audio,text,channel\na.wav,да,{channel}\n", encoding="utf-8")
        with pytest.raises(ParameterError, match=f"строка 2: канал должен быть .*: {channel}"):
            load_manifest(manifest)

    @pytest.mark.parametrize("delimiter", [";", "\t"])
    def test_delimiter(self, audio, delimiter):
        manifest = audio / "eval.csv"
        rows = ["Audio ;Text; Channel", "a.wav;да, нет;0", "b.mp3;Ответ;", ";;", ""]
        text = "\r\n".join(rows).replace(";", delimiter)
        manifest.write_text(text, encoding="utf-8-sig", newline="")
        assert load_manifest(manifest) == [
            Sample(audio / "a.wav", "да, нет", 0),
            Sample(audio / "b.mp3", "Ответ"),
        ]

    def test_unquoted_delimiter_in_text(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("audio,text\na.wav,Наталья, благодарю\n", encoding="utf-8")
        with pytest.raises(ParameterError, match="строка 2: значений больше, чем колонок"):
            load_manifest(manifest)

    def test_trailing_empty_values_from_excel(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("audio;text\na.wav;да, нет;;\n", encoding="utf-8")
        assert load_manifest(manifest) == [Sample(audio / "a.wav", "да, нет")]

    def test_blank_rows(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("audio,text\n , \na.wav,да\n,,,\n,\n,нет\n", encoding="utf-8")
        with pytest.raises(ParameterError, match="строка 6: нет пути к аудио"):
            load_manifest(manifest)


class TestJsonl:
    def test_rows(self, audio):
        manifest = write_jsonl(
            audio / "eval.JSONL",
            [{"audio": "a.wav", "text": "да"}, {"audio": "b.mp3", "text": "нет", "channel": 0}],
        )
        manifest.write_text(manifest.read_text("utf-8") + "\n  \n", encoding="utf-8")
        assert load_manifest(manifest) == [
            Sample(audio / "a.wav", "да"),
            Sample(audio / "b.mp3", "нет", 0),
        ]

    def test_unicode_line_separators(self, audio):
        text = "раз\u2028два\u2029три\x85четыре"
        manifest = write_jsonl(audio / "eval.jsonl", [{"audio": "a.wav", "text": text}])
        assert load_manifest(manifest) == [Sample(audio / "a.wav", text)]

    def test_channel_as_string_or_null(self, audio):
        rows = [
            {"audio": "a.wav", "text": "да", "channel": "1"},
            {"audio": "a.wav", "text": "да", "channel": None},
        ]
        samples = load_manifest(write_jsonl(audio / "eval.jsonl", rows))
        assert [sample.channel for sample in samples] == [1, None]

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"audio": "a.wav", "text": "да", "channel": True}, "канал должен быть"),
            ({"audio": "a.wav", "text": "да", "channel": 1.0}, "канал должен быть"),
            ({"audio": "a.wav", "text": ["да"]}, "пустой эталон"),
            ({"audio": 1, "text": "да"}, "нет пути к аудио"),
            ({"text": "да"}, "нет пути к аудио"),
            (["a.wav", "да"], "строка должна быть объектом JSON"),
        ],
    )
    def test_wrong_row(self, audio, row, message):
        manifest = write_jsonl(audio / "eval.jsonl", [{"audio": "a.wav", "text": "да"}, row])
        with pytest.raises(ParameterError, match=f"строка 2: {message}"):
            load_manifest(manifest)

    def test_wrong_json(self, audio):
        manifest = audio / "eval.jsonl"
        manifest.write_text('{"audio": "a.wav", "text": "да"\n', encoding="utf-8")
        with pytest.raises(ParameterError, match="строка 1: неверный JSON"):
            load_manifest(manifest)


class TestDirectory:
    def test_pairs(self, audio):
        (audio / "a.txt").write_text("Привет\n", encoding="utf-8-sig")
        (audio / "b.txt").write_text("Ответ", encoding="utf-8")
        (audio / ".hidden.wav").write_bytes(b"audio")
        (audio / "notes.md").write_text("не аудио", encoding="utf-8")
        assert load_manifest(audio) == [
            Sample(audio / "a.wav", "Привет\n"),
            Sample(audio / "b.mp3", "Ответ"),
        ]

    def test_missing_references(self, audio):
        (audio / "a.txt").write_text("Привет", encoding="utf-8")
        with pytest.raises(ParameterError, match=r"для 1 файлов: b\.mp3$"):
            load_manifest(audio)

    def test_many_missing_references(self, tmp_path):
        for number in range(7):
            (tmp_path / f"{number}.wav").write_bytes(b"audio")
        with pytest.raises(ParameterError, match=r"для 7 файлов: 0\.wav, .*, 4\.wav и других"):
            load_manifest(tmp_path)

    def test_empty_reference(self, audio):
        (audio / "a.txt").write_text(" \n", encoding="utf-8")
        (audio / "b.txt").write_text("Ответ", encoding="utf-8")
        with pytest.raises(ParameterError, match=r"a\.txt: пустой эталон"):
            load_manifest(audio)

    def test_no_audio(self, tmp_path):
        with pytest.raises(ParameterError, match="В разметке нет записей"):
            load_manifest(tmp_path)


def test_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="Разметка не найдена"):
        load_manifest(tmp_path / "eval.csv")


def test_missing_audio(audio):
    manifest = audio / "eval.csv"
    manifest.write_text("audio,text\nc.wav,да\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"Аудиофайл из разметки не найден: .*c\.wav"):
        load_manifest(manifest)


def test_unknown_format(audio):
    manifest = audio / "eval.tsv"
    manifest.write_text("audio\ttext\n", encoding="utf-8")
    with pytest.raises(ParameterError, match="CSV, JSONL или каталогом"):
        load_manifest(manifest)


class TestEncoding:
    def test_csv(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("audio,text\na.wav,Привет\n", encoding="cp1251")
        with pytest.raises(ParameterError, match=r"eval\.csv: файл не в кодировке UTF-8"):
            load_manifest(manifest)

    def test_csv_header(self, audio):
        manifest = audio / "eval.csv"
        manifest.write_text("аудио,текст\n", encoding="cp1251")
        with pytest.raises(ParameterError, match="не в кодировке UTF-8"):
            load_manifest(manifest)

    def test_jsonl(self, audio):
        manifest = audio / "eval.jsonl"
        manifest.write_text('{"audio": "a.wav", "text": "Привет"}\n', encoding="cp1251")
        with pytest.raises(ParameterError, match=r"eval\.jsonl: файл не в кодировке UTF-8"):
            load_manifest(manifest)

    def test_directory(self, audio):
        (audio / "a.txt").write_text("Привет", encoding="cp1251")
        with pytest.raises(ParameterError, match=r"a\.txt: файл не в кодировке UTF-8"):
            load_manifest(audio)


def test_long_reference(audio):
    text = "слово " * 50_000
    manifest = audio / "eval.csv"
    manifest.write_text(f'audio,text\na.wav,"{text}"\n', encoding="utf-8")
    limit = csv.field_size_limit()
    assert load_manifest(manifest) == [Sample(audio / "a.wav", text)]
    assert csv.field_size_limit() == limit


def test_multiline_text_line_number(audio):
    manifest = audio / "eval.csv"
    manifest.write_text('audio,text\na.wav,"да\nнет"\n,да\n', encoding="utf-8")
    with pytest.raises(ParameterError, match="строка 4: нет пути к аудио"):
        load_manifest(manifest)
