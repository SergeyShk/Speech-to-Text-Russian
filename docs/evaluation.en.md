# Quality evaluation

!!! info ""
    **stt-ru evaluate**, **stt_ru.evaluation**

## Description

The command transcribes the recordings from your references with one or several models and computes the word (WER) and character (CER) error rates and the speed. This way you can choose a model for your own recordings: calls, lectures, dictation. The `eval` extra is required:

``` bash
pip install "stt-ru[eval]"
stt-ru evaluate eval.csv                                         # default model
stt-ru evaluate eval.csv -m gigaam-v3-e2e-rnnt -m gigaam-v3-rnnt -o report.json
```

``` text
Модель              WER, %  CER, %  Скорость  Записей  Ошибок
gigaam-v3-e2e-rnnt     0.0     0.0     49.8x        2       0
gigaam-v3-rnnt         0.0     0.0     52.4x        2       0
```

*   **WER and CER** are the shares of word and character substitutions, deletions and insertions (Levenshtein distance) over all recordings together: the sum of errors divided by the sum of words or characters of the references. Spaces count in CER.
*   **Скорость** (speed) is how many times faster than real time recognition runs, including reading the file.
*   **Записей** (records) and **Ошибок** (errors) are how many reference rows there are and how many of them failed.

Models are loaded one after another; a model that fails to load is marked in the table, and the others are evaluated. Progress and errors go to stderr, the table to stdout.

## References { #manifest }

References are a CSV file, a JSON Lines file or a folder. Files are UTF-8, and recording paths are relative to the reference file.

**CSV** with the columns `audio`, `text` and an optional `channel`. The separator is a comma, a semicolon (as Excel saves in the Russian locale) or a tab, column names may be in any case, and empty rows are skipped.

``` text
audio,text,channel
calls/0001.wav,"Добрый день, компания «Ромашка», чем могу помочь?",1
calls/0001.wav,"Здравствуйте, хочу узнать статус заказа.",0
lectures/intro.mp3,"Сегодня мы поговорим о распознавании речи.",
```

**JSON Lines** with the same fields:

``` json
{"audio": "calls/0001.wav", "text": "Добрый день, компания «Ромашка», чем могу помочь?", "channel": 1}
{"audio": "lectures/intro.mp3", "text": "Сегодня мы поговорим о распознавании речи."}
```

**A folder** where every audio file has a reference with the same name: `call.wav` and `call.txt`.

`channel` is the channel number of a stereo recording where a channel is a speaker: the reference is compared only with the text of that channel. If there is one track (a mono recording, `--mono` or identical channels), the whole text is compared. A file that appears in the references several times is transcribed once.

## Normalization { #normalization }

Before counting, the reference and the recognized text are brought to one form: lowercase, "ё" as "е", everything except letters and digits becomes a space. Decomposed letters with diacritics are composed, and stress marks and invisible characters (soft hyphen, zero-width space) are removed.

Numbers are not converted between digits and words. If digits appear only in the reference or only in the recognized text, a warning is printed after the table: WER is inflated by the different spelling of numbers. GigaAM-v3 E2E writes numbers as digits, while GigaAM-v3 RNN-T and CTC write them as words, so for a fair comparison of such models write the numbers in the references the way the model you need writes them, or compare on recordings without numbers.

## Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `РАЗМЕТКА` (references) | - | CSV, JSONL or a folder |
| `-m`, `--model` | `gigaam-v3-e2e-rnnt` | Model; can be repeated, a repeated name counts once |
| `-o`, `--output` | - | Detailed report in JSON |
| `-q`, `--quiet` | off | Do not print progress; errors are always printed |

The other model and speech detection parameters are on the [Models](models.md#options) page; `--model-dir` is allowed only with one model. The references and arguments are checked before the models are loaded.

The exit code is 1 if a model failed to load, a recording failed or the report was not saved, and 2 for invalid arguments or references.

## Report { #report }

`-o report.json` saves the reference path and, for each model, `model`, `error` (loading error), `wer`, `cer`, `speed`, the number of records `files` and errors `failed`, `digits_mismatch` and `results` per reference row: `audio`, `channel`, `reference`, `hypothesis`, `duration`, `seconds`, `words`, `substitutions`, `deletions`, `insertions`, `chars`, `char_errors`, `wer`, `cer` and `error`. With `hypothesis` and the error counts it is easy to find the recordings where the model makes the most mistakes.

## Python library { #python }

``` python
>>> from stt_ru import Transcriber
>>> from stt_ru.evaluation import evaluate, format_table, load_manifest

>>> samples = load_manifest("eval.csv")
>>> report = evaluate(Transcriber("gigaam-v3-e2e-rnnt"), samples)
>>> report.wer, report.cer
(0.0, 0.0)
>>> print(format_table([report]))
Модель              WER, %  CER, %  Скорость  Записей  Ошибок
gigaam-v3-e2e-rnnt     0.0     0.0     48.6x        2       0
```

`load_manifest` returns a list of `Sample(audio, text, channel)`, and `evaluate` returns a `ModelReport` with the `FileResult` of each recording and the overall `wer`, `cer` and `speed`. `normalize(text)` is the normalization applied before counting.
