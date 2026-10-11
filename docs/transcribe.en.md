# The transcribe command

!!! info ""
    **stt-ru transcribe**

## Description

The command recognizes speech in files and folders. Without `--output-dir` it prints the result to the console; with it, it saves the result of every file in the chosen [formats](formats.md). The channels of a stereo recording are recognized separately: in a recorded conversation each channel is its own speaker, "Канал 0" and "Канал 1" (Channel 0 and Channel 1). Identical channels are recognized once; `--mono` mixes the channels into one.

``` bash
stt-ru transcribe call.wav                              # text to the console
stt-ru transcribe call.mp3 -f srt > call.srt            # subtitles to a file
stt-ru transcribe calls/ -o out/                        # txt, srt and json of every file in out/
stt-ru transcribe calls/ -r -o out/ -f srt,ass,csv --summary out/all.csv
stt-ru transcribe calls/ -o out/ --watch --delete-source    # folder monitoring
```

## Files { #input }

*   A file given as an argument is taken with any extension: the format is detected from its content.
*   From a folder, files with audio and video extensions are taken: `.wav`, `.flac`, `.ogg`, `.oga`, `.opus`, `.mp3`, `.m4a`, `.aac`, `.amr`, `.wma`, `.aif`, `.aiff`, `.au`, `.caf`, `.w64`, `.3gp`, `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm` in any case. Hidden files and folders (names starting with a dot) are skipped; `-r` also searches subfolders.
*   One file given by different paths is transcribed once.
*   M4A, AAC, AMR, WMA and the sound of videos are read by ffmpeg (see [Installation](installation.md)).

## Result { #output }

**Without `--output-dir`** the result is printed in one format, txt by default. For several files in a row, txt is printed with a `==> file <==` header before each file, and jsonl and csv with a single header; other formats are printed only for one file. Output to a file or a pipe is always UTF-8.

**With `--output-dir`** the result of every file is saved in the `--formats` formats (txt, srt and json by default) under the file name: `calls/2026/a.wav` from the `calls` folder gives `out/2026/a.txt`, `out/2026/a.srt` and `out/2026/a.json`. Each result file is written entirely or not at all. A file that already has all its results is skipped: running again resumes an interrupted batch, and `--overwrite` transcribes such files again. Two files with the same result path (ignoring case) are an error before any work starts.

**`--summary`** collects the segments of all files transcribed in this run into one CSV or JSONL file with the file path in the first column. If the summary file exists, rows are appended, and the CSV header is written once.

## Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `ПУТЬ...` (paths) | - | Files and folders |
| `-o`, `--output-dir` | - | Folder for the results of every file |
| `-f`, `--formats` | `txt`; `txt,srt,json` with `-o` | Comma-separated formats: `txt`, `json`, `jsonl`, `srt`, `vtt`, `ass`, `csv` |
| `--summary` | - | One `.csv` or `.jsonl` file with the segments of the files of this run |
| `--overwrite` | off | Transcribe files that already have results |
| `-r`, `--recursive` | off | Also search subfolders |
| `--delete-source` | off | Delete a file after transcribing it and writing its results |
| `-j`, `--workers` | `1` | How many files to transcribe at once |
| `--watch` | off | Watch folders and transcribe new files |
| `--interval` | `10` | Pause between folder scans, s |
| `--min-age` | `10` | How many seconds a file must stay unchanged to be taken |
| `-q`, `--quiet` | off | Do not print progress; errors are always printed |

The model, device and speech detection parameters (`-m`, `--quantization`, `--model-dir`, `--device`, `--mono`, `--ffmpeg`, `--vad-*`) are shared by all commands and described on the [Models](models.md#options) page. All parameters and files are checked before the model is loaded; if there are no files, the model is not loaded.

## Deleting sources { #delete-source }

`--delete-source` deletes a file after all its results and summary rows are written. It needs `--output-dir` or `--summary`, so that the result is not lost. A file that already has results is not deleted: it may be another recording with the same name. A failed deletion is an error of that file.

## Folder monitoring { #watch }

`--watch` scans the folders every `--interval` seconds and takes a file when it has not changed for `--min-age` seconds: this way a recording that is still being copied is not transcribed. It needs `--output-dir`, and the paths must be folders. A failed file is retried only after it changes. A missing folder and files with the same result names do not stop monitoring: a message is printed about them. Monitoring runs until Ctrl+C.

``` bash
stt-ru transcribe /archive/wav -r -o /archive/out -f csv,ass --summary /archive/out/all.csv --watch --delete-source
```

## Progress and exit codes { #exit-codes }

Progress goes to stderr and the console result to stdout (the messages are in Russian):

``` text
Загрузка модели gigaam-v3-e2e-rnnt...
[1/2] calls/a.wav: фрагментов 5, 15.8 с записи за 0.3 с
[2/2] calls/b.wav: ошибка - ffmpeg не прочитал файл calls/b.wav: Error opening input: Invalid data found when processing input
Готово: распознано 1 из 2, пропущено 0, с ошибкой 1
```

An error in one file does not stop the others. Ctrl+C cancels the queue and waits for the files in progress; a second Ctrl+C exits immediately.

| Code | When |
| :--: | :--- |
| `0` | All files are transcribed or skipped |
| `1` | A file or model loading error, a closed output (for example, a pipe to `head`) |
| `2` | Invalid arguments |
| `130` | Interrupted with Ctrl+C |
