# Output formats

The transcription result is saved in seven formats. All files are UTF-8 with `\n` line breaks. The speaker is "Канал N" (Channel N) for the channels of a stereo recording; if the channels are mixed into one (`--mono`, a mono recording or identical channels), there is no speaker.

| Format | Used for |
| :----- | :------- |
| [`txt`](#txt) | Reading the text: one segment per line |
| [`json`](#json) | Programs: information about the recording and the segments |
| [`jsonl`](#jsonl) | Programs and streams: one segment per line, files can be appended to each other |
| [`srt`](#srt) | Subtitles for video players and editors |
| [`vtt`](#vtt) | WebVTT subtitles for browsers |
| [`ass`](#ass) | Aegisub subtitles with the speaker in the Name field |
| [`csv`](#csv) | Spreadsheets and analytics: Excel, pandas |

The examples below are the result of `stt-ru transcribe call.wav` for a stereo recording of a conversation.

## TXT { #txt }

``` text
Канал 1: Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.
Канал 0: Угу
Канал 0: Угу
Канал 1: Я в свою очередь с вами прощаюсь. Всего доброго, до свидания.
Канал 0: До свидания.
```

## JSON { #json }

An object with the fields `source` (file name), `model`, `duration` (recording duration, s), `channels` (number of recognized channels) and `segments`. A segment has `start` and `end` in seconds, `channel`, `speaker` and `text`; for mixed channels `channel` and `speaker` are `null`.

``` json
{
  "source": "call.wav",
  "model": "gigaam-v3-e2e-rnnt",
  "duration": 15.751,
  "channels": 2,
  "segments": [
    {
      "start": 0.994,
      "end": 9.054,
      "channel": 1,
      "speaker": "Канал 1",
      "text": "Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии."
    },
    {
      "start": 3.49,
      "end": 3.934,
      "channel": 0,
      "speaker": "Канал 0",
      "text": "Угу"
    }
  ]
}
```

## JSON Lines { #jsonl }

One segment per line with the file name in `source`: the results of several files can be collected into one file (this is how `--summary out/all.jsonl` works).

``` json
{"source": "call.wav", "start": 0.994, "end": 9.054, "channel": 1, "speaker": "Канал 1", "text": "Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии."}
{"source": "call.wav", "start": 3.49, "end": 3.934, "channel": 0, "speaker": "Канал 0", "text": "Угу"}
```

## SRT { #srt }

The speaker is at the beginning of the subtitle text.

``` text
1
00:00:00,994 --> 00:00:09,054
Канал 1: Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.

2
00:00:03,490 --> 00:00:03,934
Канал 0: Угу
```

## WebVTT { #vtt }

The speaker is in the `<v>` voice tag and at the beginning of the text: players that do not show voices still show who is speaking.

``` text
WEBVTT

00:00:00.994 --> 00:00:09.054
<v Канал 1>Канал 1: Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.

00:00:03.490 --> 00:00:03.934
<v Канал 0>Канал 0: Угу
```

## ASS { #ass }

Subtitles in the format saved by [Aegisub](https://aegisub.org/): the speaker is in the Name field, the recording file in `Audio File`, times are accurate to hundredths of a second. The format is the same as the ASS files of version 0.2.

``` text
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.99,0:00:09.05,Default,Канал 1,0,0,0,,Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии.
Dialogue: 0,0:00:03.49,0:00:03.93,Default,Канал 0,0,0,0,,Угу
```

## CSV { #csv }

Columns `Audio File`, `Start`, `End`, `Name`, `Text`, times in milliseconds, comma separator. The file starts with a BOM, so Excel opens Cyrillic text without any setup; CSV printed to the console has no BOM, so for Excel save it with `--output-dir` or `--summary`. The columns are the same as in the CSV of version 0.2, so spreadsheets and reports built on it keep working.

``` text
Audio File,Start,End,Name,Text
call.wav,994,9054,Канал 1,"Наталья, благодарю вас за ожидание. По данному вопросу я сейчас ваш звонок переведу на специалиста профильного подразделения. Оставайтесь, пожалуйста, на линии."
call.wav,3490,3934,Канал 0,Угу
```
