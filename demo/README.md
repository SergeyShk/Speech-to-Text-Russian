---
title: stt-ru
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Распознавание русской речи и разделение говорящих
---

# stt-ru

Демонстрация [stt-ru](https://github.com/SergeyShk/Speech-to-Text-Russian): распознавание русской речи моделью GigaAM-v3 и разделение говорящих в записи с одной дорожкой. Загрузите файл или нажмите кнопку с примером: запись распознается, результат можно послушать вместе с текстом и скачать в форматах TXT, JSON, JSONL, SRT, VTT, ASS и CSV.

Это бесплатный сервер на CPU: запись до 10 минут и 50 МБ, в очереди до 10 файлов, файлы и результаты удаляются через час. Не загружайте конфиденциальные записи. Для своих записей лучше [установить stt-ru](https://sergeyshk.github.io/Speech-to-Text-Russian/installation/) или запустить [Docker-образ](https://sergeyshk.github.io/Speech-to-Text-Russian/docker/).

Описание API - на странице `/docs` приложения, документация проекта - на <https://sergeyshk.github.io/Speech-to-Text-Russian/>.

## Примеры

- `звонок.mp3` - запись телефонного разговора из тестов проекта, каналы разделяют говорящих.
- `диалог.mp3` - диалог, собранный из аудиокниг [LibriVox](https://librivox.org/) (общественное достояние): читают Марк Чульский (Н. Лесков, «На краю света») и Ганна Пономаренко (З. Гиппиус, «Чортова кукла»).

Код демонстрации лежит в папке [`demo`](https://github.com/SergeyShk/Speech-to-Text-Russian/tree/master/demo) репозитория и обновляется при каждом выпуске.
