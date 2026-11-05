# Демо на Hugging Face Spaces: образ выпуска с примерами записей и пределами для публичного доступа
ARG IMAGE=ghcr.io/sergeyshk/stt-ru:latest
FROM ${IMAGE}
COPY examples /opt/stt-ru/examples
CMD ["serve", "--host", "0.0.0.0", "--port", "7860", "--diarize", "--embed-from", "https://huggingface.co", "--examples", "/opt/stt-ru/examples", "--notice", "Демонстрация на бесплатном сервере: не загружайте конфиденциальные записи. Файлы удаляются через час.", "--max-size", "50", "--max-duration", "10", "--max-queue", "10", "--keep", "1", "-q"]
