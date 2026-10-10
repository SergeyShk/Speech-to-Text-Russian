.PHONY: help uv deps lock lint ruff format mypy test test-cov clean clean-build clean-pyc clean-test build publish publish-test docs-build docs-serve docs-deploy
.DEFAULT_GOAL := help
SRC_PATH := src
TESTS_PATH := tests
PACKAGE := stt_ru

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

uv: ## Проверить наличие uv
	@which uv >/dev/null 2>&1 || { \
		echo "uv не установлен. Выполните 'curl -LsSf https://astral.sh/uv/install.sh | sh' или 'brew install uv'"; \
		exit 1; \
	}

deps: uv ## Установить зависимости
ifeq ($(MODE), ci)
	uv sync --locked --all-groups
else
	uv sync --all-groups
endif

lock: uv ## Обновить lock-файл до последних версий зависимостей
	uv lock --upgrade

lint: ruff mypy ## Запустить все проверки кода

ruff: deps ## Проверить и отформатировать код с помощью ruff
ifeq ($(MODE), ci)
	uv run ruff check $(SRC_PATH) $(TESTS_PATH)
	uv run ruff format $(SRC_PATH) $(TESTS_PATH) --check
else
	uv run ruff check $(SRC_PATH) $(TESTS_PATH) --fix
	uv run ruff format $(SRC_PATH) $(TESTS_PATH)
endif

format: deps ## Отформатировать код
	uv run ruff format $(SRC_PATH) $(TESTS_PATH)

mypy: deps ## Проверить типы с помощью mypy
	uv run mypy

test: deps ## Запустить тесты
	uv run pytest

test-cov: deps ## Запустить тесты с проверкой покрытия
	uv run pytest --cov $(PACKAGE)

clean: clean-build clean-pyc clean-test ## Удалить все артефакты
	rm -f .coverage coverage.xml

clean-build: ## Удалить артефакты сборки
	rm -fr build/ dist/ .eggs/ target/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Удалить артефакты компиляции
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

clean-test: ## Удалить артефакты тестирования и линтинга
	rm -fr .pytest_cache .mypy_cache .ruff_cache

build: clean uv ## Собрать дистрибутив
	uv build
	ls -l dist

publish: build ## Опубликовать релиз на PyPI
	uv publish

publish-test: build ## Опубликовать релиз на TestPyPI
	uv publish --index testpypi

docs-build: deps ## Собрать документацию
	rm -fr site/
	uv run mkdocs build --strict

docs-serve: deps ## Запустить сервер документации
	uv run mkdocs serve

docs-deploy: deps ## Задеплоить документацию
	uv run mkdocs gh-deploy
