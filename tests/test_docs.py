"""У каждой русской страницы документации есть английская с теми же заголовками, якорями и ссылками"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"
PAGES = sorted(path for path in DOCS.rglob("*.md") if not path.name.endswith(".en.md"))


def anchors(text: str) -> set[str]:
    return set(re.findall(r"\{ #([\w-]+) \}", text))


def links(text: str) -> set[str]:
    return set(re.findall(r"\]\(([^)#\s]+\.md)", text))


@pytest.mark.parametrize("page", PAGES, ids=[str(page.relative_to(DOCS)) for page in PAGES])
def test_english_page(page):
    english = page.with_name(page.name[:-3] + ".en.md")
    assert english.is_file(), f"нет английской версии {english.relative_to(DOCS)}"
    russian_text = page.read_text(encoding="utf-8")
    english_text = english.read_text(encoding="utf-8")
    assert anchors(english_text) == anchors(russian_text)
    assert links(english_text) == links(russian_text)
    assert russian_text.count("\n## ") == english_text.count("\n## ")


def test_no_orphan_english_pages():
    english = {path.relative_to(DOCS) for path in DOCS.rglob("*.en.md")}
    assert english == {page.relative_to(DOCS).with_suffix(".en.md") for page in PAGES}
