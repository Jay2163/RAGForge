
from pathlib import Path

import pytest

from app.services.document_parser import DocumentParser


def test_parse_text_file(tmp_path: Path):
    file_path = tmp_path / "example.txt"

    file_path.write_text(
        "PostgreSQL is a relational database.",
        encoding="utf-8",
    )

    parser = DocumentParser()

    result = parser.parse(str(file_path))

    assert result.filename == "example.txt"
    assert result.file_type == "txt"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert (
        result.pages[0].content
        == "PostgreSQL is a relational database."
    )


def test_parse_markdown_file(tmp_path: Path):
    file_path = tmp_path / "example.md"

    file_path.write_text(
        "# PostgreSQL\n\n"
        "PostgreSQL supports indexes.",
        encoding="utf-8",
    )

    parser = DocumentParser()

    result = parser.parse(str(file_path))

    assert result.filename == "example.md"
    assert result.file_type == "md"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1

    assert "# PostgreSQL" in result.pages[0].content
    assert "PostgreSQL supports indexes." in result.pages[0].content


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "example.docx"

    file_path.write_bytes(b"fake document")

    parser = DocumentParser()

    with pytest.raises(ValueError):
        parser.parse(str(file_path))


def test_parse_pdf():
    parser = DocumentParser()

    result = parser.parse(
        "tests/data/postgresql_guide.pdf"
    )

    assert result.file_type == "pdf"
    assert result.filename == "postgresql_guide.pdf"
    assert len(result.pages) > 0

    for page in result.pages:
        assert page.page_number >= 1
