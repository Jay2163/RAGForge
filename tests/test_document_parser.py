
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

def test_parse_markdown_sections(tmp_path: Path):
    file_path = tmp_path / "example.md"

    file_path.write_text(
        "# PostgreSQL\n\n"
        "PostgreSQL is a relational database.\n\n"
        "## Indexes\n\n"
        "Indexes improve query performance.\n\n"
        "## Transactions\n\n"
        "Transactions provide atomicity.",
        encoding="utf-8",
    )

    parser = DocumentParser()

    result = parser.parse(
        str(file_path)
    )

    assert len(result.blocks) == 3

    assert result.blocks[0].section == "PostgreSQL"
    assert (
        result.blocks[0].content
        == "PostgreSQL is a relational database."
    )

    assert result.blocks[1].section == "Indexes"
    assert (
        result.blocks[1].content
        == "Indexes improve query performance."
    )

    assert result.blocks[2].section == "Transactions"
    assert (
        result.blocks[2].content
        == "Transactions provide atomicity."
    )

    for block in result.blocks:
        assert block.page_number == 1
