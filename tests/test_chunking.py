from app.services.chunking import ChunkingService
from app.services.document_parser import ParsedPage



def test_chunk_text():
    text = """
PostgreSQL indexes improve query performance.

B-tree is the default PostgreSQL index type.
It supports equality and range queries.

Indexes also increase storage requirements
and can slow down write operations.
"""

    service = ChunkingService(
        max_chunk_size=100,
        overlap=20,
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.content.strip()


def test_empty_text():
    service = ChunkingService()

    assert service.chunk_text("") == []


def test_whitespace_text():
    service = ChunkingService()

    assert service.chunk_text("   \n\n   ") == []


def test_invalid_configuration():
    try:
        ChunkingService(
            max_chunk_size=100,
            overlap=100,
        )
        assert False
    except ValueError:
        assert True


def test_large_paragraph_is_split():
    text = (
        "PostgreSQL is a powerful relational database system. "
        "It supports indexing and query optimization. "
        "Indexes can significantly improve read performance. "
        "However, indexes also introduce write overhead. "
        "Database engineers must therefore choose indexes carefully."
    )

    service = ChunkingService(
        max_chunk_size=100,
        overlap=20,
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1


def test_overlap_preserves_words():
    text = """
PostgreSQL indexes improve query performance.
B-tree indexes are useful for equality queries.
Database engineers should analyze query plans.
"""

    service = ChunkingService(
        max_chunk_size=100,
        overlap=20,
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    # The overlap should consist of complete words,
    # not arbitrary character fragments.
    for chunk in chunks[1:]:
        assert not chunk.content.startswith("e ")
        assert not chunk.content.startswith("y ")

def test_chunk_metadata():
    text = """
PostgreSQL indexes improve query performance.

B-tree is the default PostgreSQL index type.
"""

    service = ChunkingService(
        max_chunk_size=100,
        overlap=20,
    )

    chunks = service.chunk_text(text)

    assert chunks[0].chunk_index == 0

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.content

def test_chunk_document_preserves_page_metadata():

    pages = [
        ParsedPage(
            page_number=1,
            content=(
                "PostgreSQL indexes improve query performance."
            ),
        ),
        ParsedPage(
            page_number=2,
            content=(
                "B-tree indexes support equality "
                "and range queries."
            ),
        ),
    ]

    service = ChunkingService(
        max_chunk_size=100,
        overlap=20,
    )

    chunks = service.chunk_document(pages)

    assert len(chunks) > 0

    assert chunks[0].page_number == 1

    page_numbers = {
        chunk.page_number
        for chunk in chunks
    }

    assert 1 in page_numbers
    assert 2 in page_numbers

    chunk_indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    assert chunk_indexes == list(
        range(len(chunks))
    )