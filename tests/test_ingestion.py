from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.services.ingestion import IngestionService
from app.models.document_chunk import DocumentChunk


def test_ingest_document(tmp_path: Path):
    database_url = "sqlite:///:memory:"

    engine = create_engine(
        database_url,
        connect_args={
            "check_same_thread": False,
        },
    )

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=engine)

    file_path = tmp_path / "example.txt"

    file_path.write_text(
        """
PostgreSQL indexes improve query performance.

B-tree indexes support equality and range queries.
""",
        encoding="utf-8",
    )

    db = TestingSessionLocal()

    service = IngestionService(db)

    document = service.ingest(
        str(file_path)
    )

    assert document.id is not None
    assert document.filename == "example.txt"
    assert document.title == "example"

    assert len(document.content) > 0

    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document.id
        )
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    assert len(chunks) > 0

    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].section is None
    assert len(chunks[0].content) > 0

    db.close()