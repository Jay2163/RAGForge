from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.chunking import ChunkingService
from app.services.document_parser import DocumentParser


class IngestionService:
    def __init__(
        self,
        db: Session,
        parser: DocumentParser | None = None,
        chunker: ChunkingService | None = None,
    ):
        self.db = db
        self.parser = parser or DocumentParser()
        self.chunker = chunker or ChunkingService()

    def ingest(self, file_path: str) -> Document:
        parsed_document = self.parser.parse(file_path)

        chunks = self.chunker.chunk_blocks(
            parsed_document.blocks
        )

        document = Document(
            filename=parsed_document.filename,
            title=Path(file_path).stem,
            content=parsed_document.full_text,
        )

        self.db.add(document)
        self.db.flush()

        document_chunks = []

        for chunk in chunks:
            document_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_number=chunk.page_number,
                section=chunk.section,
            )

            document_chunks.append(document_chunk)

        self.db.add_all(document_chunks)

        self.db.commit()

        self.db.refresh(document)

        return document