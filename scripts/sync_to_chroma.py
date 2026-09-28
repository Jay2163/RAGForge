from app.db.database import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.services.chroma_search import ChromaSearchService


def sync_postgres_to_chroma():
    """
    Syncs all document chunks and pre-computed embeddings from PostgreSQL into ChromaDB.
    """
    db = SessionLocal()
    chroma_service = ChromaSearchService()

    try:
        print("=" * 70)
        print("SYNCING POSTGRESQL CHUNKS TO CHROMADB")
        print("=" * 70)

        # 1. Fetch chunks from PostgreSQL
        chunks = db.query(DocumentChunk).filter(DocumentChunk.embedding.is_not(None)).all()
        print(f"Found {len(chunks)} chunks in PostgreSQL.")

        # 2. Prepare payload
        payload = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ]

        # 3. Clear existing and add to ChromaDB
        print("Resetting ChromaDB collection...")
        chroma_service.reset()

        print("Indexing chunks into ChromaDB...")
        chroma_service.add_chunks(payload, batch_size=100)

        total_in_chroma = chroma_service.count()
        print(f"Successfully synced {total_in_chroma} chunks into ChromaDB!")
        print("ChromaDB storage directory: ./chroma_db")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    sync_postgres_to_chroma()
