import os
from dataclasses import dataclass
from typing import Any
import chromadb

from app.services.embedding import EmbeddingService


@dataclass
class ChromaSearchResult:
    """Standardized representation of a chunk retrieved from ChromaDB."""
    id: int
    content: str
    document_id: int
    chunk_index: int
    page_number: int | None
    section: str | None
    score: float  # Cosine similarity (0.0 to 1.0)


class ChromaSearchService:
    """
    Production-grade ChromaDB Vector Store Service.
    
    Why ChromaDB?
    - Uses HNSW (Hierarchical Navigable Small World) index for sub-millisecond approximate nearest neighbor (ANN) search.
    - Zero external database server dependencies (persists to local disk).
    - Stores metadata alongside vectors for filtered retrieval.
    """

    COLLECTION_NAME = "ragforge_chunks"
    PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

    def __init__(
        self,
        persist_directory: str | None = None,
        embedder: EmbeddingService | None = None,
    ):
        self.persist_dir = persist_directory or self.PERSIST_DIR
        self.embedder = embedder or EmbeddingService()

        # 1. Initialize persistent storage client on disk
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 2. Get or create collection configured for cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of vectors in ChromaDB."""
        return self.collection.count()

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """
        Batch-insert chunks into ChromaDB with pre-computed or generated embeddings.
        """
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [str(c["id"]) for c in batch]
            documents = [c["content"] for c in batch]
            metadatas = [
                {
                    "chunk_id": int(c["id"]),
                    "document_id": int(c["document_id"]),
                    "chunk_index": int(c["chunk_index"]),
                    "page_number": int(c.get("page_number") or 0),
                    "section": str(c.get("section") or ""),
                }
                for c in batch
            ]

            # Re-use pre-computed embeddings if available, else compute using MiniLM
            if all("embedding" in c and c["embedding"] is not None for c in batch):
                embeddings = [c["embedding"] for c in batch]
            else:
                embeddings = self.embedder.embed_texts(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[tuple[ChromaSearchResult, float]]:
        """
        Perform semantic similarity search in ChromaDB.
        Returns a list of (ChromaSearchResult, similarity_score) tuples.
        """
        query_embedding = self.embedder.embed_text(query)
        where_filter = {"document_id": document_id} if document_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output: list[tuple[ChromaSearchResult, float]] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # Convert cosine distance to cosine similarity: [0 to 1]
            similarity = max(0.0, 1.0 - float(dist))

            result = ChromaSearchResult(
                id=int(cid),
                content=doc,
                document_id=meta.get("document_id", 0),
                chunk_index=meta.get("chunk_index", 0),
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                score=similarity,
            )
            output.append((result, similarity))

        return output

    def reset(self) -> None:
        """Clear all indexed data from ChromaDB."""
        self.client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
