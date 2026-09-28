import chromadb


def inspect_chroma(limit: int = 3):
    """
    Inspects chunks, metadata, and raw embedding vectors stored in ChromaDB.
    """
    print("=" * 80)
    print("CHROMADB EMBEDDING & STORAGE INSPECTION")
    print("=" * 80)

    # 1. Connect to the local persistent ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection("ragforge_chunks")
    except Exception as e:
        print(f"Error: Collection 'ragforge_chunks' not found. ({e})")
        print("Run 'python scripts/sync_to_chroma.py' first to index chunks.")
        return

    total_chunks = collection.count()
    print(f"Collection Name:       {collection.name}")
    print(f"Total Vectors Indexed: {total_chunks}")
    print(f"Storage Directory:     ./chroma_db")
    print(f"Inspecting First {limit} Chunks...\n")

    # 2. Fetch sample records with documents, metadata, and raw embedding vectors
    sample = collection.get(
        limit=limit,
        include=["embeddings", "documents", "metadatas"],
    )

    ids = sample.get("ids", [])
    documents = sample.get("documents", [])
    metadatas = sample.get("metadatas", [])
    embeddings = sample.get("embeddings", [])

    for i in range(len(ids)):
        chunk_id = ids[i]
        content = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else []

        print("=" * 80)
        print(f"CHUNK ID:         {chunk_id}")
        print(f"DOCUMENT ID:      {meta.get('document_id', 'N/A')}")
        print(f"PAGE NUMBER:      {meta.get('page_number', 'N/A')}")
        print(f"SECTION:          {meta.get('section', 'N/A')}")
        print(f"TEXT PREVIEW:     {content[:150]}...")
        if len(embedding) > 0:
            print(f"EMBEDDING DIM:    {len(embedding)} dimensions (Float32)")
            print(
                f"VECTOR VALUES:    [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, ... , {embedding[-1]:.4f}]"
            )
        else:
            print("EMBEDDING:        Not loaded")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_chroma(limit=3)
