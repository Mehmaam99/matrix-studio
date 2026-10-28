import chromadb
from sentence_transformers import SentenceTransformer

from app.knowledge import load_knowledge, split_knowledge

COLLECTION_NAME = "clinic_knowledge"
DB_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    text = load_knowledge()
    chunks = split_knowledge(text)

    if not chunks:
        raise ValueError("Knowledge base is empty.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=[f"clinic-faq-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "clinic_faqs.txt"} for _ in chunks],
    )

    return collection, model


if __name__ == "__main__":
    collection, model = build_vector_store()
    print(f"Vector store created successfully.")
    print(f"Stored chunks: {collection.count()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
