from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.vector_store import (
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_MODEL,
    build_vector_store,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    if not query.strip():
        return []

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        build_vector_store()

    model = get_embedding_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": document,
            "distance": distance,
            "metadata": metadata,
        }
        for document, distance, metadata in zip(
            documents, distances, metadatas
        )
    ]


if __name__ == "__main__":
    question = "When is the dental clinic open?"
    matches = retrieve_context(question)

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(matches)}")

    for index, match in enumerate(matches, start=1):
        print(f"\n--- Match {index} ---")
        print(f"Distance: {match['distance']:.4f}")
        print(match["text"])
