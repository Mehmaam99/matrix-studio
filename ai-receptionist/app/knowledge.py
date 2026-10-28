from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_FILE = Path("data/clinic_faqs.txt")


def load_knowledge() -> str:
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    return KNOWLEDGE_FILE.read_text(encoding="utf-8")


def split_knowledge(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    content = load_knowledge()
    chunks = split_knowledge(content)

    print(f"Loaded characters: {len(content)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
