import os

from dotenv import load_dotenv
from openai import OpenAI

from app.retriever import retrieve_context

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = os.getenv("GROQ_MODEL")


def answer_from_knowledge(question: str) -> str:
    if not question.strip():
        return "Please enter a question."

    if not MODEL:
        raise RuntimeError("GROQ_MODEL is missing from .env")

    matches = retrieve_context(question, top_k=3)

    if not matches:
        return "I couldn't find relevant information. Please contact clinic staff."

    context = "\n\n".join(
        match["text"] for match in matches
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a dental clinic receptionist. "
                    "Answer using only the supplied knowledge-base context. "
                    "If the context does not contain the answer, say that "
                    "clinic staff must confirm it. Do not invent policies, "
                    "prices, availability, or bookings. Do not diagnose "
                    "medical conditions. For difficulty breathing, advise "
                    "urgent emergency medical care."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Knowledge-base context:\n{context}\n\n"
                    f"Patient question: {question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or (
        "Sorry, I couldn't generate an answer."
    )


if __name__ == "__main__":
    question = "What time does the clinic open?"
    print(f"Question: {question}")
    print(f"Answer: {answer_from_knowledge(question)}")
