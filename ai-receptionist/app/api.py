from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import run_receptionist_graph

app = FastAPI(
    title="AI Receptionist API",
    version="0.2.0",
    description="LangGraph-powered dental clinic AI receptionist",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = run_receptionist_graph(
            message=request.message,
            session_id=request.session_id,
        )

        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
        )
    except Exception as exc:
        print(f"Chat request failed: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail="AI service temporarily unavailable.",
        ) from exc
