from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    # Placeholder response so deployment and wiring can be validated
    # before full RAG retrieval/generation is implemented.
    return QueryResponse(
        answer=(
            "Backend scaffold is live. Implement Pinecone retrieval and LLM "
            "generation in /api/query next."
        ),
        citations=[
            {
                "file_path": "README.md",
                "line_start": 1,
                "line_end": 1,
                "score": 0.0,
            }
        ],
    )
