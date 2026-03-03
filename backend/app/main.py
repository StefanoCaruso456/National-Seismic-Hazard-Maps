from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field, ValidationError

from app.config import settings


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)


openai_client: OpenAI | None = None
pinecone_client: Pinecone | None = None


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


def get_openai_client() -> OpenAI:
    global openai_client
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    if openai_client is None:
        openai_client = OpenAI(api_key=settings.openai_api_key)
    return openai_client


def get_pinecone_index():
    global pinecone_client
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not configured")
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return pinecone_client.Index(settings.pinecone_index_name)


def embed_question(question: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=question,
    )
    return response.data[0].embedding


def normalize_matches(query_response: object) -> list:
    if hasattr(query_response, "matches"):
        return query_response.matches or []
    if isinstance(query_response, dict):
        return query_response.get("matches", [])
    return []


def normalize_metadata(match: object) -> dict:
    metadata = getattr(match, "metadata", None)
    if metadata is None and isinstance(match, dict):
        metadata = match.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def match_score(match: object) -> float:
    value = getattr(match, "score", None)
    if value is None and isinstance(match, dict):
        value = match.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_citation(metadata: dict, score: float) -> dict:
    try:
        line_start = int(metadata.get("line_start", 1))
    except (TypeError, ValueError):
        line_start = 1
    try:
        line_end = int(metadata.get("line_end", line_start))
    except (TypeError, ValueError):
        line_end = line_start

    return {
        "file_path": metadata.get("file_path", "unknown"),
        "line_start": line_start,
        "line_end": line_end,
        "score": round(score, 4),
    }


def metadata_chunk_text(metadata: dict) -> str:
    return (
        metadata.get("chunk_text")
        or metadata.get("text")
        or metadata.get("content")
        or metadata.get("code")
        or ""
    )


def build_context(citations: list[dict], chunks: list[str]) -> str:
    parts = []
    for idx, (citation, chunk) in enumerate(zip(citations, chunks), start=1):
        parts.append(
            (
                f"[{idx}] {citation['file_path']}:{citation['line_start']}-{citation['line_end']}\n"
                f"{chunk.strip()}"
            )
        )
    return "\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legacy code assistant. Use only retrieved context. "
                    "If context is insufficient, say so clearly and suggest next queries."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Answer with concise technical detail and reference citation numbers like [1], [2]."
                ),
            },
        ],
    )
    content = completion.choices[0].message.content
    return content.strip() if content else "No answer generated."


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        question_vector = embed_question(payload.question)
        index = get_pinecone_index()
        results = index.query(
            vector=question_vector,
            top_k=payload.top_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vector retrieval failed: {exc}") from exc

    matches = normalize_matches(results)
    if not matches:
        return QueryResponse(
            answer=(
                "I could not find relevant indexed code for that question. "
                "Try a more specific function name, file name, or keyword."
            ),
            citations=[],
        )

    citations: list[dict] = []
    chunks: list[str] = []
    for match in matches:
        metadata = normalize_metadata(match)
        score = match_score(match)
        citations.append(extract_citation(metadata, score))
        chunks.append(metadata_chunk_text(metadata))

    context = build_context(citations, chunks)

    try:
        answer = generate_answer(payload.question, context)
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Response validation error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    return QueryResponse(answer=answer, citations=citations)
