# LegacyLens Backend

FastAPI backend scaffold for Railway deployment.

## Local run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Railway settings

- Source repo: `StefanoCaruso456/National-Seismic-Hazard-Maps`
- Root Directory: `backend`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Healthcheck Path: `/health`

Required env vars:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default: `legacylens-openai-index`)
- `PINECONE_NAMESPACE` (default: `nshmp-main`)
