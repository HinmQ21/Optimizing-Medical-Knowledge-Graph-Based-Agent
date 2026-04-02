"""Step 6B: FastAPI server for medical knowledge retrieval (optional).

Usage:
    cd baseline/
    ./stage2_venv312/bin/python -m scripts.serve.retrieval_api
    # or with uvicorn directly:
    uvicorn scripts.serve.retrieval_api:app --host 0.0.0.0 --port 8001

Test:
    curl -X POST http://localhost:8001/search \
        -H "Content-Type: application/json" \
        -d '{"queries": ["What drugs treat diabetes?"], "top_k": 5}'
"""

from fastapi import FastAPI
from pydantic import BaseModel

from .retrieval_tool import MedicalKnowledgeTool

app = FastAPI(title="Medical Knowledge Retrieval API")


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int = 5


class SearchResult(BaseModel):
    query: str
    results: list[str]


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    tool = MedicalKnowledgeTool.load()
    return [
        SearchResult(query=q, results=tool.retrieve(q, req.top_k))
        for q in req.queries
    ]


@app.get("/health")
def health():
    tool = MedicalKnowledgeTool.load()
    return {
        "status": "ok",
        "hyperedges": tool.idx_he.ntotal,
        "entities": tool.idx_ent.ntotal,
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8001)
