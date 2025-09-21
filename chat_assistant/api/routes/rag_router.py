from fastapi import APIRouter, HTTPException

from chat_assistant.global_config import TEMPERATURE, K_SAMPLES
from chat_assistant.retrieval_search.vbd_manager import VBDManager
from chat_assistant.api.dto.source import Source
from chat_assistant.api.dto.query_request import QueryRequest
from chat_assistant.api.dto.query_response import QueryResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


vdb: VBDManager | None = None


@router.post("/create_db")
def create_db(sources: list[Source]):
    global vdb
    manager = VBDManager()
    db = manager.create_db([(s.source_type, s.path_or_url) for s in sources])
    vdb = manager
    return {
        "status": "created",
        "docs": len(db._collection.get()['ids'])
    }


@router.post("/add_document")
def add_document(source: Source):
    if not vdb:
        raise HTTPException(status_code=400, detail="DB not initialized")
    vdb.add_documents(source.source_type, source.path_or_url)
    return {"status": "document added"}


@router.delete("/delete_document/{doc_id}")
def delete_document(doc_id: str):
    if not vdb:
        raise HTTPException(status_code=400, detail="DB not initialized")
    vdb.delete_document(doc_id)
    return {"status": f"document {doc_id} deleted"}


@router.post("/update_document/{doc_id}")
def update_document(doc_id: str, source: Source):
    if not vdb:
        raise HTTPException(status_code=400, detail="DB not initialized")
    vdb.update_document(doc_id, source.source_type, source.path_or_url)
    return {"status": f"document {doc_id} updated"}


@router.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    if not vdb:
        raise HTTPException(status_code=400, detail="DB not initialized")
    result = vdb.search_with_llm_messages(
        query=request.query,
        k=request.k or K_SAMPLES,
        temperature=request.temperature or TEMPERATURE
    )
    return QueryResponse(**result)
