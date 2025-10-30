# routes/admin.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List

from app.api.dependencies import get_redis_service
from app.services.redis_service import RedisService
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()

@router.get("/diagnostics/embeddings/{doc_id}")
async def check_embeddings(
    doc_id: str,
    redis_service: RedisService = Depends(get_redis_service)
):
    """Check embedding status for a document"""
    # Check if document exists in Redis
    doc_exists = await redis_service.document_exists(doc_id)
    if not doc_exists:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in Redis")
    
    # Get document embedding
    doc_embedding = await redis_service.get_document_embedding(doc_id)
    
    # Get chunk IDs
    chunk_ids = await redis_service.get_chunk_ids_for_document(doc_id)
    
    # Check chunk embeddings
    chunk_results = []
    for chunk_id in chunk_ids:
        embedding = await redis_service.get_chunk_embedding(chunk_id)
        chunk_results.append({
            "chunk_id": chunk_id,
            "has_embedding": embedding is not None,
            "embedding_dimension": len(embedding) if embedding else 0
        })
    
    return {
        "doc_id": doc_id,
        "has_document_embedding": doc_embedding is not None,
        "document_embedding_dimension": len(doc_embedding) if doc_embedding else 0,
        "chunk_count": len(chunk_ids),
        "chunks_with_embeddings": sum(1 for c in chunk_results if c["has_embedding"]),
        "chunk_details": chunk_results
    }
