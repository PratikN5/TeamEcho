# api/routes/documents.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Response
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID
import os
import hashlib

from app.models.database import DocumentSource, DocumentStatus
from app.models.db import get_db
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.services.document_service import DocumentService
from app.models.schemas import DocumentCreate, DocumentResponse, StatusResponse
from app.services.processing_service import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.redis_service import RedisService
db_gen = get_db()
db = next(db_gen)
postgres_service = PostgresService(db)
s3_service = S3Service()
embedding_service = EmbeddingService(RedisService())
redis_service = RedisService()
document_service = DocumentService(postgres_service, s3_service)
processing_service = DocumentProcessingService(document_service , embedding_service, redis_service)

router = APIRouter(prefix="/documents", tags=["documents"])

# Service dependencies
def get_document_service(db: Session = Depends(get_db)):
    postgres_service = PostgresService(db)
    s3_service = S3Service()
    return DocumentService(postgres_service, s3_service)

@router.post("/upload", response_model=StatusResponse)
async def upload_document(
    file: UploadFile = File(...),
    source: Optional[str] = Form("upload"),
    uploaded_by: Optional[str] = Form("user"),
    document_service: DocumentService = Depends(get_document_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a document to the system
    1. Read file content
    2. Store in S3
    3. Create document record in PostgreSQL
    """
    try:
        # Read file content
        content = await file.read()
        
        # Calculate MD5 checksum
        checksum = hashlib.md5(content).hexdigest()
        
        # Create document
        document_create = DocumentCreate(
            filename=file.filename,
            original_filename=file.filename,
            mime_type=file.content_type,
            source=DocumentSource(source),
            uploaded_by=uploaded_by,
            file_size=len(content),
            checksum=checksum
        )
        
        # Store document
        document = await document_service.create_document(document_create, content)
        
        return StatusResponse(
            success=True,
            message="Document uploaded successfully",
            data={"document_id": str(document.id), "filename": document.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}", response_model=StatusResponse)
async def get_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get document metadata"""
    try:
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return StatusResponse(
            success=True,
            message="Document found",
            data={"document": DocumentResponse.from_orm(document)}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/download")
async def download_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service)
):
    """Download document content"""
    try:
        document, content = await document_service.get_document_with_content(document_id)
        if not document or not content:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Return file content as downloadable response
        return Response(
            content=content,
            media_type=document.mime_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{document.original_filename}"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/url", response_model=StatusResponse)
async def get_document_url(
    document_id: UUID,
    expiration: int = 3600,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get presigned URL for document download"""
    try:
        url = await document_service.get_document_download_url(document_id, expiration)
        if not url:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return StatusResponse(
            success=True,
            message=f"URL generated (expires in {expiration} seconds)",
            data={"url": url, "document_id": str(document_id), "expiration": expiration}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}", response_model=StatusResponse)
async def delete_document(
    document_id: UUID,
    permanent: bool = False,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Delete document
    If permanent=True, delete from S3 and database
    Otherwise, mark as deleted in database only
    """
    try:
        success = await document_service.delete_document(document_id, permanent)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return StatusResponse(
            success=True,
            message=f"Document {'permanently deleted' if permanent else 'marked as deleted'}",
            data={"document_id": str(document_id), "permanent": permanent}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=StatusResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    source: Optional[str] = None,
    status: Optional[str] = None,
    document_service: DocumentService = Depends(get_document_service)
):
    """List documents with optional filters"""
    try:
        # Convert string parameters to enums if needed
        source_enum = DocumentSource(source) if source else None
        status_enum = DocumentStatus(status) if status else None
        
        documents = await document_service.list_documents(
            limit=limit,
            offset=offset,
            source=source_enum,
            status=status_enum
        )
        
        return StatusResponse(
            success=True,
            message=f"Retrieved {len(documents)} documents",
            data={"documents": [DocumentResponse.from_orm(doc) for doc in documents]}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

