# api/routes/ingestion.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Header , File , UploadFile
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import hmac
import hashlib
import json
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import get_db
from app.services.document_service import DocumentService
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.services.ocr_service import OCRService
from app.services.processing_service import DocumentProcessingService
from app.services.github_service import GitHubService
from app.services.sharepoint_service import SharePointService
from app.agents.ingestion_agent import IngestionAgent
from app.models.schemas import StatusResponse
from app.services.document_chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.redis_service import RedisService
from app.core.config import get_settings

settings = get_settings()

router = APIRouter()

# Service dependencies
def get_ingestion_agent(db: Session = Depends(get_db)):
    postgres_service = PostgresService(db)
    s3_service = S3Service()
    document_service = DocumentService(postgres_service, s3_service)
    processing_service = DocumentProcessingService(
        document_service=document_service,
        chunking_service=ChunkingService(),
        embedding_service=EmbeddingService(RedisService()),
        redis_service=RedisService()
    )
    
    # Initialize optional services based on configuration
    github_service = GitHubService() if settings.GITHUB_INTEGRATION_ENABLED else None
    sharepoint_service = SharePointService() if settings.SHAREPOINT_INTEGRATION_ENABLED else None
    
    return IngestionAgent(
        document_service=document_service,
        processing_service=processing_service,
        github_service=github_service,
        sharepoint_service=sharepoint_service,
        s3_service=s3_service
    )

@router.post("/start", response_model=StatusResponse)
async def start_ingestion_agent(
    background_tasks: BackgroundTasks,
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """Start the ingestion agent monitoring loop in the background"""
    try:
        # Start monitoring in background
        background_tasks.add_task(ingestion_agent.start_monitoring)
        
        return StatusResponse(
            success=True,
            message="Ingestion agent started in background",
            data={"status": "started"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/github", response_model=StatusResponse)
async def github_webhook(
    request: Request,
    x_hub_signature: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Webhook endpoint for GitHub events
    Verifies signature and processes the event
    """
    try:
        # Verify GitHub webhook signature if configured
        if settings.GITHUB_WEBHOOK_SECRET:
            # Read raw body
            body = await request.body()
            
            # Verify signature
            if not x_hub_signature or not x_hub_signature.startswith("sha1="):
                raise HTTPException(status_code=401, detail="Invalid signature format")
            
            signature = x_hub_signature.replace("sha1=", "")
            
            # Calculate expected signature
            mac = hmac.new(
                settings.GITHUB_WEBHOOK_SECRET.encode(),
                msg=body,
                digestmod=hashlib.sha1
            )
            expected_signature = mac.hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse payload
        payload = await request.json()
        
        # Add event type from header
        payload["event_type"] = x_github_event
        
        # Process webhook
        result = await ingestion_agent.process_github_webhook(payload)
        
        return StatusResponse(
            success=True,
            message="GitHub webhook processed",
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sharepoint", response_model=StatusResponse)
async def sharepoint_webhook(
    request: Request,
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Webhook endpoint for SharePoint events
    """
    try:
        # Parse payload
        payload = await request.json()
        
        # Process webhook
        result = await ingestion_agent.process_sharepoint_webhook(payload)
        
        return StatusResponse(
            success=True,
            message="SharePoint webhook processed",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/s3-event", response_model=StatusResponse)
async def s3_event(
    event: Dict[str, Any],
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Process an S3 event notification
    This endpoint can be called by AWS Lambda when new files are added to S3
    """
    try:
        # Process S3 event
        result = await ingestion_agent.process_s3_event(event)
        
        return StatusResponse(
            success=True,
            message="S3 event processed",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/github", response_model=StatusResponse)
async def sync_github(
    background_tasks: BackgroundTasks,
    run_in_background: bool = True,
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Manually trigger GitHub synchronization
    """
    try:
        if run_in_background:
            # Run in background
            background_tasks.add_task(ingestion_agent.fetch_from_github)
            
            return StatusResponse(
                success=True,
                message="GitHub sync started in background",
                data={"status": "started"}
            )
        else:
            # Run immediately
            await ingestion_agent.fetch_from_github()
            
            return StatusResponse(
                success=True,
                message="GitHub sync completed",
                data={"status": "completed"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/sharepoint", response_model=StatusResponse)
async def sync_sharepoint(
    background_tasks: BackgroundTasks,
    run_in_background: bool = True,
    ingestion_agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Manually trigger SharePoint synchronization
    """
    try:
        if run_in_background:
            # Run in background
            background_tasks.add_task(ingestion_agent.fetch_from_sharepoint)
            
            return StatusResponse(
                success=True,
                message="SharePoint sync started in background",
                data={"status": "started"}
            )
        else:
            # Run immediately
            await ingestion_agent.fetch_from_sharepoint()
            
            return StatusResponse(
                success=True,
                message="SharePoint sync completed",
                data={"status": "completed"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/test-ocr")
async def test_ocr(
    file: UploadFile = File(...),
    use_gpu: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Test endpoint for OCR service"""
    try:
        # Initialize OCR service
        ocr_service = OCRService(use_gpu=use_gpu)
        
        # Read file content
        content = await file.read()
        
        # Process with OCR
        text, metadata = ocr_service.extract_text_from_file(content, file.filename)
        
        return {
            "filename": file.filename,
            "text_length": len(text),
            "text_sample": text,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

