from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import json
import os
from datetime import datetime
from app.models.schemas import DocumentCreate

from app.core.config import get_settings
from app.services.redis_service import RedisService
from app.services.document_service import DocumentService
from app.services.elasticsearch_service import ElasticsearchService
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.services.embedding_service import EmbeddingService
from app.services.processing_service import DocumentProcessingService
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.api.dependencies import get_redis_service, get_postgres_service, get_s3_service , get_embedding_service

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_redis_service():
    return RedisService()

def get_document_service(
    postgres_service: PostgresService = Depends(get_postgres_service),
    s3_service: S3Service = Depends(get_s3_service)
):
    return DocumentService(postgres_service, s3_service)

def get_elasticsearch_service():
    return ElasticsearchService()

def get_knowledge_graph_agent():
    return KnowledgeGraphAgent()

def get_embedding_service(
    redis_service: RedisService = Depends(get_redis_service)
):
    return EmbeddingService(redis_service)


def get_processing_service(
    document_service=Depends(get_document_service),
    embedding_service=Depends(get_embedding_service),
    redis_service=Depends(get_redis_service)
):
    return DocumentProcessingService(
        document_service=document_service,
        embedding_service=embedding_service,
        redis_service=redis_service
    )

@router.post("/diagnostics/process-and-analyze", response_model=Dict[str, Any])
async def process_and_analyze_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    processing_service: DocumentProcessingService = Depends(get_processing_service),
    elasticsearch_service: ElasticsearchService = Depends(get_elasticsearch_service),
    knowledge_graph_agent: KnowledgeGraphAgent = Depends(get_knowledge_graph_agent)
):
    """
    Process a document and analyze it with the full pipeline:
    1. Ingest document
    2. Process with OCR
    3. Generate embeddings
    4. Extract entities for knowledge graph
    5. Index code if it's a code file
    
    Returns diagnostic information about each step
    """
    try:
        # Step 1: Save the document
        content = await file.read()
        filename = file.filename
        content_type = file.content_type or "application/octet-stream"
        
        

        document_create = DocumentCreate(
            filename=filename,                   # required
            original_filename=filename,          # required (or however you get the original filename)
            source="upload",        # required, must match your enum!
            uploaded_by="diagnostic_user",       # optional, if you want
            file_size=len(content),              # optional
            mime_type=content_type,              # optional
            # ... add more as needed ...
        )

        document = await document_service.create_document(
            document_create,
            content
        )

        
        if not document:
            raise HTTPException(status_code=500, detail="Failed to create document")
        
        doc_id = document.id
        
        
        # Step 2: Process the document
        process_success = await processing_service.process_document(doc_id)
        
        # Get processed document with extracted text
        processed_doc = await document_service.get_document(doc_id)
        print("Processed Doc:", processed_doc)
        
        # Step 3: Check if it's a code file and index it
        is_code = False
        code_analysis = {}
        if content_type in ["text/x-python", "text/javascript", "application/json", "text/plain"]:
            # Try to detect if it's code
            if any(filename.endswith(ext) for ext in [".py", ".js", ".java", ".c", ".cpp", ".cs", ".go", ".ts"]):
                is_code = True
                
                # Index code file
                await elasticsearch_service.index_code_file(
                    file_path=filename,
                    content=processed_doc.text,
                    metadata={
                        "doc_id": doc_id,
                        "last_modified": datetime.utcnow().isoformat()
                    }
                )
                
                # Analyze code structure
                code_analysis = await elasticsearch_service.analyze_code_structure(filename)
        
        # Step 4: Extract entities and build knowledge graph
        entities = []
        relationships = []
        graph_built = False
        
        if processed_doc:
            # Extract entities
            entities = await knowledge_graph_agent.extract_entities(
                processed_doc,
                doc_id
            )
            
            # Extract relationships
            if entities:
                relationships = await knowledge_graph_agent.extract_relationships(
                    entities,
                    processed_doc
                )
            
            # Build knowledge graph
            if entities or relationships:
                graph_built = await knowledge_graph_agent.build_graph(
                    doc_id=doc_id,
                    title=filename,
                    entities=entities,
                    relationships=relationships,
                    metadata={
                        "source": "diagnostic_endpoint",
                        "content_type": content_type,
                        "is_code": is_code
                    }
                )
                
                # Link code file if applicable
                if is_code:
                    await knowledge_graph_agent.link_code_to_document(
                        doc_id=doc_id,
                        code_path=filename
                    )
        
        # Return diagnostic information
        return {
            "document": {
                "id": doc_id,
                "filename": filename,
                "content_type": content_type,
                "size": len(content),
                "status": processed_doc.status,
                "text_extracted": bool(processed_doc.text),
                "text_length": len(processed_doc.text) if processed_doc.text else 0
            },
            "processing": {
                "success": process_success,
                "metadata": processed_doc.metadata
            },
            "knowledge_graph": {
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "graph_built": graph_built,
                "sample_entities": entities[:5] if entities else []
            },
            "code_analysis": code_analysis if is_code else {"is_code": False}
        }
    
    except Exception as e:
        logger.error(f"Error in process_and_analyze_document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/diagnostics/knowledge-graph/{doc_id}", response_model=Dict[str, Any])
async def get_document_knowledge_graph(
    doc_id: str = Path(..., description="Document ID"),
    knowledge_graph_agent: KnowledgeGraphAgent = Depends(get_knowledge_graph_agent)
):
    """Get knowledge graph for a document"""
    try:
        graph_data = await knowledge_graph_agent.get_document_graph(doc_id)
        
        if not graph_data["nodes"]:
            raise HTTPException(status_code=404, detail=f"No knowledge graph found for document {doc_id}")
        
        return graph_data
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting knowledge graph: {str(e)}")

@router.get("/diagnostics/code-search", response_model=List[Dict[str, Any]])
async def search_code(
    query: str = Query(..., description="Search query"),
    language: Optional[str] = Query(None, description="Filter by programming language"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    limit: int = Query(10, description="Maximum number of results"),
    elasticsearch_service: ElasticsearchService = Depends(get_elasticsearch_service)
):
    """Search for code using Elasticsearch"""
    try:
        results = await elasticsearch_service.search_code(
            query=query,
            language=language,
            file_path=file_path,
            limit=limit
        )
        
        return results
    except Exception as e:
        logger.error(f"Error searching code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching code: {str(e)}")

@router.get("/diagnostics/flow-check", response_model=Dict[str, Any])
async def check_processing_flow(
    redis_service: RedisService = Depends(get_redis_service),
    document_service: DocumentService = Depends(get_document_service),
    elasticsearch_service: ElasticsearchService = Depends(get_elasticsearch_service),
    knowledge_graph_agent: KnowledgeGraphAgent = Depends(get_knowledge_graph_agent)
):
    """
    Check if all components of the processing flow are working correctly
    """
    results = {
        "status": "ok",
        "components": {},
        "message": "All components are operational"
    }
    
    # Check Redis connection
    try:
        redis_ping = redis_service.ping()
        results["components"]["redis"] = {
            "status": "ok" if redis_ping else "error",
            "message": "Connected" if redis_ping else "Failed to connect"
        }
        if not redis_ping:
            results["status"] = "error"
    except Exception as e:
        results["components"]["redis"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        results["status"] = "error"
    
    # Check Document Service
    try:
        # Just check if we can get a count of documents
        doc_count = await document_service.get_document_count()
        results["components"]["document_service"] = {
            "status": "ok",
            "message": f"Connected, {doc_count} documents in database"
        }
    except Exception as e:
        results["components"]["document_service"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        results["status"] = "error"
    
    # Check Elasticsearch
    try:
        es_connected = elasticsearch_service.es is not None
        results["components"]["elasticsearch"] = {
            "status": "ok" if es_connected else "error",
            "message": "Connected" if es_connected else "Failed to connect"
        }
        if not es_connected:
            results["status"] = "error"
    except Exception as e:
        results["components"]["elasticsearch"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        results["status"] = "error"
    
    # Check Neo4j connection
    try:
        neo4j_connected = knowledge_graph_agent.driver is not None
        results["components"]["neo4j"] = {
            "status": "ok" if neo4j_connected else "error",
            "message": "Connected" if neo4j_connected else "Failed to connect"
        }
        if not neo4j_connected:
            results["status"] = "error"
    except Exception as e:
        results["components"]["neo4j"] = {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        results["status"] = "error"
    
    # Update overall status message
    if results["status"] == "error":
        results["message"] = "One or more components are not operational"
    
    return results
