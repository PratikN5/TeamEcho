from fastapi import FastAPI, Depends
from app.api.v1.endpoints import chat
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1.endpoints import documents, ingestion
from app.models.db import create_tables, drop_tables
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
import os
from app.api.v1.endpoints import documents, ingestion , test , diagnotics # Import your API route
from app.models.db import get_db, Base, engine
from app.services.document_service import DocumentService
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.services.embedding_service import EmbeddingService
from app.services.redis_service import RedisService
from app.services.github_service import GitHubService
from app.services.sharepoint_service import SharePointService
from app.agents.ingestion_agent import IngestionAgent
from app.services.processing_service import DocumentProcessingService  # You'll need to create this
from app.services.elasticsearch_service import ElasticsearchService
from app.core.config import get_settings
import logging
 
settings = get_settings()
 
 
logger = logging.getLogger("api")
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # """Run when the API starts"""
    # global ingestion_agent_task
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, create_tables)
    es_service = ElasticsearchService()
    await es_service.init()
 
    # # Start ingestion agent if enabled
    # if settings.INGESTION_AGENT_ENABLED:
    #     logger.info("Starting ingestion agent as background task")
    #     ingestion_agent_task = asyncio.create_task(start_ingestion_agent())
    # else:
    #     logger.info("Ingestion agent disabled, not starting")
 
    yield
     # Shutdown logic here (if needed)
    await es_service.close()
 
 
 
app = FastAPI(title="Genovate API" , lifespan=lifespan)
 
executor = ThreadPoolExecutor(max_workers=1)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# # Global variable to hold the ingestion agent task
# ingestion_agent_task = None
 
# async def start_ingestion_agent():
#     """Start and run the ingestion agent as a background process"""
#     try:
#         logger.info("Starting ingestion agent in background")
       
#         # Create database session
#         db = next(get_db())
       
#         try:
#             # Initialize services
#             postgres_service = PostgresService(db)
#             s3_service = S3Service()
#             document_service = DocumentService(postgres_service, s3_service)
#             redis_service = RedisService()
#             embedding_service = EmbeddingService(redis_service)
           
#             # Create processing service
#             processing_service = DocumentProcessingService(
#                 document_service=document_service,
#                 embedding_service=embedding_service,
#                 redis_service=redis_service
#             )
           
#             # Initialize optional services based on configuration
#             github_service = None
#             sharepoint_service = None
           
#             if settings.GITHUB_INTEGRATION_ENABLED:
#                 logger.info("GitHub integration enabled")
#                 github_service = GitHubService()
           
#             if settings.SHAREPOINT_INTEGRATION_ENABLED:
#                 logger.info("SharePoint integration enabled")
#                 sharepoint_service = SharePointService()
           
#             # Create ingestion agent
#             agent = IngestionAgent(
#                 document_service=document_service,
#                 processing_service=processing_service,
#                 github_service=github_service,
#                 sharepoint_service=sharepoint_service,
#                 s3_service=s3_service
#             )
           
#             # Start monitoring
#             logger.info(f"Starting monitoring loop with {settings.INGESTION_POLLING_INTERVAL}s interval")
#             await agent.start_monitoring()
       
#         finally:
#             # Close database session
#             db.close()
   
#     except Exception as e:
#         logger.error(f"Error in ingestion agent: {str(e)}", exc_info=True)
#         # Wait before restarting to avoid rapid restart loops
#         await asyncio.sleep(30)
#         # Restart the agent
#         asyncio.create_task(start_ingestion_agent())
 
 
 
# @app.on_event("shutdown")
# async def shutdown_event():
#     """Run when the API shuts down"""
#     global ingestion_agent_task
   
#     # Cancel the ingestion agent task if it's running
#     if ingestion_agent_task:
#         logger.info("Shutting down ingestion agent")
#         ingestion_agent_task.cancel()
#         try:
#             await ingestion_agent_task
#         except asyncio.CancelledError:
#             logger.info("Ingestion agent task cancelled")
 
# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}
 
 
# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["multi ingestion agent"])
app.include_router(test.router, prefix="/api/v1", tags=["test"])
app.include_router(diagnotics.router, prefix="/api/v1", tags=["diagnotics"])