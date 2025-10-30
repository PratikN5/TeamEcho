# scripts/run_ingestion_agent.py
import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.document_service import DocumentService
from services.postgres_service import PostgresService
from services.s3_service import S3Service
from services.processing_service import DocumentProcessingService
from services.document_chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.redis_service import RedisService
from services.github_service import GitHubService
from services.sharepoint_service import SharePointService
from agents.ingestion_agent import IngestionAgent
from app.models.db import get_db
db_gen = get_db()

from app.core.config import get_settings

settings = get_settings()


logger = logging.getLogger("ingestion-agent")

async def main():
    """Run the ingestion agent"""
    try:
        logger.info("Starting ingestion agent")
        print("inside ingestion agent")
        
        # Create database session
        db = next(db_gen) 
        
        try:
            # Initialize services
            postgres_service = PostgresService(db)
            s3_service = S3Service()
            document_service = DocumentService(postgres_service, s3_service)
            redis_service = RedisService()
            embedding_service = EmbeddingService(redis_service)
            chunking_service = ChunkingService()
            processing_service = DocumentProcessingService(
                document_service=document_service,
                chunking_service=chunking_service,
                embedding_service=embedding_service,
                redis_service=redis_service
            )
            
            # Initialize optional services based on configuration
            github_service = None
            sharepoint_service = None
            
            if settings.GITHUB_INTEGRATION_ENABLED:
                logger.info("GitHub integration enabled")
                github_service = GitHubService()
            
            if settings.SHAREPOINT_INTEGRATION_ENABLED:
                logger.info("SharePoint integration enabled")
                sharepoint_service = SharePointService()
            
            # Create ingestion agent
            agent = IngestionAgent(
                document_service=document_service,
                processing_service=processing_service,
                github_service=github_service,
                sharepoint_service=sharepoint_service,
                s3_service=s3_service
            )
            
            # Start monitoring
            logger.info(f"Starting monitoring loop with {settings.INGESTION_POLLING_INTERVAL}s interval")
            await agent.start_monitoring()
        
        finally:
            # Close database session
            db.close()
    
    except Exception as e:
        logger.error(f"Error in ingestion agent: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
