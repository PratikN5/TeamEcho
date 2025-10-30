from fastapi import Depends
from app.services.redis_service import RedisService
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.models.db import get_db
from app.services.embedding_service import EmbeddingService

# Redis service singleton
_redis_service = None

def get_redis_service() -> RedisService:
    """
    Get or create a Redis service instance
    
    Returns:
        RedisService: Redis service instance
    """
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service



def get_postgres_service(db=Depends(get_db)):
    return PostgresService(db)

def get_s3_service():
    return S3Service()

def get_embedding_service(redis_service: RedisService = Depends(get_redis_service)):
    return EmbeddingService(redis_service)