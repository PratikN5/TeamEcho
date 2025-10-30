# services/embedding_service.py
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime
import os
import json
import boto3
from botocore.config import Config

from app.core.config import get_settings

settings = get_settings()
from app.services.redis_service import RedisService

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, redis_service: RedisService):
        self.redis = redis_service
        self.embedding_model = self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on settings"""
        # Use AWS Bedrock for embeddings
        try:
            bedrock_config = Config(
                region_name=settings.AWS_REGION,
                retries={"max_attempts": 3, "mode": "standard"}
            )
            
            return boto3.client(
                service_name="bedrock-runtime",
                config=bedrock_config,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using AWS Bedrock"""
        try:
            # Use Claude 3 Haiku for embeddings
            response = self.embedding_model.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": text
                })
            )
            
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding")
            
            if not embedding:
                raise ValueError("No embedding returned from model")
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * settings.VECTOR_DIMENSION
    
    async def embed_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Embed entire document and store in Redis"""
        try:
            # Generate embedding
            embedding = await self.get_embedding(text)
            
            # Store in Redis
            success = await self.redis.store_document_embedding(
                doc_id=doc_id,
                embedding=embedding,
                metadata=metadata
            )
            
            return success
        except Exception as e:
            logger.error(f"Error embedding document {doc_id}: {str(e)}")
            return False
    
    async def embed_chunks(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Embed document chunks and store in Redis"""
        try:
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                # Generate chunk ID
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # Extract text and metadata
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                metadata.update({
                    "chunk_index": i,
                    "doc_id": doc_id
                })
                
                # Generate embedding
                embedding = await self.get_embedding(text)
                
                # Store in Redis
                await self.redis.store_chunk_embedding(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata
                )
            
            logger.info(f"Embedded {len(chunks)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Error embedding chunks for document {doc_id}: {str(e)}")
            return []
    
    async def search_similar_chunks(
        self,
        query: str,
        filter_str: str = "*",
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query"""
        try:
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
            
            # Search in Redis
            results = await self.redis.vector_search(
                query_embedding=query_embedding,
                index_name="chunk_vector_idx",
                filter_str=filter_str,
                k=k,
                min_score=min_score
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    async def search_similar_documents(
        self,
        query: str,
        filter_str: str = "*",
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents based on query"""
        try:
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
            
            # Search in Redis
            results = await self.redis.vector_search(
                query_embedding=query_embedding,
                index_name="doc_vector_idx",
                filter_str=filter_str,
                k=k,
                min_score=min_score
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
