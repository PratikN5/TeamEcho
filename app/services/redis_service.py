# services/redis_service.py
import redis
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime, timedelta

from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self):
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password= None,
            db=settings.REDIS_DB,
            decode_responses=True  # For regular operations
        )
        
        # Binary client for vector operations (no decode_responses)
        self.redis_binary = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        
        # Initialize indexes if they don't exist
        self._initialize_indexes()
    
    def _initialize_indexes(self):
        """Initialize Redis indexes for vector search and JSON data"""
        try:
            # Check if indexes already exist
            
            existing_indexes = self.redis_client.execute_command("FT._LIST")


            # Create document vector index if it doesn't exist
            if b'doc_vector_idx' not in existing_indexes:
                self._create_document_vector_index()
            
            # Create chunk vector index if it doesn't exist
            if b'chunk_vector_idx' not in existing_indexes:
                self._create_chunk_vector_index()
            
            logger.info("Redis indexes initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Redis indexes: {str(e)}")
    
    def _create_document_vector_index(self):
        """Create index for document-level embeddings"""
        try:
            # Define schema
            schema = (
                TextField("$.metadata.doc_id", as_name="doc_id"),
                TextField("$.metadata.filename", as_name="filename"),
                TextField("$.metadata.source", as_name="source"),
                NumericField("$.metadata.timestamp", as_name="timestamp"),
                TagField("$.metadata.status", as_name="status"),
                TagField("$.metadata.doc_type", as_name="doc_type"),
                VectorField("$.embedding", 
                           "HNSW", {
                               "TYPE": "FLOAT32",
                               "DIM": settings.VECTOR_DIMENSION,
                               "DISTANCE_METRIC": "COSINE"
                           }, as_name="embedding")
            )
            
            # Create index
            self.redis_client.ft("doc_vector_idx").create_index(
                schema,
                definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.JSON)
            )
            logger.info("Document vector index created")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                logger.info("Document vector index already exists")
            else:
                logger.error(f"Error creating document vector index: {str(e)}")
                raise
    
    def _create_chunk_vector_index(self):
        """Create index for text chunk embeddings"""
        try:
            # Define schema
            schema = (
                TextField("$.metadata.doc_id", as_name="doc_id"),
                TextField("$.metadata.chunk_id", as_name="chunk_id"),
                TextField("$.text", as_name="text"),
                NumericField("$.metadata.chunk_index", as_name="chunk_index"),
                TagField("$.metadata.source", as_name="source"),
                VectorField("$.embedding", 
                           "HNSW", {
                               "TYPE": "FLOAT32",
                               "DIM": settings.VECTOR_DIMENSION,
                               "DISTANCE_METRIC": "COSINE"
                           }, as_name="embedding")
            )
            
            # Create index
            self.redis_client.ft("chunk_vector_idx").create_index(
                schema,
                definition=IndexDefinition(prefix=["chunk:"], index_type=IndexType.JSON)
            )
            logger.info("Chunk vector index created")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                logger.info("Chunk vector index already exists")
            else:
                logger.error(f"Error creating chunk vector index: {str(e)}")
                raise

    # Vector Storage Methods
    async def store_document_embedding(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Store document-level embedding with metadata"""
        try:
            # Create Redis key
            redis_key = f"doc:{doc_id}"
            
            # Prepare data
            data = {
                "embedding": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", ""),
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "status": metadata.get("status", "active"),
                    "timestamp": metadata.get("timestamp", datetime.utcnow().timestamp()),
                    **{k: v for k, v in metadata.items() if k not in ["filename", "source", "doc_type", "status", "timestamp"]}
                }
            }
            
            # Store as JSON
            self.redis_client.json().set(redis_key, "$", data)
            
            # Set TTL if specified
            if "ttl" in metadata and isinstance(metadata["ttl"], int):
                self.redis_client.expire(redis_key, metadata["ttl"])
            
            logger.info(f"Stored document embedding for {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing document embedding for {doc_id}: {str(e)}")
            return False
    
    async def store_chunk_embedding(
        self,
        doc_id: str,
        chunk_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Store text chunk embedding with metadata"""
        try:
            # Create Redis key
            redis_key = f"chunk:{doc_id}:{chunk_id}"
            
            # Prepare data
            data = {
                "text": text,
                "embedding": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "source": metadata.get("source", ""),
                    **{k: v for k, v in metadata.items() if k not in ["chunk_index", "source"]}
                }
            }
            
            # Store as JSON
            self.redis_client.json().set(redis_key, "$", data)
            
            # Set TTL if specified
            if "ttl" in metadata and isinstance(metadata["ttl"], int):
                self.redis_client.expire(redis_key, metadata["ttl"])
            
            logger.info(f"Stored chunk embedding for {doc_id}:{chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing chunk embedding for {doc_id}:{chunk_id}: {str(e)}")
            return False
    
    async def vector_search(
        self,
        query_embedding: List[float],
        index_name: str = "chunk_vector_idx",
        return_fields: List[str] = ["text", "metadata", "vector_score"],
        filter_str: str = "*",
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            query_embedding: Vector to search for
            index_name: Redis index name (chunk_vector_idx or doc_vector_idx)
            return_fields: Fields to return in results
            filter_str: Filter expression (e.g., "@source:{pdf}")
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Convert embedding to bytes for Redis
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build query
            q = (
                Query(f"{filter_str}=>[KNN {k} @embedding $query_vector AS vector_score]")
                .sort_by("vector_score", asc=False)
                .paging(0, k)
                .return_fields(*return_fields)
                .dialect(2)
            )
            
            # Execute search
            results = self.redis_client.ft(index_name).search(
                q, {"query_vector": query_vector}
            )
            
            # Process results
            processed_results = []
            for doc in results.docs:
                # Extract JSON data
                doc_data = json.loads(doc.json) if hasattr(doc, 'json') else {}
                
                # Extract score
                score = float(doc.vector_score) if hasattr(doc, 'vector_score') else 0.0
                
                # Skip if below minimum score
                if score < min_score:
                    continue
                
                # Add to results
                processed_results.append({
                    "id": doc.id,
                    "score": score,
                    "text": doc_data.get("text", ""),
                    "metadata": doc_data.get("metadata", {})
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    async def delete_document_vectors(self, doc_id: str) -> int:
        """Delete all vectors for a document (document and chunks)"""
        try:
            # Delete document vector
            doc_key = f"doc:{doc_id}"
            doc_deleted = self.redis_client.delete(doc_key)
            
            # Delete all chunk vectors
            chunk_pattern = f"chunk:{doc_id}:*"
            chunk_keys = self.redis_client.keys(chunk_pattern)
            chunks_deleted = 0
            if chunk_keys:
                chunks_deleted = self.redis_client.delete(*chunk_keys)
            
            total_deleted = doc_deleted + chunks_deleted
            logger.info(f"Deleted {total_deleted} vectors for document {doc_id}")
            return total_deleted
        except Exception as e:
            logger.error(f"Error deleting vectors for document {doc_id}: {str(e)}")
            return 0
    
    # Chat Memory Methods
    async def store_chat_memory(
        self,
        session_id: str,
        message_id: str,
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store chat message in Redis"""
        try:
            # Create Redis key
            redis_key = f"chat:{session_id}:{message_id}"
            
            # Prepare data
            timestamp = datetime.utcnow().timestamp()
            data = {
                "session_id": session_id,
                "message_id": message_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            
            # Store as JSON
            self.redis_client.json().set(redis_key, "$", data)
            
            # Set TTL
            self.redis_client.expire(redis_key, settings.CHAT_MEMORY_TTL)
            
            # Update session last activity
            session_key = f"chat_session:{session_id}"
            session_data = {
                "session_id": session_id,
                "last_activity": timestamp,
                "message_count": self.redis_client.json().get(session_key, "$.message_count") + 1 if self.redis_client.exists(session_key) else 1
            }
            self.redis_client.json().set(session_key, "$", session_data)
            self.redis_client.expire(session_key, settings.CHAT_MEMORY_TTL)
            
            # Add to session message list
            list_key = f"chat_messages:{session_id}"
            self.redis_client.lpush(list_key, message_id)
            self.redis_client.expire(list_key, settings.CHAT_MEMORY_TTL)
            
            # Trim list if too long
            if self.redis_client.llen(list_key) > settings.MAX_CHAT_HISTORY:
                self.redis_client.ltrim(list_key, 0, settings.MAX_CHAT_HISTORY - 1)
            
            logger.info(f"Stored chat message {message_id} for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing chat message for session {session_id}: {str(e)}")
            return False
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            # Get message IDs from list
            list_key = f"chat_messages:{session_id}"
            message_ids = self.redis_client.lrange(list_key, 0, limit - 1)
            
            if not message_ids:
                return []
            
            # Get messages
            messages = []
            for message_id in message_ids:
                redis_key = f"chat:{session_id}:{message_id}"
                message_data = self.redis_client.json().get(redis_key)
                if message_data:
                    messages.append(message_data)
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get("timestamp", 0))
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history for session {session_id}: {str(e)}")
            return []
    
    async def create_chat_session(self, user_id: str) -> str:
        """Create a new chat session"""
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create session
            session_key = f"chat_session:{session_id}"
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow().timestamp(),
                "last_activity": datetime.utcnow().timestamp(),
                "message_count": 0
            }
            
            # Store session
            self.redis_client.json().set(session_key, "$", session_data)
            self.redis_client.expire(session_key, settings.CHAT_MEMORY_TTL)
            
            # Create message list
            list_key = f"chat_messages:{session_id}"
            self.redis_client.expire(list_key, settings.CHAT_MEMORY_TTL)
            
            logger.info(f"Created chat session {session_id} for user {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating chat session for user {user_id}: {str(e)}")
            raise
    
    async def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session details"""
        try:
            session_key = f"chat_session:{session_id}"
            session_data = self.redis_client.json().get(session_key)
            return session_data
        except Exception as e:
            logger.error(f"Error getting chat session {session_id}: {str(e)}")
            return None
    
    # Caching Methods
    async def set_cache(
        self,
        key: str,
        value: Union[str, Dict, List],
        expire_seconds: int = 3600
    ) -> bool:
        """Set cache value with expiration"""
        try:
            cache_key = f"cache:{key}"
            
            # Store value based on type
            if isinstance(value, (dict, list)):
                self.redis_client.json().set(cache_key, "$", value)
            else:
                self.redis_client.set(cache_key, str(value))
            
            # Set expiration
            if expire_seconds > 0:
                self.redis_client.expire(cache_key, expire_seconds)
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {str(e)}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Union[str, Dict, List]]:
        """Get cached value"""
        try:
            cache_key = f"cache:{key}"
            
            # Check if JSON
            if self.redis_client.type(cache_key) == "ReJSON-RL":
                return self.redis_client.json().get(cache_key)
            else:
                return self.redis_client.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {str(e)}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cached value"""
        try:
            cache_key = f"cache:{key}"
            deleted = self.redis_client.delete(cache_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {str(e)}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            cache_key = f"cache:{key}"
            return bool(self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Error checking cache for key {key}: {str(e)}")
            return False
    
    # Helper Methods
    def ping(self) -> bool:
        """Check Redis connection"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection error: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
                "total_keys": sum(self.redis_client.dbsize() for db in range(16)),
                "vector_keys": len(self.redis_client.keys("doc:*")) + len(self.redis_client.keys("chunk:*")),
                "chat_sessions": len(self.redis_client.keys("chat_session:*")),
                "cache_keys": len(self.redis_client.keys("cache:*"))
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {str(e)}")
            return {"error": str(e)}

    async def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in Redis"""
        try:
            exists = self.redis_client.exists(f"doc:{doc_id}")
            return bool(exists)
        except Exception as e:
            logger.error(f"Error checking if document {doc_id} exists: {str(e)}")
            return False

    
    async def get_document_embedding(self, doc_id: str) -> Optional[List[float]]:
      """Get document embedding from Redis"""
      try:
          embedding_data = self.redis_client.json().get(f"doc:{doc_id}", "$.embedding")
          if embedding_data:
              # RedisJSON returns list inside a list [[...]]
              return embedding_data[0] if isinstance(embedding_data[0], list) else embedding_data
          return None
      except Exception as e:
          logger.error(f"Error getting embedding for document {doc_id}: {str(e)}")
          return None

    
    async def get_chunk_ids_for_document(self, doc_id: str) -> List[str]:
      """Get all chunk IDs for a document"""
      try:
          chunk_keys = self.redis_client.keys(f"chunk:{doc_id}:*")
          return [key.split(":")[-1] for key in chunk_keys]
      except Exception as e:
          logger.error(f"Error getting chunk IDs for document {doc_id}: {str(e)}")
          return []

    
    async def get_chunk_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """
        Get chunk embedding from Redis
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk embedding as list of floats, or None if not found
        """
        try:
            embedding_data = self.redis_client.json().get(f"chunk:{chunk_id}", "$.embedding")
            if embedding_data:
                return embedding_data[0] if isinstance(embedding_data[0], list) else embedding_data
            return None
        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk_id}: {str(e)}")
            return None