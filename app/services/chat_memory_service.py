# services/chat_memory_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

from services.redis_service import RedisService
from services.postgres_service import PostgresService
from models.schemas import ChatSessionCreate, ChatMessageCreate
from config.settings import settings

logger = logging.getLogger(__name__)

class ChatMemoryService:
    def __init__(self, redis_service: RedisService, postgres_service: PostgresService = None):
        self.redis = redis_service
        self.postgres = postgres_service  # Optional, for persistent storage
    
    async def create_session(self, user_id: str) -> str:
        """Create a new chat session"""
        try:
            # Create in Redis
            session_id = await self.redis.create_chat_session(user_id)
            
            # Store in PostgreSQL if available (for persistence)
            if self.postgres:
                session_create = ChatSessionCreate(
                    session_id=session_id,
                    user_id=user_id
                )
                self.postgres.create_chat_session(session_create)
            
            return session_id
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            raise
    
    async def add_message(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to the chat history"""
        try:
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Prepare metadata
            full_metadata = {
                "timestamp": datetime.utcnow().timestamp(),
                "sources": sources or []
            }
            if metadata:
                full_metadata.update(metadata)
            
            # Store in Redis
            await self.redis.store_chat_memory(
                session_id=session_id,
                message_id=message_id,
                user_message=user_message,
                ai_response=ai_response,
                metadata=full_metadata
            )
            
            # Store in PostgreSQL if available
            if self.postgres:
                message_create = ChatMessageCreate(
                    session_id=session_id,
                    message_id=message_id,
                    user_message=user_message,
                    ai_response=ai_response,
                    sources=sources,
                    metadata=metadata
                )
                self.postgres.create_chat_message(message_create)
            
            return message_id
        except Exception as e:
            logger.error(f"Error adding chat message: {str(e)}")
            raise
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = settings.MAX_CHAT_HISTORY
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            # Try Redis first
            history = await self.redis.get_chat_history(session_id, limit)
            
            # If empty and PostgreSQL is available, try PostgreSQL
            if not history and self.postgres:
                pg_history = self.postgres.get_chat_history(session_id, limit)
                
                # Convert PostgreSQL history to Redis format
                history = [
                    {
                        "session_id": msg.session_id,
                        "message_id": msg.message_id,
                        "user_message": msg.user_message,
                        "ai_response": msg.ai_response,
                        "timestamp": msg.timestamp.timestamp(),
                        "metadata": {
                            "sources": msg.sources or []
                        }
                    }
                    for msg in pg_history
                ]
                
                # Restore to Redis if found in PostgreSQL
                for msg in history:
                    await self.redis.store_chat_memory(
                        session_id=msg["session_id"],
                        message_id=msg["message_id"],
                        user_message=msg["user_message"],
                        ai_response=msg["ai_response"],
                        metadata=msg["metadata"]
                    )
            
            return history
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 10,
        max_tokens: int = 2000
    ) -> str:
        """
        Get conversation context formatted for LLM context window
        Limited by both message count and approximate token count
        """
        try:
            history = await self.get_chat_history(session_id, max_messages)
            
            # Format conversation
            formatted_history = []
            token_count = 0
            
            for msg in history:
                user_msg = f"Human: {msg['user_message']}"
                ai_msg = f"AI: {msg['ai_response']}"
                
                # Approximate token count (1 token ≈ 4 chars)
                user_tokens = len(user_msg) // 4
                ai_tokens = len(ai_msg) // 4
                
                # Check if adding these messages would exceed token limit
                if token_count + user_tokens + ai_tokens > max_tokens:
                    break
                
                formatted_history.append(user_msg)
                formatted_history.append(ai_msg)
                token_count += user_tokens + ai_tokens
            
            return "\n\n".join(formatted_history)
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return ""
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        try:
            # Try Redis first
            session = await self.redis.get_chat_session(session_id)
            
            # If not found and PostgreSQL is available, try PostgreSQL
            if not session and self.postgres:
                pg_session = self.postgres.get_chat_session(session_id)
                if pg_session:
                    session = {
                        "session_id": pg_session.session_id,
                        "user_id": pg_session.user_id,
                        "created_at": pg_session.created_date.timestamp(),
                        "last_activity": pg_session.last_activity.timestamp(),
                        "message_count": len(self.postgres.get_chat_history(session_id))
                    }
            
            return session
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}")
            return None
