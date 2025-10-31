from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, Any, Optional, List
import uuid
import logging
from app.services.bedrock_service import BedrockService
from app.services.redis_service import RedisService
from app.services.embedding_service import EmbeddingService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.qa_retrieval_agent import QARetrievalAgent
from uuid import uuid4
router = APIRouter()
logger = logging.getLogger(__name__)
# Dependency injection
def get_bedrock_service():
    return BedrockService()
def get_redis_service():
    return RedisService()
def get_elasticsearch_service():
    return ElasticsearchService()
def get_knowledge_graph_service():
    return KnowledgeGraphService()
def get_qa_agent(
    bedrock_service: BedrockService = Depends(get_bedrock_service),
    redis_service: RedisService = Depends(get_redis_service),
    elasticsearch_service: ElasticsearchService = Depends(get_elasticsearch_service),
    knowledge_graph_service: KnowledgeGraphService = Depends(get_knowledge_graph_service)
):
    return QARetrievalAgent(
        bedrock_service=bedrock_service,
        redis_service=redis_service,
        elasticsearch_service=elasticsearch_service,
        embedding_service=EmbeddingService(),
        knowledge_graph_service=knowledge_graph_service
    )
@router.post("/chat", response_model=Dict[str, Any])
async def chat_with_context(
    query: str = Body(..., embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    user_id: Optional[str] = Body("anonymous", embed=True),
    include_sources: bool = Body(True, embed=True),
    include_follow_ups: bool = Body(True, embed=True),
    qa_agent: QARetrievalAgent = Depends(get_qa_agent)
):
    """
    Chat with contextual retrieval and reasoning
    This endpoint processes a user query through the QA Retrieval Agent, which:
    1. Analyzes the query to determine the best retrieval strategy
    2. Retrieves relevant context from vector DB, knowledge graph, and code index
    3. Generates a comprehensive answer with reasoning
    4. Provides source citations and follow-up questions
    """
    try:
        # ✅ Auto-generate session_id if null
        if not session_id:
            session_id = str(uuid4())
        # Process the query through the QA agent
        response = await qa_agent.process_query(
            query=query,
            session_id=session_id,
            user_id=user_id,
            include_sources=include_sources,
            follow_up_suggestions=include_follow_ups
        )
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/chat/simple", response_model=Dict[str, str])
async def chat_with_claude(
    message: str = Body(..., embed=True),
    bedrock_service: BedrockService = Depends(get_bedrock_service)
):
    """Simple chat endpoint that directly calls the LLM without context retrieval"""
    try:
        response = await bedrock_service.get_completion(message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/chat/sessions/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_chat_history(
    session_id: str,
    limit: int = Query(10, ge=1, le=50),
    redis_service: RedisService = Depends(get_redis_service)
):
    """Get chat history for a session"""
    try:
        history = await redis_service.get_chat_history(session_id, limit)
        return history
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/chat/sessions", response_model=Dict[str, str])
async def create_chat_session(
    user_id: str = Body("anonymous", embed=True),
    redis_service: RedisService = Depends(get_redis_service)
):
    """Create a new chat session"""
    try:
        session_id = await redis_service.create_chat_session(user_id)
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
   
 
 
@router.post("/debug/qa")
async def debug_qa(request: Dict[str, Any]):
       query = request.get("query", "What is a knowledge graph?")
       
       # Initialize services
       bedrock_service = BedrockService()
       redis_service = RedisService()
       elasticsearch_service = ElasticsearchService()
       kg_agent = KnowledgeGraphAgent()
       kg_service = KnowledgeGraphService(kg_agent)
       
       # Create QA agent
       qa_agent = QARetrievalAgent(
           bedrock_service=bedrock_service,
           redis_service=redis_service,
           embedding_service=EmbeddingService(),
           elasticsearch_service=elasticsearch_service,
           knowledge_graph_service=kg_service,
           llm_service=bedrock_service
       )
       
       # Test components
       vector_test = await qa_agent.test_vector_search(query)
       kg_test = await qa_agent.test_knowledge_graph(query)
       code_test = await qa_agent.test_code_search(query)
       
       # Get embedding
       embedding = await qa_agent._get_query_embedding(query)
       
       return {
           "query": query,
           "embedding_generated": embedding is not None,
           "vector_search_test": vector_test,
           "knowledge_graph_test": kg_test,
           "code_search_test": code_test
       }
   
 
@router.get("/diagnostics")
async def run_diagnostics():
       results = {
           "redis": {"status": "unknown"},
           "elasticsearch": {"status": "unknown"},
           "neo4j": {"status": "unknown"},
           "bedrock": {"status": "unknown"}
       }
       
       # Test Redis
       try:
           redis_service = RedisService()
           ping_result = redis_service.ping()
           results["redis"] = {
               "status": "connected" if ping_result else "failed",
               "details": ping_result
           }
       except Exception as e:
           results["redis"] = {"status": "error", "message": str(e)}
       
       # Test Elasticsearch
       try:
           es_service = ElasticsearchService()
           health = es_service.es.cluster.health()
           results["elasticsearch"] = {
               "status": "connected",
               "details": health
           }
       except Exception as e:
           results["elasticsearch"] = {"status": "error", "message": str(e)}
       
       # Test Neo4j
       try:
           kg_agent = KnowledgeGraphAgent()
           driver_connected = kg_agent.driver is not None
           results["neo4j"] = {
               "status": "connected" if driver_connected else "failed"
           }
       except Exception as e:
           results["neo4j"] = {"status": "error", "message": str(e)}
       
       # Test Bedrock
       try:
           bedrock_service = BedrockService()
           response = await bedrock_service.get_completion("This is a test")
           results["bedrock"] = {
               "status": "connected" if response else "failed",
               "response_sample": response[:50] if response else None
           }
       except Exception as e:
           results["bedrock"] = {"status": "error", "message": str(e)}
       
       return results
   
 
@router.get("/debug/chroma")
async def debug_chroma():
    """Debug ChromaDB collections"""
    try:
        embedding_service = EmbeddingService()
       
        # Get collection stats
        doc_count = embedding_service.doc_collection.count()
        chunk_count = embedding_service.chunk_collection.count()
       
        # Get sample documents
        doc_sample = embedding_service.doc_collection.get(limit=5, include=["metadatas"])
        chunk_sample = embedding_service.chunk_collection.get(limit=5, include=["metadatas"])
       
        return {
            "collections": {
                "documents": {
                    "count": doc_count,
                    "sample": doc_sample
                },
                "chunks": {
                    "count": chunk_count,
                    "sample": chunk_sample
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in chroma debug: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))