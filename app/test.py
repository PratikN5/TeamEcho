# from app.services.document_service import DocumentService
# from app.services.embedding_service import EmbeddingService
# from app.services.postgres_service import PostgresService
# from app.services.s3_service import S3Service
# from app.services.redis_service import RedisService
# import asyncio
# from app.models.db import get_db
# db_gen = get_db()
# db = next(db_gen) 
# redis_service = RedisService()

# async def test_embedding_generation(doc_id):
#     postgres_service = PostgresService(db)
#     s3_service = S3Service()
#     document_service = DocumentService(postgres_service, s3_service)
#     embedding_service = EmbeddingService(redis_service)

#     result = await document_service.get_document_with_content(doc_id)
#     print(result)
#     # ✅ handle list response
#     # Handle tuple response: (<Document>, bytes)
#     if isinstance(result, tuple):
#         document, content = result
#         text = content.decode("utf-8", errors="ignore")
#     elif isinstance(result, list) and len(result) > 0:
#         document = result[0]
#         text = document.get("text", "")
#     else:
#         print(f"❌ No document found for {doc_id}")
#         return

#     print(f"📄 Found document: {document.filename}")
#     print(f"🧾 Extracted text (first 300 chars):\n{text[:300] if text else 'No text available.'}")

#     if not text:
#         print("⚠️ No text found — embedding cannot be generated.")
#         return

#     # Generate embedding
#     embedding = await embedding_service.get_embedding(text)
#     print(f"✅ Embedding generated successfully.")
#     print(f"📏 Embedding length: {len(embedding)}")
#     print(f"🔢 First 10 dimensions: {embedding[:10]}")

# if __name__ == "__main__":
#     asyncio.run(test_embedding_generation("2fcf2666-f9b6-4d5e-9da0-d31720011dab"))




import asyncio
import logging
from app.services.document_service import DocumentService
from app.services.processing_service import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.redis_service import RedisService
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent # your KG service
from app.services.document_chunking_service import ChunkingService
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.models.db import get_db
db_gen = get_db()
db = next(db_gen) 
logging.basicConfig(level=logging.INFO)
postgres_service = PostgresService(db)
s3_service = S3Service()

async def main():
    # Initialize all services
    document_service = DocumentService(postgres_service, s3_service)
    
    redis_service = RedisService()
    embedding_service = EmbeddingService(redis_service)
    kg_service = KnowledgeGraphAgent()
    chunking_service = ChunkingService()

    processing_service = DocumentProcessingService(
        document_service=document_service,
        embedding_service=embedding_service,
        redis_service=redis_service,
        kg_service=kg_service,
        chunking_service=chunking_service,
        use_gpu=False
    )

    # --- TEST DOCUMENT ---
    document_id = "91c9a154-2cf8-4175-87c1-75896419ff67"  # replace with a real document ID
    logging.info(f"Starting processing test for document: {document_id}")

    extracted_text = await processing_service.process_document(document_id)

    if extracted_text:
        logging.info(f"Processing completed for {document_id}")
        logging.info(f"Extracted text preview: {extracted_text[:200]}")  # first 200 chars
    else:
        logging.error(f"Processing failed for {document_id}")

if __name__ == "__main__":
    asyncio.run(main())


# import os
# import redis
# import socket

# def get_redis_host():
#     host = os.getenv("REDIS_HOST", "localhost")
#     # If 'redis' isn't resolvable locally, fallback to localhost
#     try:
#         socket.gethostbyname(host)
#     except socket.gaierror:
#         host = "localhost"
#     return host

# redis_host = get_redis_host()
# redis_port = int(os.getenv("REDIS_PORT", 6380))

# print(f"Connecting to Redis at {redis_host}:{redis_port}...")
# r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# print("PING:", r.ping())

# modules = r.execute_command("MODULE", "LIST")
# print("\nLoaded modules:")
# for mod in modules:
#     mod_info = {mod[i]: mod[i + 1] for i in range(0, len(mod), 2)}
#     print(f" - {mod_info['name']} (version: {mod_info['ver']})")

# import asyncio
# import logging
# import sys
# from app.services.document_service import DocumentService
# from app.services.postgres_service import PostgresService
# from app.services.s3_service import S3Service
# from app.services.processing_service import DocumentProcessingService
# from app.services.document_chunking_service import ChunkingService
# from app.services.embedding_service import EmbeddingService
# from app.services.redis_service import RedisService
# from app.agents.ingestion_agent import IngestionAgent
# from app.models.db import get_db

# # Configure logging to stdout
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )

# async def test_agent():
#     print("Starting test...")
#     db = next(get_db())
    
#     try:
#         # Initialize services
#         postgres_service = PostgresService(db)
#         s3_service = S3Service()
#         document_service = DocumentService(postgres_service, s3_service)
#         redis_service = RedisService()
#         embedding_service = EmbeddingService(redis_service)
#         chunking_service = ChunkingService()
#         processing_service = DocumentProcessingService(
#             document_service=document_service,
#             chunking_service=chunking_service,
#             embedding_service=embedding_service,
#             redis_service=redis_service
#         )
        
#         # Create ingestion agent with just S3 service
#         agent = IngestionAgent(
#             document_service=document_service,
#             processing_service=processing_service,
#             s3_service=s3_service
#         )
        
#         # Test S3 check directly
#         print("Testing S3 check_s3_events...")
#         await agent.check_s3_events()
#         print("S3 check completed")
        
#         # Test one cycle of monitoring
#         print("Testing one monitoring cycle...")
#         agent.polling_interval = 1  # Set to 1 second for quick test
        
#         # Override the monitoring loop to run just once
#         original_start_monitoring = agent.start_monitoring
        
#         async def test_monitoring():
#             try:
#                 print("Running modified monitoring loop")
#                 logger = logging.getLogger(__name__)
#                 logger.info("Running ingestion cycle")
#                 print("Inside ingestion cycle")
                
#                 if agent.s3_monitoring_enabled and agent.s3_service:
#                     print("Checking S3 events")
#                     await agent.check_s3_events()
#                     print("Finished checking S3")
                
#                 print("Test monitoring cycle complete")
#             except Exception as e:
#                 print(f"ERROR: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
        
#         await test_monitoring()
#         print("Test completed successfully")
    
#     finally:
#         db.close()

# if __name__ == "__main__":
#     asyncio.run(test_agent())
