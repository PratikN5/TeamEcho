import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime
import os
import json
import boto3
from botocore.config import Config
import chromadb
from chromadb.config import Settings
 
from app.core.config import get_settings
 
settings = get_settings()
 
logger = logging.getLogger(__name__)
 
class EmbeddingService:
    def __init__(self):
        self.embedding_model = self._initialize_embedding_model()
        self.chroma_client = self._initialize_chroma_client()
        self.doc_collection = None
        self.chunk_collection = None
        self._initialize_collections()
   
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
   
    def _initialize_chroma_client(self):
        """Initialize ChromaDB client"""
        try:
            # For persistent storage
            if hasattr(settings, 'CHROMA_PERSIST_DIRECTORY'):
                return chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIRECTORY,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # For in-memory storage (development)
                return chromadb.Client(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
   
    def _initialize_collections(self):
        """Initialize ChromaDB collections for documents and chunks"""
        try:
            # Create or get document collection
            self.doc_collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
           
            # Create or get chunk collection
            self.chunk_collection = self.chroma_client.get_or_create_collection(
                name="chunks",
                metadata={"hnsw:space": "cosine"}
            )
           
            logger.info("ChromaDB collections initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise
   
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using AWS Bedrock"""
        try:
            # Use Amazon Titan for embeddings
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
        """Embed entire document and store in ChromaDB"""
        try:
            # Generate embedding
            embedding = await self.get_embedding(text)
           
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.now().isoformat()
           
            # Store in ChromaDB
            self.doc_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )
           
            logger.info(f"Document {doc_id} embedded and stored successfully")
            return True
        except Exception as e:
            logger.error(f"Error embedding document {doc_id}: {str(e)}")
            return False
   
    async def embed_chunks(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Embed document chunks and store in ChromaDB"""
        try:
            chunk_ids = []
            chunk_embeddings = []
            chunk_texts = []
            chunk_metadatas = []
           
            for i, chunk in enumerate(chunks):
                # Generate chunk ID
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
               
                # Extract text and metadata
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                metadata.update({
                    "chunk_index": i,
                    "doc_id": doc_id,
                    "timestamp": datetime.now().isoformat()
                })
               
                # Generate embedding
                embedding = await self.get_embedding(text)
               
                chunk_embeddings.append(embedding)
                chunk_texts.append(text)
                chunk_metadatas.append(metadata)
           
            # Batch add to ChromaDB
            self.chunk_collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
           
            logger.info(f"Embedded {len(chunks)} chunks for document {doc_id}")
            return chunk_ids
        except Exception as e:
            logger.error(f"Error embedding chunks for document {doc_id}: {str(e)}")
            return []
   
    async def search_similar_chunks(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query"""
        try:
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
           
            # Search in ChromaDB
            results = self.chunk_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict if filter_dict else None
            )
           
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    score = 1 - results['distances'][0][i]  # Convert distance to similarity
                   
                    if score >= min_score:
                        formatted_results.append({
                            "id": results['ids'][0][i],
                            "text": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "score": score
                        })
           
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
   
    async def search_similar_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents based on query"""
        try:
            logger.info(f"Searching for documents similar to: '{query}'")
            logger.info(f"Filter: {filter_dict}, k: {k}, min_score: {min_score}")
           
            # Check if collection has documents
            collection_count = self.doc_collection.count()
            logger.info(f"Total documents in collection: {collection_count}")
           
            if collection_count == 0:
                logger.warning("Document collection is empty!")
                return []
           
            # Generate query embedding
            query_embedding = await self.get_embedding(query)
            logger.info(f"Generated embedding with length: {len(query_embedding)}")
           
            # Search in ChromaDB
            logger.info("Querying ChromaDB...")
            results = self.doc_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection_count),  # Don't request more than exist
                where=filter_dict if filter_dict else None
            )
           
            # Log raw results
            logger.info(f"""Raw query results: {json.dumps({
    'ids_length': len(results['ids'][0]) if results['ids'] else 0,
    'has_documents': bool(results['documents']),
    'has_metadatas': bool(results['metadatas']),
    'has_distances': bool(results['distances'])
})}""")
 
           
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    logger.info(f"Document {i}: ID={results['ids'][0][i]}, Score={score}")
                   
                    if score >= min_score:
                        formatted_results.append({
                            "id": results['ids'][0][i],
                            "text": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "score": score
                        })
                    else:
                        logger.info(f"Document {results['ids'][0][i]} excluded: score {score} < min_score {min_score}")
            else:
                logger.warning("ChromaDB returned no results or empty result structure")
           
            logger.info(f"Returning {len(formatted_results)} formatted results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}", exc_info=True)
            return []
 
   
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its chunks from ChromaDB"""
        try:
            # Delete document
            try:
                self.doc_collection.delete(ids=[doc_id])
            except Exception as e:
                logger.warning(f"Document {doc_id} not found in collection: {str(e)}")
           
            # Delete all chunks for this document
            chunk_results = self.chunk_collection.get(
                where={"doc_id": doc_id}
            )
           
            if chunk_results and chunk_results['ids']:
                self.chunk_collection.delete(ids=chunk_results['ids'])
                logger.info(f"Deleted {len(chunk_results['ids'])} chunks for document {doc_id}")
           
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
   
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections"""
        try:
            doc_count = self.doc_collection.count()
            chunk_count = self.chunk_collection.count()
           
            return {
                "documents": doc_count,
                "chunks": chunk_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"documents": 0, "chunks": 0}
       
    async def get_all_documents(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all documents from collection with optional filter"""
        try:
            results = self.doc_collection.get(
                where=filter_dict,
                limit=limit,
                include=["documents", "metadatas", "embeddings"]
            )
           
            documents = []
            if results and results.get('ids'):
                for i, doc_id in enumerate(results['ids']):
                    documents.append({
                        "id": doc_id,
                        "text": results['documents'][i] if results.get('documents') else "",
                        "metadata": results['metadatas'][i] if results.get('metadatas') else {},
                        "has_embedding": results['embeddings'][i] is not None if results.get('embeddings') else False
                    })
           
            logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []