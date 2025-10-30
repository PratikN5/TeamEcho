# services/processing_service.py
import logging
from typing import Dict, Any, Optional
from app.services.ocr_service import OCRService
from io import BytesIO
from docx import Document

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """
    Service for processing documents after they've been ingested
    """
    def __init__(self, document_service, embedding_service, redis_service,  kg_service=None, chunking_service=None, use_gpu=False):
        self.document_service = document_service
        self.embedding_service = embedding_service
        self.redis_service = redis_service
        self.chunking_service = chunking_service
        self.kg_service = kg_service
        self.ocr_service = OCRService(use_gpu=use_gpu)
        logger.info(f"Document Processing Service initialized with GPU support: {use_gpu}")

    async def process_document(self, document_id: str):
        """
        Process a document by its ID
        
        Args:
            document_id: The ID of the document to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Processing document: {document_id}")
            
            # Get document from database
            document = await self.document_service.get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return False
            
            # Download document content
            print("starting to get document with content")
            document, content = await self.document_service.get_document_with_content(document_id)
            if not content:
                logger.error(f"Could not retrieve document content: {document_id}")
                return False
            print("got document content", len(content) , content[:20] , type(content))
            # Extract text using OCR service
            print("starting to extract text from file")
            extracted_text, metadata = self.ocr_service.extract_text_from_file(
                file_content=content,
                filename=document.filename
            )
            print("extracted text from file", len(extracted_text) if extracted_text else 0)

            # 🟩 Step 2 — Fallback if OCR didn’t extract anything
            if not extracted_text and document.filename.endswith(".docx"):
                try:
                    doc = Document(BytesIO(content))
                    extracted_text = "\n".join([p.text for p in doc.paragraphs])
                    metadata["extraction_method"] = "docx_text"
                    logger.info(f"Extracted text from DOCX fallback for {document_id}")
                except Exception as e:
                    logger.error(f"DOCX fallback extraction failed: {e}")

            if not extracted_text:
                logger.warning(f"No text could be extracted for document {document_id}")
                return False
            print("extracted text length", len(extracted_text) if extracted_text else 0)
            
            # Update document with extracted text
            print("updating document text and metadata")
            await self.document_service.update_document_text(
                document_id=document_id,
                text=extracted_text
            )
            print("updated document text")
            
            # Add extracted metadata
            print("adding extracted metadata")
            await self.document_service.add_extracted_metadata(
                document_id=document_id,
                metadata=metadata,
                extraction_method=metadata.get('extraction_method', 'ocr')
            )
            print("added extracted metadata")

            # --- Knowledge Graph Integration ---
            print("checking for KG service")
            if self.kg_service:
                # Extract entities & relations (can use NLP / regex / custom logic)
                entities = await self.kg_service.extract_entities(extracted_text)
                relations = await self.kg_service.extract_relationships(entities, extracted_text)
                
                # Add document to KG
                await self.kg_service.build_graph(document_id, document.filename, entities, relations)
                logger.info(f"Knowledge Graph updated for document {document_id}")
            print("KG service done")
            
            # If chunking service is available, chunk the document
            print("checking for chunking service")
            if self.chunking_service:
                if not isinstance(extracted_text, str):
                    logger.error(f"Expected extracted_text to be str but got {type(extracted_text)} for {document_id}")
                else:
                    chunks = await self.chunking_service.chunk_document(
    extracted_text,
    {"document_id": str(document_id)}
)
                    logger.info(f"Document {document_id} chunked into {len(chunks)} chunks")
            print("chunking service done",chunks if 'chunks' in locals() else "no chunks")

            print("generating embeddings for document")
            # Generate embeddings for the document
            await self.embedding_service.embed_document(
                doc_id=str(document_id),
                text=extracted_text,
                metadata=metadata
            )
            print("generated embeddings for document",document_id)
            
            # Mark document as processed
            print("updating document status to processed")
            await self.document_service.update_document_status(
                document_id=document_id,
                status="processed"
            )
            print("updated document status to processed",document_id)
            logger.info(f"Successfully processed document: {document_id}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
            
            # Mark document as failed
            try:
                await self.document_service.update_document_status(
                    document_id=document_id,
                    status="failed"
                )
                
                # Add error to metadata
                await self.document_service.add_extracted_metadata(
                    document_id=document_id,
                    metadata={"error": str(e)},
                    extraction_method="processing_error"
                )
            except Exception as update_error:
                logger.error(f"Error updating document status: {str(update_error)}")
            
            return False
