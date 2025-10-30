# services/document_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy import select


from app.models.database import Document, DocumentMetadata, DocumentStatus, DocumentSource
from app.services.postgres_service import PostgresService
from app.services.s3_service import S3Service
from app.models.schemas import DocumentCreate, DocumentUpdate

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for document operations, integrating PostgreSQL and S3
    """
    
    def __init__(self, postgres_service: PostgresService, s3_service: S3Service):
        self.postgres = postgres_service
        self.s3 = s3_service
    
    async def create_document(
        self,
        document_create: DocumentCreate,
        content: bytes
    ) -> Document:
        """
        Create a new document
        1. Upload content to S3
        2. Store metadata in PostgreSQL
        """
        try:
            # Create document record in PostgreSQL first
            document = self.postgres.create_document(document_create)
            
            # Upload to S3
            upload_result = await self.s3.upload_document(
                file_content=content,
                filename=document.original_filename,
                document_id=str(document.id),
                source=document.source.value,
                content_type=document.mime_type,
                metadata={
                    'document_id': str(document.id),
                    'uploaded_by': document.uploaded_by,
                    'version': str(document.version)
                }
            )

            
            # Update document with S3 information
            document_update = DocumentUpdate(
                s3_key=upload_result['s3_key'],
                s3_version_id=upload_result['version_id'],
                checksum=upload_result['checksum'],
                file_size=upload_result['size'],
                status=DocumentStatus.UPLOADED
            )

            
            updated_document = self.postgres.update_document(document.id, document_update)
            logger.info(f"Created document {document.id} and uploaded to S3")
            
            return updated_document
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            # If document was created in PostgreSQL but S3 upload failed,
            # mark it as ERROR
            if 'document' in locals():
                self.postgres.update_document(
                    document.id,
                    DocumentUpdate(status=DocumentStatus.ERROR)
                )
            raise
    
    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get document metadata from PostgreSQL"""
        return self.postgres.get_document(document_id)
    
    async def get_document_with_content(
        self,
        document_id: UUID,
        version_id: Optional[str] = None
    ) -> Tuple[Optional[Document], Optional[bytes]]:
        """
        Get document metadata and content
        1. Get metadata from PostgreSQL
        2. Download content from S3
        """
        try:
            # Get document metadata
            document = self.postgres.get_document(document_id)
            print(f"Fetched document metadata for {document_id}: {document}")
            if not document or not document.s3_key:
                return None, None
            
            # Get content from S3
            content, _ = await self.s3.download_document(
                s3_key=document.s3_key,
                version_id=version_id or document.s3_version_id
            )
            print(f"Downloaded content for document {document_id}, size: {len(content) if content else 'None'}")
            return document, content
        except Exception as e:
            logger.error(f"Error getting document with content {document_id}: {str(e)}")
            return None, None
    
    async def get_document_download_url(
        self,
        document_id: UUID,
        expiration: int = 3600
    ) -> Optional[str]:
        """Generate presigned URL for document download"""
        try:
            # Get document metadata
            document = self.postgres.get_document(document_id)
            if not document or not document.s3_key:
                return None
            
            # Generate URL
            url = await self.s3.get_document_url(
                s3_key=document.s3_key,
                expiration=expiration,
                version_id=document.s3_version_id
            )
            
            return url
        except Exception as e:
            logger.error(f"Error generating download URL for document {document_id}: {str(e)}")
            return None
    
    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus
    ) -> bool:
        """Update document status"""
        try:
            document_update = DocumentUpdate(status=status)
            updated = self.postgres.update_document(document_id, document_update)
            return updated is not None
        except Exception as e:
            logger.error(f"Error updating document status {document_id}: {str(e)}")
            return False
    
    async def delete_document(
        self,
        document_id: UUID,
        permanent: bool = False
    ) -> bool:
        """
        Delete document
        If permanent=True, delete from S3 and database
        Otherwise, mark as deleted in database only
        """
        try:
            # Get document first
            document = self.postgres.get_document(document_id)
            if not document:
                return False
            
            if permanent:
                # Delete from S3
                if document.s3_key:
                    await self.s3.delete_document(
                        s3_key=document.s3_key,
                        delete_all_versions=True
                    )
                
                # Delete from PostgreSQL
                self.postgres.delete_document(document_id)
            else:
                # Mark as deleted in PostgreSQL
                document_update = DocumentUpdate(
                    status=DocumentStatus.DELETED,
                    deleted_date=datetime.utcnow()
                )
                self.postgres.update_document(document_id, document_update)
            
            logger.info(f"Document {document_id} {'permanently deleted' if permanent else 'marked as deleted'}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        source: Optional[DocumentSource] = None,
        status: Optional[DocumentStatus] = None
    ) -> List[Document]:
        """List documents with optional filters"""
        return self.postgres.list_documents(
            limit=limit,
            offset=offset,
            source=source,
            status=status
        )
    
    async def get_document_versions(
        self,
        document_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all versions of a document from S3
        """
        try:
            # Get document metadata
            document = self.postgres.get_document(document_id)
            if not document or not document.s3_key:
                return []
            
            # Get versions from S3
            versions = await self.s3.list_document_versions(document.s3_key)
            
            return versions
        except Exception as e:
            logger.error(f"Error getting document versions {document_id}: {str(e)}")
            return []
    
    async def add_extracted_metadata(
        self,
        document_id: UUID,
        metadata: Dict[str, Any],
        extraction_method: str
    ) -> bool:
        """
        Add extracted metadata to document
        """
        try:
            # Add metadata to PostgreSQL
            for key, value in metadata.items():
                self.postgres.create_metadata(
                    doc_id=document_id,
                    key=key,
                    value=str(value),
                    extraction_method=extraction_method
                )
            
            return True
        except Exception as e:
            logger.error(f"Error adding metadata for document {document_id}: {str(e)}")
            return False
    
    async def get_document_metadata_complete(
        self,
        document_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete document metadata including extracted metadata
        """
        try:
            # Get document
            document = self.postgres.get_document(document_id)
            if not document:
                return None
            
            # Get extracted metadata
            metadata_items = self.postgres.get_document_metadata(document_id)
            
            # Format metadata
            extracted_metadata = {}
            for item in metadata_items:
                if item.extraction_method not in extracted_metadata:
                    extracted_metadata[item.extraction_method] = {}
                extracted_metadata[item.extraction_method][item.key] = item.value
            
            # Build complete metadata
            result = {
                "document_info": {
                    "id": str(document.id),
                    "filename": document.filename,
                    "original_filename": document.original_filename,
                    "mime_type": document.mime_type,
                    "source": document.source.value,
                    "status": document.status.value,
                    "version": document.version,
                    "file_size": document.file_size,
                    "checksum": document.checksum,
                    "upload_date": document.upload_date.isoformat() if document.upload_date else None,
                    "uploaded_by": document.uploaded_by,
                    "s3_key": document.s3_key
                },
                "extracted_metadata": extracted_metadata
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting complete metadata for document {document_id}: {str(e)}")
            return None
    
    async def get_documents_for_processing(
        self,
        status: DocumentStatus = DocumentStatus.UPLOADED,
        limit: int = 10
    ) -> List[Tuple[Document, bytes]]:
        """
        Get documents ready for processing with their content
        """
        try:
            # Get documents with specified status
            documents = self.postgres.list_documents(
                limit=limit,
                status=status
            )
            
            result = []
            for document in documents:
                # Get content from S3
                content, _ = await self.s3.download_document(
                    s3_key=document.s3_key,
                    version_id=document.s3_version_id
                )
                
                if content:
                    result.append((document, content))
            
            return result
        except Exception as e:
            logger.error(f"Error getting documents for processing: {str(e)}")
            return []
    
    async def search_documents(
        self,
        query: str,
        source: Optional[DocumentSource] = None,
        status: Optional[str] = None,
        metadata_filters: Optional[Dict[str, str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata
        """
        try:
            # Get documents matching search criteria
            documents = self.postgres.search_documents(
                query=query,
                source=source,
                status=status,
                metadata_filters=metadata_filters,
                limit=limit
            )
            
            # Format results
            results = []
            for document in documents:
                # Get complete metadata
                metadata = await self.get_document_metadata_complete(document.id)
                if metadata:
                    results.append(metadata)
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
        
      # Use list_documents() instead of get_document()
    async def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        source: Optional[DocumentSource] = None,
        status: Optional[DocumentStatus] = None
    ) -> List[Document]:
        return self.postgres.list_documents(
            limit=limit,
            offset=offset,
            source=source,
            status=status
        )

    async def update_document_text(
        self,
        document_id: UUID,
        text: str
    ) -> bool:
        """
        Update the main text content of a document.
        """
        try:
            # Assuming the DocumentUpdate Pydantic model has a 'text' or 'content' field
            document_update = DocumentUpdate(
                # Use 'text=text' or 'content=text' based on your model
                text=text
            )
            print("updating document text in DB", document_id, len(text) if text else 0)
            updated = self.postgres.update_document(document_id, document_update)
            return updated is not None
        except Exception as e:
            logger.error(f"Error updating document text for {document_id}: {str(e)}")
            return False
        
    async def get_document_by_s3_key(self, s3_key: str):
        return self.postgres.get_document_by_s3_key(s3_key)

    async def get_document_by_checksum(self, checksum: str):
        return self.postgres.get_document_by_checksum(checksum)



    async def update_document_timestamp(self, document_id: UUID) -> bool:
        """
        Update the 'updated_at' timestamp for a document in PostgreSQL.
        """
        try:
            document_update = DocumentUpdate(
                updated_at=datetime.utcnow()
            )
            updated = self.postgres.update_document(document_id, document_update)
            return updated is not None
        except Exception as e:
            logger.error(f"Error updating document timestamp for {document_id}: {str(e)}")
            return False
