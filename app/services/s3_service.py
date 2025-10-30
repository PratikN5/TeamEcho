# services/s3_service.py
import boto3
import logging
import os
import uuid
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any, Tuple, BinaryIO, List
from datetime import datetime, timedelta
import mimetypes
import hashlib
from app.core.config import get_settings

settings = get_settings()
prefixes = os.getenv("S3_MONITORING_PREFIXES", "upload/,sharepoint/,github/").split(",")

logger = logging.getLogger(__name__)

class S3Service:
    """
    Service for interacting with AWS S3 or compatible storage (like MinIO)
    Handles document upload, download, versioning, and URL generation
    """
    
    def __init__(self):
        """Initialize S3 client with configuration from settings"""
        self.s3_client = self._initialize_s3_client()
        self.bucket_name = settings.S3_BUCKET_NAME
        self._ensure_bucket_exists()
    
    def _initialize_s3_client(self):
        """Initialize S3 client with AWS credentials or endpoint URL for MinIO"""
        try:
            # Check if using MinIO (local development) or AWS S3
            if settings.S3_ENDPOINT_URL:
                # MinIO or custom S3-compatible storage
                s3_client = boto3.client(
                    's3',
                    endpoint_url=settings.S3_ENDPOINT_URL,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    config=boto3.session.Config(signature_version='s3v4')
                )
                logger.info(f"Initialized S3 client with custom endpoint: {settings.S3_ENDPOINT_URL}")
            else:
                # AWS S3
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                logger.info(f"Initialized AWS S3 client in region: {settings.AWS_REGION}")
            
            return s3_client
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _ensure_bucket_exists(self):
        """Check if bucket exists, create it if it doesn't"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            
            if error_code == '404':
                # Bucket doesn't exist, create it
                logger.info(f"Creating S3 bucket: {self.bucket_name}")
                
                if settings.S3_ENDPOINT_URL:
                    # MinIO doesn't need location constraint
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    # AWS S3 needs location constraint
                    location = {'LocationConstraint': settings.AWS_REGION}
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration=location
                    )
                
                # Enable versioning
                self.s3_client.put_bucket_versioning(
                    Bucket=self.bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                logger.info(f"Created S3 bucket '{self.bucket_name}' with versioning enabled")
            else:
                logger.error(f"Error checking S3 bucket: {str(e)}")
                raise
    
    def _generate_s3_key(self, document_id: str, filename: str, source: str) -> str:
        """Generate S3 key with organized folder structure"""
        # Extract extension
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = '.bin'  # Default extension if none provided
        
        # Generate date-based path (YYYY/MM/DD)
        date_path = datetime.utcnow().strftime('%Y/%m/%d')
        
        # Generate key: source/YYYY/MM/DD/document_id/filename
        s3_key = f"{source}/{date_path}/{document_id}{ext}"
        
        return s3_key
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str,
        source: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload document to S3
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            document_id: Unique document ID
            source: Document source (e.g., 'upload', 'api', 'email')
            content_type: MIME type of the file
            metadata: Additional metadata to store with the file
            
        Returns:
            Dict with upload details
        """
        try:
            # Generate S3 key
            s3_key = self._generate_s3_key(document_id, filename, source)
  
            
            # Determine content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
                if not content_type:
                    content_type = 'application/octet-stream'
            
            # Calculate checksum
            checksum = hashlib.md5(file_content).hexdigest()
            
            # Prepare metadata
            s3_metadata = {
                'document_id': document_id,
                'original_filename': filename,
                'source': source,
                'upload_date': datetime.utcnow().isoformat(),
                'checksum': checksum
            }
            
            # Add additional metadata if provided
            if metadata:
                # Convert all metadata values to strings for S3
                s3_metadata.update({k: str(v) for k, v in metadata.items()})
            
            # Upload to S3
            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                Metadata=s3_metadata
            )
            
            # Get version ID
            version_id = response.get('VersionId')
            
            logger.info(f"Uploaded document {document_id} to S3: {s3_key} (version: {version_id})")

            return {
                'document_id': document_id,
                's3_key': s3_key,
                'version_id': version_id,
                'content_type': content_type,
                'size': len(file_content),
                'checksum': checksum
            }
        except Exception as e:
            logger.error(f"Error uploading document {document_id} to S3: {str(e)}")
            raise
    
    async def download_document(
        self,
        s3_key: str,
        version_id: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Download document from S3
        
        Args:
            s3_key: S3 object key
            version_id: Specific version to download (optional)
            
        Returns:
            Tuple of (file_content, metadata)
        """
        try:
            # Prepare parameters
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            # Add version ID if provided
            if version_id:
                params['VersionId'] = version_id
            
            # Download from S3
            response = self.s3_client.get_object(**params)
            
            # Read content
            file_content = response['Body'].read()
            
            # Extract metadata
            metadata = {
                'content_type': response.get('ContentType', 'application/octet-stream'),
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified', datetime.utcnow()).isoformat(),
                'version_id': response.get('VersionId'),
                'e_tag': response.get('ETag', '').strip('"'),
                **response.get('Metadata', {})
            }
            
            logger.info(f"Downloaded document from S3: {s3_key} (version: {version_id or 'latest'})")
            
            return file_content, metadata
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'NoSuchKey':
                logger.error(f"Document not found in S3: {s3_key}")
                return None, {}
            else:
                logger.error(f"Error downloading document from S3: {s3_key}, {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error downloading document from S3: {s3_key}, {str(e)}")
            raise
    
    async def get_document_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        version_id: Optional[str] = None
    ) -> str:
        """
        Generate presigned URL for document access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            version_id: Specific version for URL (optional)
            
        Returns:
            Presigned URL
        """
        try:
            # Prepare parameters
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            # Add version ID if provided
            if version_id:
                params['VersionId'] = version_id
            
            # Generate URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {s3_key} (expires in {expiration}s)")
            
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL for {s3_key}: {str(e)}")
            raise
    
    async def delete_document(
        self,
        s3_key: str,
        delete_all_versions: bool = False
    ) -> bool:
        """
        Delete document from S3
        
        Args:
            s3_key: S3 object key
            delete_all_versions: If True, delete all versions of the document
            
        Returns:
            Success status
        """
        try:
            if delete_all_versions:
                # List all versions
                versions = self.s3_client.list_object_versions(
                    Bucket=self.bucket_name,
                    Prefix=s3_key
                )
                
                # Delete markers
                if 'DeleteMarkers' in versions:
                    for marker in versions['DeleteMarkers']:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=s3_key,
                            VersionId=marker['VersionId']
                        )
                
                # Delete versions
                if 'Versions' in versions:
                    for version in versions['Versions']:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=s3_key,
                            VersionId=version['VersionId']
                        )
                
                logger.info(f"Deleted all versions of document {s3_key}")
            else:
                # Delete latest version (creates delete marker)
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.info(f"Marked document {s3_key} as deleted (latest version)")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting document {s3_key}: {str(e)}")
            return False
    
    async def list_document_versions(
        self,
        s3_key: str
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a document
        
        Args:
            s3_key: S3 object key
            
        Returns:
            List of version details
        """
        try:
            # List all versions
            response = self.s3_client.list_object_versions(
                Bucket=self.bucket_name,
                Prefix=s3_key
            )
            
            versions = []
            
            # Process versions
            if 'Versions' in response:
                for version in response['Versions']:
                    versions.append({
                        'version_id': version['VersionId'],
                        'last_modified': version['LastModified'].isoformat(),
                        'size': version['Size'],
                        'is_latest': version['IsLatest'],
                        'e_tag': version['ETag'].strip('"')
                    })
            
            # Process delete markers
            if 'DeleteMarkers' in response:
                for marker in response['DeleteMarkers']:
                    versions.append({
                        'version_id': marker['VersionId'],
                        'last_modified': marker['LastModified'].isoformat(),
                        'is_latest': marker['IsLatest'],
                        'is_delete_marker': True
                    })
            
            # Sort by last modified (newest first)
            versions.sort(key=lambda x: x['last_modified'], reverse=True)
            
            logger.info(f"Listed {len(versions)} versions for document {s3_key}")
            
            return versions
        except Exception as e:
            logger.error(f"Error listing versions for document {s3_key}: {str(e)}")
            raise
    
    async def copy_document(
        self,
        source_key: str,
        destination_key: str,
        source_version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Copy document within the same bucket
        
        Args:
            source_key: Source S3 key
            destination_key: Destination S3 key
            source_version_id: Specific version to copy (optional)
            
        Returns:
            Copy details
        """
        try:
            # Prepare source path
            source_path = f"{self.bucket_name}/{source_key}"
            if source_version_id:
                source_path += f"?versionId={source_version_id}"
            
            # Copy object
            response = self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource=source_path,
                Key=destination_key
            )
            
            logger.info(f"Copied document from {source_key} to {destination_key}")
            
            return {
                'source_key': source_key,
                'destination_key': destination_key,
                'version_id': response.get('VersionId'),
                'copy_status': response.get('CopyObjectResult', {}).get('ETag', '').strip('"')
            }
        except Exception as e:
            logger.error(f"Error copying document from {source_key} to {destination_key}: {str(e)}")
            raise
    
    async def check_document_exists(
        self,
        s3_key: str,
        version_id: Optional[str] = None
    ) -> bool:
        """
        Check if document exists in S3
        
        Args:
            s3_key: S3 object key
            version_id: Specific version to check (optional)
            
        Returns:
            True if document exists
        """
        try:
            # Prepare parameters
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            # Add version ID if provided
            if version_id:
                params['VersionId'] = version_id
            
            # Check if object exists
            self.s3_client.head_object(**params)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                return False
            else:
                logger.error(f"Error checking if document exists: {s3_key}, {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error checking if document exists: {s3_key}, {str(e)}")
            raise
    
    async def update_document_metadata(
        self,
        s3_key: str,
        metadata: Dict[str, str],
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update document metadata
        
        Args:
            s3_key: S3 object key
            metadata: New metadata dictionary
            version_id: Specific version to update (optional)
            
        Returns:
            Update details
        """
        try:
            # First, get current object to preserve attributes
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            if version_id:
                params['VersionId'] = version_id
            
            current = self.s3_client.head_object(**params)
            
            # Convert all metadata values to strings for S3
            s3_metadata = {k: str(v) for k, v in metadata.items()}
            
            # Copy object to itself with new metadata
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            if version_id:
                copy_source['VersionId'] = version_id
            
            response = self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource=copy_source,
                Key=s3_key,
                Metadata=s3_metadata,
                MetadataDirective='REPLACE',
                ContentType=current.get('ContentType', 'application/octet-stream')
            )
            
            logger.info(f"Updated metadata for document {s3_key}")
            
            return {
                's3_key': s3_key,
                'version_id': response.get('VersionId'),
                'metadata': s3_metadata
            }
        except Exception as e:
            logger.error(f"Error updating metadata for document {s3_key}: {str(e)}")
            raise
        

      # Add these methods to the existing S3Service class

    async def list_recent_files(
        self,
        prefix: str = "",
        minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        List files added or modified in the last X minutes
        
        Args:
            prefix: S3 key prefix to filter by
            minutes: Number of minutes to look back
            
        Returns:
            List of file information dictionaries
        """
        try:
            # Calculate cutoff time
            import time
            cutoff_time = time.time() - (minutes * 60)
            
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix or ""
            )
            # print("Listed objects from S3" , response)
            
            # Filter by last modified time
            recent_files = []
            
            if "Contents" in response:
                for item in response["Contents"]:
                    # Check if file was modified after cutoff time
                    if item["LastModified"].timestamp() > cutoff_time:
                        recent_files.append({
                            "key": item["Key"],
                            "size": item["Size"],
                            "last_modified": item["LastModified"].isoformat(),
                            "etag": item["ETag"].strip('"')
                        })
            
            logger.info(f"Found {len(recent_files)} files modified in the last {minutes} minutes")
            return recent_files
        
        except Exception as e:
            logger.error(f"Error listing recent files: {str(e)}")
            return []

    async def configure_event_notifications(
        self,
        lambda_arn: str,
        events: List[str] = ["s3:ObjectCreated:*"]
    ) -> bool:
        """
        Configure S3 event notifications to a Lambda function
        
        Args:
            lambda_arn: ARN of the Lambda function to trigger
            events: List of S3 event types to trigger on
            
        Returns:
            Success status
        """
        try:
            # Configure event notification
            notification_config = {
                'LambdaFunctionConfigurations': [
                    {
                        'LambdaFunctionArn': lambda_arn,
                        'Events': events,
                    }
                ]
            }
            
            self.s3_client.put_bucket_notification_configuration(
                Bucket=self.bucket_name,
                NotificationConfiguration=notification_config
            )
            
            logger.info(f"Configured S3 event notifications to {lambda_arn}")
            return True
        
        except Exception as e:
            logger.error(f"Error configuring S3 event notifications: {str(e)}")
            return False
        

        

