# agents/ingestion_agent.py
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import uuid
import os
import json

from app.services.document_service import DocumentService
from app.services.processing_service import DocumentProcessingService
from app.services.github_service import GitHubService
from app.services.sharepoint_service import SharePointService
from app.services.s3_service import S3Service
from app.models.database import DocumentSource, DocumentStatus
from app.models.schemas import DocumentCreate
from app.core.config import get_settings

settings = get_settings()
github_service = GitHubService()

logger = logging.getLogger(__name__)

class IngestionAgent:
    """
    Centralized agent for ingesting documents from multiple sources:
    - Direct upload (handled by API)
    - GitHub repositories
    - SharePoint sites
    - S3 bucket events
    
    The agent can run as a background process to periodically check sources
    or be triggered by webhook events.
    """
     # Paths to exclude from ingestion (local envs, caches, version control, etc.)
    EXCLUDED_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".mypy_cache", ".pytest_cache"}
    
    def __init__(
        self,
        document_service: DocumentService,
        processing_service: DocumentProcessingService,
        github_service: Optional[GitHubService] = None,
        sharepoint_service: Optional[SharePointService] = None,
        s3_service: Optional[S3Service] = None
    ):
        self.document_service = document_service
        self.processing_service = processing_service
        self.github_service = github_service or GitHubService()
        self.sharepoint_service = sharepoint_service or SharePointService()
        self.s3_service = s3_service
        
        # Track processed items to avoid duplicates
        self._processed_items = set()
        self._max_processed_items = 10000  # Prevent memory growth
        
        # Configuration
        self.polling_interval = settings.INGESTION_POLLING_INTERVAL  # seconds
        self.github_enabled = settings.GITHUB_INTEGRATION_ENABLED
        self.sharepoint_enabled = settings.SHAREPOINT_INTEGRATION_ENABLED
        self.s3_monitoring_enabled = settings.S3_EVENT_MONITORING_ENABLED

        # Load state if available
        self._load_state()

    def _should_skip_path(self, path: str) -> bool:
        """Check if a path should be skipped based on excluded directories"""
        return any(excluded in path.split("/") for excluded in self.EXCLUDED_DIRS)

        
    
    def _load_state(self):
        """Load agent state from disk to resume operations"""
        try:
            state_file = os.path.join(settings.DATA_DIR, "ingestion_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                    
                # Load last check times
                self.last_github_check = datetime.fromisoformat(
                    state.get("last_github_check", "2000-01-01T00:00:00")
                )
                self.last_sharepoint_check = datetime.fromisoformat(
                    state.get("last_sharepoint_check", "2000-01-01T00:00:00")
                )
                
                # Load processed items (limited set)
                self._processed_items = set(state.get("processed_items", [])[-self._max_processed_items:])
                
                logger.info(f"Loaded ingestion agent state with {len(self._processed_items)} processed items")
            else:
                # Default initial state
                self.last_github_check = datetime.utcnow() - timedelta(days=settings.INITIAL_LOOKBACK_DAYS)
                self.last_sharepoint_check = datetime.utcnow() - timedelta(days=settings.INITIAL_LOOKBACK_DAYS)
                logger.info("No previous state found, starting with default lookback period")
        except Exception as e:
            logger.error(f"Error loading ingestion state: {str(e)}")
            # Default initial state
            self.last_github_check = datetime.utcnow() - timedelta(days=settings.INITIAL_LOOKBACK_DAYS)
            self.last_sharepoint_check = datetime.utcnow() - timedelta(days=settings.INITIAL_LOOKBACK_DAYS)
    
    def _save_state(self):
        """Save agent state to disk"""
        try:
            state_file = os.path.join(settings.DATA_DIR, "ingestion_state.json")
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            state = {
                "last_github_check": self.last_github_check.isoformat(),
                "last_sharepoint_check": self.last_sharepoint_check.isoformat(),
                "processed_items": list(self._processed_items)[-self._max_processed_items:],
                "last_updated": datetime.utcnow().isoformat()
            }
            
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Saved ingestion agent state")
        except Exception as e:
            logger.error(f"Error saving ingestion state: {str(e)}")
    
    def _mark_as_processed(self, item_id: str):
        """Mark an item as processed to avoid duplicates"""
        self._processed_items.add(item_id)
        
        # Prevent memory growth by limiting the set size
        if len(self._processed_items) > self._max_processed_items:
            # Convert to list, keep most recent items, convert back to set
            as_list = list(self._processed_items)
            self._processed_items = set(as_list[-self._max_processed_items:])
    
    def _is_processed(self, item_id: str) -> bool:
        """Check if an item has been processed already"""
        return item_id in self._processed_items
    
    async def start_monitoring(self):
        """Start the monitoring loop for all enabled sources"""
        logger.info("Starting ingestion agent monitoring loop")
        
        try:
            while True:
                logger.info("Running ingestion cycle")
                print("inside ingestion cycle monitoring loop")
                
                # Check each source
                if self.github_enabled and self.github_service:
                    await self.fetch_from_github()
                
                if self.sharepoint_enabled and self.sharepoint_service:
                    await self.fetch_from_sharepoint()
                
                if self.s3_monitoring_enabled and self.s3_service:
                    await self.check_s3_events()
                
                # Save state after each cycle
                self._save_state()
                
                # Wait for next cycle
                logger.info(f"Ingestion cycle complete, sleeping for {self.polling_interval} seconds")
                await asyncio.sleep(self.polling_interval)
        except Exception as e:
            logger.error(f"Error in ingestion monitoring loop: {str(e)}")
            # Save state on error
            self._save_state()
            raise
    
    async def process_upload(self, file_content: bytes, filename: str, source: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            s3_key = metadata.get("s3_key") if metadata else None
            checksum = self._calculate_checksum(file_content)
            
            # Check if document already exists (by s3_key or checksum)
            existing_doc = None
            if s3_key:
                existing_doc = await self.document_service.get_document_by_s3_key(s3_key)
            if not existing_doc:
                existing_doc = await self.document_service.get_document_by_checksum(checksum)
            
            if existing_doc:
                logger.info(f"Document already exists in DB: {existing_doc.id}, updating existing document.")
                
                # Update metadata or reprocess text
                await self.document_service.update_document_timestamp(existing_doc.id)
                
                # Re-run processing on the same document
                await self.processing_service.process_document(existing_doc.id)
                
                return {
                    "success": True,
                    "document_id": str(existing_doc.id),
                    "filename": existing_doc.filename,
                    "status": existing_doc.status.value
                }

            # If new document
            document_id = str(uuid.uuid4())
            document_create = DocumentCreate(
                filename=filename,
                original_filename=filename,
                mime_type=self._guess_mime_type(filename),
                source=DocumentSource(source),
                uploaded_by=metadata.get("uploaded_by", "system"),
                file_size=len(file_content),
                checksum=checksum
            )

            document = await self.document_service.create_document(document_create, file_content)

            if metadata:
                await self.document_service.add_extracted_metadata(
                    document_id=document.id,
                    metadata=metadata,
                    extraction_method="upload_metadata"
                )

            logger.info(f"Processed new upload: {filename} (ID: {document.id})")

            # Process the document
            await self.processing_service.process_document(document.id)

            return {
                "success": True,
                "document_id": str(document.id),
                "filename": document.filename,
                "status": document.status.value
            }

        except Exception as e:
            logger.error(f"Error processing upload {filename}: {str(e)}")
            return {"success": False, "error": str(e), "filename": filename}

    
    async def fetch_from_github(self):
        """
        Fetch documents from configured GitHub repositories
        """
        if not self.github_service:
            logger.warning("GitHub service not configured, skipping")
            return
        
        try:
            logger.info(f"Checking GitHub repositories since {self.last_github_check}")
            
            # Get list of repositories to monitor
            repositories = settings.GITHUB_REPOSITORIES
            
            for repo in repositories:
                # Get updated files since last check
                files = await self.github_service.get_updated_files(
                    repo=repo,
                    since=self.last_github_check,
                    file_extensions=settings.GITHUB_FILE_EXTENSIONS
                )
                
                logger.info(f"Found {len(files)} updated files in {repo}")
                
                # Process each file
                for file_info in files:
                    file_id = f"github:{repo}:{file_info['path']}"
                    path = file_info["path"]

                    # Skip excluded directories
                    if self._should_skip_path(path):
                        logger.debug(f"Skipping excluded path: {path}")
                        continue
                    
                    # Skip if already processed
                    if self._is_processed(file_id):
                        logger.debug(f"Skipping already processed file: {file_id}")
                        continue
                    
                    try:
                        # Download file content
                        content = await self.github_service.get_file_content(
                            repo=repo,
                            path=file_info['path'],
                            ref=file_info.get('ref', 'main')
                        )
                        
                        if content:
                            # Extract filename from path
                            filename = os.path.basename(file_info['path'])
                            
                            # Prepare metadata
                            metadata = {
                                "repository": repo,
                                "path": file_info['path'],
                                "commit_id": file_info.get('commit_id'),
                                "commit_message": file_info.get('commit_message'),
                                "author": file_info.get('author'),
                                "last_modified": file_info.get('last_modified'),
                                "source_url": file_info.get('url')
                            }
                            
                            # Process the file
                            result = await self.process_upload(
                                file_content=content,
                                filename=filename,
                                source="github",
                                metadata=metadata
                            )
                            
                            if result["success"]:
                                # Mark as processed
                                self._mark_as_processed(file_id)
                                logger.info(f"Processed GitHub file: {file_info['path']} from {repo}")
                            else:
                                logger.error(f"Failed to process GitHub file: {file_info['path']} from {repo}")
                    except Exception as e:
                        logger.error(f"Error processing GitHub file {file_info['path']}: {str(e)}")
            
            # Update last check time
            self.last_github_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error fetching from GitHub: {str(e)}")
    
    async def fetch_from_sharepoint(self):
        """
        Fetch documents from configured SharePoint sites
        """
        if not self.sharepoint_service:
            logger.warning("SharePoint service not configured, skipping")
            return
        
        try:
            logger.info(f"Checking SharePoint sites since {self.last_sharepoint_check}")
            
            # Get list of SharePoint sites and document libraries to monitor
            sites = settings.SHAREPOINT_SITES
            
            for site in sites:
                # Get updated files since last check
                files = await self.sharepoint_service.get_updated_files(
                    site_url=site["url"],
                    library=site["library"],
                    since=self.last_sharepoint_check,
                    file_extensions=settings.SHAREPOINT_FILE_EXTENSIONS
                )
                
                logger.info(f"Found {len(files)} updated files in {site['url']}/{site['library']}")
                
                # Process each file
                for file_info in files:
                    file_id = f"sharepoint:{site['url']}:{file_info['path']}"
                    
                    # Skip if already processed
                    if self._is_processed(file_id):
                        logger.debug(f"Skipping already processed file: {file_id}")
                        continue
                    
                    try:
                        # Download file content
                        content = await self.sharepoint_service.get_file_content(
                            site_url=site["url"],
                            path=file_info['path']
                        )
                        
                        if content:
                            # Extract filename from path
                            filename = os.path.basename(file_info['path'])
                            
                            # Prepare metadata
                            metadata = {
                                "site_url": site["url"],
                                "library": site["library"],
                                "path": file_info['path'],
                                "author": file_info.get('author'),
                                "last_modified": file_info.get('last_modified'),
                                "created_date": file_info.get('created_date'),
                                "version": file_info.get('version'),
                                "source_url": file_info.get('url')
                            }
                            
                            # Process the file
                            result = await self.process_upload(
                                file_content=content,
                                filename=filename,
                                source="sharepoint",
                                metadata=metadata
                            )
                            
                            if result["success"]:
                                # Mark as processed
                                self._mark_as_processed(file_id)
                                logger.info(f"Processed SharePoint file: {file_info['path']} from {site['url']}")
                            else:
                                logger.error(f"Failed to process SharePoint file: {file_info['path']} from {site['url']}")
                    except Exception as e:
                        logger.error(f"Error processing SharePoint file {file_info['path']}: {str(e)}")
            
            # Update last check time
            self.last_sharepoint_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error fetching from SharePoint: {str(e)}")
    
    async def check_s3_events(self):
        """
        Check for new files added directly to S3 bucket.
        Supports multiple prefixes (upload/, sharepoint/, github/).
        """
        if not self.s3_service:
            logger.warning("S3 service not configured, skipping")
            return

        try:
            prefixes = os.getenv("S3_MONITORING_PREFIXES", "upload/,sharepoint/,github/").split(",")
            lookback_minutes = settings.S3_MONITORING_LOOKBACK_MINUTES
            total_files_processed = 0

            for prefix in prefixes:
                prefix = prefix.strip()
                if not prefix:
                    continue

                logger.info(f"Checking S3 prefix: {prefix}")

                recent_files = await self.s3_service.list_recent_files(
                    prefix=prefix,
                    minutes=lookback_minutes
                )

                logger.info(f"Found {len(recent_files)} recent files under prefix '{prefix}'")

                for file_info in recent_files:
                    file_id = f"s3:{file_info['key']}"

                    # Skip if already processed
                    if self._is_processed(file_id):
                        continue

                    try:
                        # Skip files with unsupported extensions
                        _, ext = os.path.splitext(file_info["key"])
                        if ext.lower() not in settings.S3_MONITORED_EXTENSIONS:
                            continue

                        content, metadata = await self.s3_service.download_document(file_info["key"])
                        if not content:
                            continue

                        filename = os.path.basename(file_info["key"])

                        # Detect source type from prefix
                        if file_info["key"].startswith("upload/"):
                            source = "manual"
                        elif file_info["key"].startswith("sharepoint/"):
                            source = "sharepoint"
                        elif file_info["key"].startswith("github/"):
                            source = "github"
                        else:
                            source = "s3"

                        result = await self.process_upload(
                            file_content=content,
                            filename=filename,
                            source=source,
                            metadata={
                                "s3_key": file_info["key"],
                                "last_modified": file_info.get("last_modified"),
                                "size": file_info.get("size"),
                                "etag": file_info.get("etag"),
                                **metadata
                            },
                        )

                        if result["success"]:
                            self._mark_as_processed(file_id)
                            total_files_processed += 1
                            logger.info(f"Processed S3 file: {file_info['key']}")
                        else:
                            logger.error(f"Failed to process S3 file: {file_info['key']}")

                    except Exception as e:
                        logger.error(f"Error processing S3 file {file_info['key']}: {str(e)}")

            logger.info(f"Total processed files this cycle: {total_files_processed}")

        except Exception as e:
            logger.error(f"Error checking S3 events: {str(e)}")

    
    async def process_github_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a GitHub webhook event
        """
        if not self.github_service:
            logger.warning("GitHub service not configured, skipping webhook")
            return {"success": False, "error": "GitHub service not configured"}
        
        try:
            # Check event type
            event_type = payload.get("event_type")
            
            if event_type not in ["push", "pull_request"]:
                logger.info(f"Ignoring GitHub webhook event type: {event_type}")
                return {"success": True, "message": f"Ignored event type: {event_type}"}
            
            # Extract repository information
            repository = payload.get("repository", {}).get("full_name")
            
            if not repository:
                return {"success": False, "error": "Missing repository information"}
            
            # Check if this repository is in our monitored list
            if repository not in settings.GITHUB_REPOSITORIES:
                logger.info(f"Ignoring webhook for non-monitored repository: {repository}")
                return {"success": True, "message": f"Repository not monitored: {repository}"}
            
            # Process based on event type
            if event_type == "push":
                # Get modified files from the push event
                commits = payload.get("commits", [])
                modified_files = []
                
                for commit in commits:
                    modified_files.extend(commit.get("added", []))
                    modified_files.extend(commit.get("modified", []))
                
                # Filter by extension
                filtered_files = []
                for file_path in modified_files:
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in settings.GITHUB_FILE_EXTENSIONS:
                        filtered_files.append(file_path)
                
                logger.info(f"Processing {len(filtered_files)} files from GitHub webhook push event")
                
                # Process each file
                results = []
                for file_path in filtered_files:
                    file_id = f"github_webhook:{repository}:{file_path}"
                    
                    # Skip if already processed
                    if self._is_processed(file_id):
                        logger.debug(f"Skipping already processed file: {file_id}")
                        continue
                    
                    try:
                        # Get file content
                        content = await self.github_service.get_file_content(
                            repo=repository,
                            path=file_path
                        )
                        
                        if content:
                            # Extract filename
                            filename = os.path.basename(file_path)
                            
                            # Get commit info for the file
                            commit_info = next(
                                (c for c in commits if file_path in c.get("added", []) + c.get("modified", [])),
                                {}
                            )
                            
                            # Process the file
                            result = await self.process_upload(
                                file_content=content,
                                filename=filename,
                                source="github",
                                metadata={
                                    "repository": repository,
                                    "path": file_path,
                                    "commit_id": commit_info.get("id"),
                                    "commit_message": commit_info.get("message"),
                                    "author": commit_info.get("author", {}).get("name"),
                                    "webhook_event": "push"
                                }
                            )
                            
                            if result["success"]:
                                # Mark as processed
                                self._mark_as_processed(file_id)
                                logger.info(f"Processed GitHub webhook file: {file_path}")
                            
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing GitHub webhook file {file_path}: {str(e)}")
                        results.append({
                            "success": False,
                            "error": str(e),
                            "file": file_path
                        })
                
                return {
                    "success": True,
                    "event_type": "push",
                    "repository": repository,
                    "processed_count": len(results),
                    "results": results
                }
            
            elif event_type == "pull_request":
                # Handle pull request events if needed
                action = payload.get("action")
                
                if action not in ["opened", "synchronize", "reopened"]:
                    return {"success": True, "message": f"Ignored pull request action: {action}"}
                
                # For now, just trigger a full repository check
                # This could be optimized to only check files in the PR
                await self.fetch_from_github()
                
                return {
                    "success": True,
                    "event_type": "pull_request",
                    "repository": repository,
                    "action": action,
                    "message": "Triggered repository check"
                }
        
        except Exception as e:
            logger.error(f"Error processing GitHub webhook: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_sharepoint_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a SharePoint webhook notification
        """
        if not self.sharepoint_service:
            logger.warning("SharePoint service not configured, skipping webhook")
            return {"success": False, "error": "SharePoint service not configured"}
        
        try:
            # Extract site and resource information
            resource = payload.get("resource")
            site_url = payload.get("siteUrl")
            
            if not resource or not site_url:
                return {"success": False, "error": "Missing resource or site URL"}
            
            # Check if this site is in our monitored list
            site_found = False
            for site in settings.SHAREPOINT_SITES:
                if site["url"] == site_url:
                    site_found = True
                    break
            
            if not site_found:
                logger.info(f"Ignoring webhook for non-monitored SharePoint site: {site_url}")
                return {"success": True, "message": f"Site not monitored: {site_url}"}
            
            # For SharePoint webhooks, we typically just get a notification that something changed
            # We need to query for the specific changes
            logger.info(f"Received SharePoint webhook for {site_url}, triggering sync")
            
            # Trigger a sync for this site
            await self.fetch_from_sharepoint()
            
            return {
                "success": True,
                "site_url": site_url,
                "message": "Triggered SharePoint sync"
            }
        
        except Exception as e:
            logger.error(f"Error processing SharePoint webhook: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate MD5 checksum for file content"""
        import hashlib
        return hashlib.md5(content).hexdigest()
