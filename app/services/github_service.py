# services/github_service.py
import logging
import aiohttp
import base64
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import asyncio


from app.core.config import get_settings

settings = get_settings()


logger = logging.getLogger(__name__)

class GitHubService:
    """
    Service for interacting with GitHub API to fetch repository contents
    """
    
    def __init__(self):
        """Initialize GitHub API client"""
        
        self.api_base_url = "https://api.github.com"
        self.token = settings.GITHUB_API_TOKEN
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.token}" if self.token else None
        }
        # Remove None values from headers
        self.headers = {k: v for k, v in self.headers.items() if v is not None}
        
        # Rate limiting
        self.rate_limit_remaining = 5000  # Default GitHub rate limit
        self.rate_limit_reset = 0
        
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make a request to GitHub API with rate limit handling"""
        # Check rate limits
        if self.rate_limit_remaining <= 5:
            # Wait until rate limit reset
            now = time.time()
            if now < self.rate_limit_reset:
                wait_time = self.rate_limit_reset - now + 1  # Add 1 second buffer
                logger.warning(f"GitHub API rate limit reached. Waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=self.headers, **kwargs) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 5000))
                self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
                
                if response.status == 403 and self.rate_limit_remaining == 0:
                    # Rate limit exceeded
                    reset_time = datetime.fromtimestamp(self.rate_limit_reset)
                    logger.error(f"GitHub API rate limit exceeded. Resets at {reset_time}")
                    raise Exception(f"GitHub API rate limit exceeded. Resets at {reset_time}")
                
                if response.status >= 400:
                    # API error
                    error_text = await response.text()
                    logger.error(f"GitHub API error: {response.status} - {error_text}")
                    raise Exception(f"GitHub API error: {response.status} - {error_text}")
                
                # Return JSON response
                return await response.json()
    
    async def get_updated_files(
        self,
        repo: str,
        since: datetime,
        file_extensions: Optional[List[str]] = None,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Get list of files updated since a specific time
        
        Args:
            repo: Repository name (format: "owner/repo")
            since: Datetime to check updates since
            file_extensions: List of file extensions to include (e.g., [".md", ".txt"])
            branch: Branch to check (default: "main")
            
        Returns:
            List of file information dictionaries
        """
        try:
            # Convert datetime to ISO format for GitHub API
            since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Get commits since the specified time
            commits_url = f"{self.api_base_url}/repos/{repo}/commits"
            commits = await self._make_request(
                "GET",
                commits_url,
                params={"since": since_iso, "sha": branch}
            )
            
            # Track all modified files
            modified_files = []
            seen_files = set()
            
            # Process each commit to extract modified files
            for commit in commits:
                commit_sha = commit["sha"]
                commit_url = f"{self.api_base_url}/repos/{repo}/commits/{commit_sha}"
                commit_detail = await self._make_request("GET", commit_url)
                
                # Extract files modified in this commit
                for file_info in commit_detail.get("files", []):
                    file_path = file_info["filename"]
                    
                    # Skip if already processed
                    if file_path in seen_files:
                        continue
                    
                    # Filter by extension if specified
                    if file_extensions:
                        _, ext = os.path.splitext(file_path)
                        if ext.lower() not in file_extensions:
                            continue
                    
                    # Add to results
                    modified_files.append({
                        "path": file_path,
                        "commit_id": commit_sha,
                        "commit_message": commit["commit"]["message"],
                        "author": commit["commit"]["author"]["name"],
                        "last_modified": commit["commit"]["author"]["date"],
                        "url": f"https://github.com/{repo}/blob/{branch}/{file_path}",
                        "ref": branch
                    })
                    
                    seen_files.add(file_path)
            
            logger.info(f"Found {len(modified_files)} modified files in {repo} since {since_iso}")
            return modified_files
        
        except Exception as e:
            logger.error(f"Error getting updated files from GitHub {repo}: {str(e)}")
            return []
    
    async def get_file_content(
        self,
        repo: str,
        path: str,
        ref: str = "main"
    ) -> Optional[bytes]:
        """
        Get content of a specific file from GitHub
        
        Args:
            repo: Repository name (format: "owner/repo")
            path: File path within the repository
            ref: Branch or commit SHA (default: "main")
            
        Returns:
            File content as bytes, or None if not found
        """
        try:
            # Get file content
            content_url = f"{self.api_base_url}/repos/{repo}/contents/{path}"
            content_data = await self._make_request(
                "GET",
                content_url,
                params={"ref": ref}
            )
            
            # Check if content is too large (GitHub returns a download_url for large files)
            if "content" in content_data:
                # Decode base64 content
                content = base64.b64decode(content_data["content"])
                return content
            elif "download_url" in content_data:
                # Download large file
                async with aiohttp.ClientSession() as session:
                    async with session.get(content_data["download_url"]) as response:
                        if response.status == 200:
                            return await response.read()
                        else:
                            logger.error(f"Error downloading file from {content_data['download_url']}: {response.status}")
                            return None
            else:
                logger.error(f"Unexpected response format for file {path}: {content_data}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting file content from GitHub {repo}/{path}: {str(e)}")
            return None
    
    async def get_repository_structure(
        self,
        repo: str,
        path: str = "",
        ref: str = "main",
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get structure of a repository or directory
        
        Args:
            repo: Repository name (format: "owner/repo")
            path: Directory path within the repository (empty for root)
            ref: Branch or commit SHA (default: "main")
            recursive: Whether to recursively list directories
            
        Returns:
            List of file and directory information
        """
        try:
            # Get contents of directory
            contents_url = f"{self.api_base_url}/repos/{repo}/contents/{path}"
            contents = await self._make_request(
                "GET",
                contents_url,
                params={"ref": ref}
            )
            
            # Process results
            result = []
            
            for item in contents:
                item_info = {
                    "name": item["name"],
                    "path": item["path"],
                    "type": item["type"],
                    "size": item.get("size", 0),
                    "url": item["html_url"],
                    "download_url": item.get("download_url")
                }
                
                result.append(item_info)
                
                # Recursively get subdirectories if requested
                if recursive and item["type"] == "dir":
                    subdir_items = await self.get_repository_structure(
                        repo=repo,
                        path=item["path"],
                        ref=ref,
                        recursive=True
                    )
                    result.extend(subdir_items)
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting repository structure for {repo}/{path}: {str(e)}")
            return []
    
    async def setup_webhook(
        self,
        repo: str,
        webhook_url: str,
        secret: str,
        events: List[str] = ["push", "pull_request"]
    ) -> Dict[str, Any]:
        """
        Set up a webhook for a repository
        
        Args:
            repo: Repository name (format: "owner/repo")
            webhook_url: URL to receive webhook events
            secret: Secret for webhook signature verification
            events: List of events to subscribe to
            
        Returns:
            Webhook creation result
        """
        try:
            # Create webhook
            webhook_data = {
                "name": "web",
                "active": True,
                "events": events,
                "config": {
                    "url": webhook_url,
                    "content_type": "json",
                    "secret": secret,
                    "insecure_ssl": "0"
                }
            }
            
            webhook_url = f"{self.api_base_url}/repos/{repo}/hooks"
            result = await self._make_request("POST", webhook_url, json=webhook_data)
            
            logger.info(f"Created webhook for {repo}")
            return result
        
        except Exception as e:
            logger.error(f"Error setting up webhook for {repo}: {str(e)}")
            raise
