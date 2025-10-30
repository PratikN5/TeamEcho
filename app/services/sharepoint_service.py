# services/sharepoint_service.py
import logging
import aiohttp
import base64
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import msal
from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

class SharePointService:
    """
    Service for interacting with SharePoint via Microsoft Graph API
    """
    
    def __init__(self):
        """Initialize SharePoint API client"""
        self.tenant_id = settings.SHAREPOINT_TENANT_ID
        self.client_id = settings.SHAREPOINT_CLIENT_ID
        self.client_secret = settings.SHAREPOINT_CLIENT_SECRET
        self.scopes = ["https://graph.microsoft.com/.default"]
        # Auth token cache
        self._access_token = None
        self._token_expires_at = 0
        
        # API base URL
        self.api_base_url = "https://graph.microsoft.com/v1.0"
    
    async def _get_access_token(self) -> str:
      now = datetime.utcnow().timestamp()
      if self._access_token and now < self._token_expires_at - 60:
          return self._access_token

      try:
          app = msal.ConfidentialClientApplication(
              client_id=self.client_id,
              authority=f"https://login.microsoftonline.com/{self.tenant_id}",
              client_credential=self.client_secret
          )
          result = app.acquire_token_for_client(scopes=self.scopes)

          if not result or "access_token" not in result:
              logger.error(f"Failed to acquire token: {result}")
              raise Exception("Access token not acquired from MSAL")

          self._access_token = result["access_token"]
          self._token_expires_at = now + result.get("expires_in", 3599)
          logger.debug(f"Acquired new access token: {self._access_token[:10]}...")  # show first 10 chars
          return self._access_token

      except Exception as e:
          logger.error(f"Error acquiring SharePoint access token: {str(e)}")
          raise

    
    async def _make_request(self, method: str, url: str, **kwargs) -> Any:
        """Make a request to Microsoft Graph API"""
        try:
            # Get access token
            token = await self._get_access_token()
            
            # Set authorization header
            headers = kwargs.get("headers", {})
            headers["Authorization"] = f"Bearer {token}"
            kwargs["headers"] = headers
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    if response.status >= 400:
                        # API error
                        error_text = await response.text()
                        logger.error(f"SharePoint API error: {response.status} - {error_text}")
                        raise Exception(f"SharePoint API error: {response.status} - {error_text}")
                    
                    # Return JSON response
                    return await response.json()
        
        except Exception as e:
            logger.error(f"Error making SharePoint API request to {url}: {str(e)}")
            raise
    
    async def get_updated_files(
      self,
      site_url: str,
      library: str,
      since: datetime,
      file_extensions: Optional[List[str]] = None
  ) -> List[Dict[str, Any]]:
      try:
          since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
          site_id = await self._get_site_id(site_url)
          drive_id = await self._get_drive_id(site_id, library)
          
          # Use the search endpoint to recursively get all files
          search_url = f"{self.api_base_url}/drives/{drive_id}/root/search(q='')"
          query_params = {
              "$select": "name,webUrl,lastModifiedDateTime,createdDateTime,size,id,file,parentReference",
              "$top": 999
          }
          
          result = await self._make_request("GET", search_url, params=query_params)
          
          files = []
          for item in result.get("value", []):
              # Skip folders
              if "file" not in item:
                  continue
              
              # Check lastModifiedDateTime
              last_modified = item.get("lastModifiedDateTime")
              if not last_modified or last_modified < since_iso:
                  continue
              
              # Get full path
              file_path = item.get("parentReference", {}).get("path", "").replace("/drive/root:", "") + "/" + item["name"]
              
              # Filter by extensions
              if file_extensions:
                _, ext = os.path.splitext(item["name"])
                if ext.lower() not in [e.lower() for e in file_extensions]:
                    continue

              
              files.append({
                  "path": file_path,
                  "name": item["name"],
                  "id": item["id"],
                  "last_modified": item["lastModifiedDateTime"],
                  "created_date": item["createdDateTime"],
                  "size": item.get("size", 0),
                  "url": item["webUrl"],
                  "drive_id": drive_id
              })
          
          logger.info(f"Found {len(files)} updated files in {site_url}/{library} since {since_iso}")
          return files
      
      except Exception as e:
          logger.error(f"Error getting updated files from SharePoint {site_url}/{library}: {str(e)}")
          return []

    async def get_file_content(
        self,
        site_url: str,
        path: str,
        drive_id: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Get content of a specific file from SharePoint
        """
        try:
            # Get site ID
            site_id = await self._get_site_id(site_url)

            # If drive_id not provided, use default library
            if not drive_id:
                # Extract parts of path
                parts = path.strip("/").split("/")

                # If path starts with "drives", skip "drives" and drive id
                if parts[0].lower() == "drives":
                    parts = parts[2:]  # skip "drives" and drive id

                # Default library is usually 'Documents'
                library = "Documents"

                drive_id = await self._get_drive_id(site_id, library)

                # Remaining path relative to library
                relative_path = "/".join(parts)
            else:
                relative_path = path.strip("/")

            relative_path = relative_path.lstrip("/").replace("root:/", "")
            download_url = f"{self.api_base_url}/drives/{drive_id}/root:/{relative_path}:/content"

            token = await self._get_access_token()

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {token}"}
                async with session.get(download_url, headers=headers) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Error downloading file from SharePoint: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error getting file content from SharePoint {site_url}/{path}: {str(e)}")
            return None

    
    async def _get_site_id(self, site_url: str) -> str:
        """
        Get SharePoint site ID from URL
        
        Args:
            site_url: SharePoint site URL or ID
            
        Returns:
            Site ID
        """
        # If it looks like an ID already, return it
        if site_url.startswith(self.tenant_id):
            return site_url
        
        try:
            # Extract hostname and site path
            if site_url.startswith("https://"):
                # Full URL format
                parts = site_url.split("/")
                hostname = parts[2]
                site_path = "/".join(parts[3:]) if len(parts) > 3 else ""
            else:
                # Just the hostname and maybe path
                parts = site_url.split("/")
                hostname = parts[0]
                site_path = "/".join(parts[1:]) if len(parts) > 1 else ""
            
            # Get site ID
            if site_path:
                site_url = f"{self.api_base_url}/sites/{hostname}:/{site_path}"
            else:
                site_url = f"{self.api_base_url}/sites/{hostname}"
            
            result = await self._make_request("GET", site_url)
            
            return result["id"]
        
        except Exception as e:
            logger.error(f"Error getting SharePoint site ID for {site_url}: {str(e)}")
            raise
    
    async def _get_drive_id(self, site_id: str, library: str) -> str:
        """
        Get drive ID for a document library
        
        Args:
            site_id: SharePoint site ID
            library: Document library name
            
        Returns:
            Drive ID
        """
        try:
            # Get drives in the site
            drives_url = f"{self.api_base_url}/sites/{site_id}/drives"
            result = await self._make_request("GET", drives_url)
            
            # Find the drive with matching name
            for drive in result.get("value", []):
                if drive["name"] == library:
                    return drive["id"]
            
            # If not found, throw error
            raise Exception(f"Document library '{library}' not found in site")
        
        except Exception as e:
            logger.error(f"Error getting drive ID for {site_id}/{library}: {str(e)}")
            raise
    
    async def setup_webhook(
        self,
        site_url: str,
        library: str,
        webhook_url: str,
        expiration_days: int = 30
    ) -> Dict[str, Any]:
        """
        Set up a webhook for a SharePoint document library
        
        Args:
            site_url: SharePoint site URL or ID
            library: Document library name
            webhook_url: URL to receive webhook events
            expiration_days: Number of days until webhook expires
            
        Returns:
            Webhook creation result
        """
        try:
            # Get site ID if URL is provided
            site_id = await self._get_site_id(site_url)
            
            # Get drive ID for the document library
            drive_id = await self._get_drive_id(site_id, library)
            
            # Calculate expiration date
            expiration = datetime.utcnow() + timedelta(days=expiration_days)
            expiration_iso = expiration.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Create webhook
            webhook_data = {
                "changeType": "updated,deleted",
                "notificationUrl": webhook_url,
                "resource": f"drives/{drive_id}/root",
                "expirationDateTime": expiration_iso,
                "clientState": "SharePointWebhook"
            }
            
            subscription_url = f"{self.api_base_url}/subscriptions"
            result = await self._make_request("POST", subscription_url, json=webhook_data)
            
            logger.info(f"Created webhook for {site_url}/{library}")
            return result
        
        except Exception as e:
            logger.error(f"Error setting up webhook for {site_url}/{library}: {str(e)}")
            raise
