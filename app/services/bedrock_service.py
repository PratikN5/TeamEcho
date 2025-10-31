import boto3
import json
import logging
from typing import List, Dict, Any, Optional
from app.core.config import get_settings
 
settings = get_settings()
logger = logging.getLogger(__name__)
 
class BedrockService:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = settings.MODEL_ID
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
   
    async def get_completion(self, message: str) -> str:
        """
        Get completion from Bedrock model using a single message
       
        Args:
            message: User message
           
        Returns:
            Model response text
        """
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message}],
                }
            ],
        }
       
        request = json.dumps(native_request)
       
        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=request
            )
           
            full_response = ""
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk["type"] == "content_block_delta":
                    full_response += chunk["delta"].get("text", "")
           
            return full_response
        except Exception as e:
            logger.error(f"Error in get_completion: {str(e)}")
            raise
   
    async def get_completion_with_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Get completion from Bedrock model using a list of messages
       
        Args:
            messages: List of message objects with role and content
                     Can include a system message with role="system"
           
        Returns:
            Model response text
        """
        # Separate system message from conversation messages
        system_content = None
        conversation_messages = []
       
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
           
            if role == "system":
                # Extract system message as top-level parameter
                system_content = content
            else:
                # Convert content to Anthropic format if it's a string
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
               
                conversation_messages.append({
                    "role": role,
                    "content": content
                })
       
        # Build native request with system as top-level parameter
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation_messages,
        }
       
        # Add system parameter if system message exists
        if system_content:
            native_request["system"] = system_content
       
        request = json.dumps(native_request)
       
        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=request
            )
           
            full_response = ""
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk["type"] == "content_block_delta":
                    full_response += chunk["delta"].get("text", "")
           
            return full_response
        except Exception as e:
            logger.error(f"Error in get_completion_with_messages: {str(e)}")
            raise
   
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using Bedrock embedding model
       
        Args:
            text: Text to embed
           
        Returns:
            List of embedding values or None if error
        """
        try:
            # Use the embedding model specified in settings
            embedding_model = settings.EMBEDDING_MODEL_ID
           
            # Prepare request based on the embedding model
            if "amazon" in embedding_model.lower():
                # Amazon Titan embedding model
                request_body = {
                    "inputText": text
                }
            elif "cohere" in embedding_model.lower():
                # Cohere embedding model
                request_body = {
                    "texts": [text],
                    "input_type": "search_query"
                }
            else:
                # Default format
                request_body = {
                    "input": text
                }
           
            response = self.client.invoke_model(
                modelId=embedding_model,
                body=json.dumps(request_body)
            )
           
            # Parse response based on model
            response_body = json.loads(response["body"].read())
           
            if "amazon" in embedding_model.lower():
                embedding = response_body.get("embedding", [])
            elif "cohere" in embedding_model.lower():
                embedding = response_body.get("embeddings", [[]])[0]
            else:
                embedding = response_body.get("embedding", [])
           
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None