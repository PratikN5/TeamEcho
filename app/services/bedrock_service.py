import boto3
import json
from app.core.config import get_settings

settings = get_settings()

class BedrockService:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.model_id = settings.MODEL_ID

    async def get_completion(self, message: str) -> str:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message}],
                }
            ],
        }

        request = json.dumps(native_request)
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