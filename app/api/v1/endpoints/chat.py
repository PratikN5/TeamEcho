from fastapi import APIRouter, HTTPException
from app.services.bedrock_service import BedrockService
from typing import Dict

router = APIRouter()
bedrock_service = BedrockService()

@router.post("/chat", response_model=Dict[str, str])
async def chat_with_claude(message: str):
    try:
        response = await bedrock_service.get_completion(message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))