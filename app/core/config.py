from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()