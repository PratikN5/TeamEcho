from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application Settings
    DEBUG: bool = False
    TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # AWS Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    S3_ENDPOINT_URL: str
    MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"

    # PostgreSQL Settings
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: Optional[str] = ""
    REDIS_DB: int
    VECTOR_DIMENSION: int
    CHAT_MEMORY_TTL: int
    MAX_CHAT_HISTORY: int

    # Ingestion Settings
    INGESTION_POLLING_INTERVAL: int
    INITIAL_LOOKBACK_DAYS: int
    DATA_DIR: str

    # GitHub Integration
    GITHUB_INTEGRATION_ENABLED: bool
    GITHUB_API_TOKEN: str
    GITHUB_REPOSITORIES: List[str]
    GITHUB_FILE_EXTENSIONS: List[str]
    GITHUB_WEBHOOK_SECRET: str

    # SharePoint Integration
    SHAREPOINT_INTEGRATION_ENABLED: bool
    SHAREPOINT_TENANT_ID: str
    SHAREPOINT_CLIENT_ID: str
    SHAREPOINT_CLIENT_SECRET: str
    SHAREPOINT_SITES: List[dict]
    SHAREPOINT_FILE_EXTENSIONS: List[str]

    # S3 Event Monitoring
    S3_EVENT_MONITORING_ENABLED: bool
    S3_MONITORING_PREFIXES: List[str]
    S3_MONITORING_LOOKBACK_MINUTES: int
    S3_MONITORED_EXTENSIONS: List[str]

    # Neo4j Settings
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # Elasticsearch Settings
    ELASTICSEARCH_HOST: str
    ELASTICSEARCH_PORT: int
    ELASTICSEARCH_USERNAME: str
    ELASTICSEARCH_PASSWORD: str
    ELASTICSEARCH_VERIFY_CERTS: bool
    ELASTICSEARCH_CODE_INDEX: str

    # Agent Settings
    INGESTION_AGENT_ENABLED: bool

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()