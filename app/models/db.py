# models/database.py (continued)
from app.core.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


settings = get_settings()

# Database URL
DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency for FastAPI
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    import pkgutil
    import importlib
    import app.models as models_pkg

    for finder, name, ispkg in pkgutil.walk_packages(models_pkg.__path__, models_pkg.__name__ + "."):
        importlib.import_module(name)
    Base.metadata.create_all(bind=engine)

# Drop all tables (for development)
def drop_tables():
    Base.metadata.drop_all(bind=engine)
