from fastapi import FastAPI
from app.api.v1.endpoints import chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Genovate API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])