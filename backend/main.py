"""FastAPI Main Server Module for DubizzleBot."""

from fastapi import FastAPI
from backend.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI service exposing endpoints for chat processing, state persistence, and inventory retrieval."
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}
