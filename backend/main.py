"""FastAPI Main Server Module for DubizzleBot."""

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from backend.config import settings
from backend.models.car import CarFilter, CarListing
from backend.services.inventory import InventoryService

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI service exposing endpoints for chat processing, state persistence, and inventory retrieval."
)

inventory_service = InventoryService()

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

@app.post("/inventory/search", response_model=List[CarListing])
async def search_inventory(filters: CarFilter):
    """Deterministic inventory search endpoint filtering dataset cars by criteria."""
    try:
        results = inventory_service.search(filters)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory/summary")
async def inventory_summary():
    """Endpoint returning statistical overview of the loaded car inventory dataset."""
    return inventory_service.get_summary_statistics()
