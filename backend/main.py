"""FastAPI Main Server Module for DubizzleBot."""

import uuid
import logging
from fastapi import FastAPI, HTTPException
from typing import List
from backend.config import settings
from backend.models.car import CarFilter, CarListing
from backend.models.chat import ChatRequest, ChatResponse
from backend.models.intent import UserIntentEnum
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI service exposing endpoints for chat processing, state persistence, and inventory retrieval."
)

inventory_service = InventoryService()
memory_service = MemoryService()
chat_orchestrator = ChatOrchestrator(
    inventory_service=inventory_service,
    memory_service=memory_service
)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Processes natural language chat requests with grounded intent interpretation and inventory retrieval."""
    try:
        response = chat_orchestrator.process_chat(request)
        return response
    except Exception as e:
        logger.error(f"Endpoint error in /chat: {e}", exc_info=True)
        return ChatResponse(
            user_id=request.user_id,
            session_id=request.session_id or str(uuid.uuid4()),
            response="I apologize, but an unexpected error occurred. Please try again.",
            matched_cars=None,
            intent=UserIntentEnum.UNKNOWN,
            total_matches=0,
            requires_clarification=False
        )

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
