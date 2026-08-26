"""FastAPI Main Server Module for DubizzleBot."""

import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException

from backend.config import settings
from backend.models.car import CarFilter, CarListing
from backend.models.chat import ChatRequest, ChatResponse
from backend.models.intent import UserIntentEnum
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.booking import BookingService
from backend.services.lead import LeadService
from backend.services.phase5_resolver import Phase5Resolver
from backend.services.query_interpreter import QueryInterpreter
from backend.services.orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

# Lazy singleton references to avoid import-time database side effects
_inventory_service: Optional[InventoryService] = None
_memory_service: Optional[MemoryService] = None
_persistent_memory: Optional[PersistentMemoryService] = None
_booking_service: Optional[BookingService] = None
_lead_service: Optional[LeadService] = None
_chat_orchestrator: Optional[ChatOrchestrator] = None

def get_inventory_service() -> InventoryService:
    global _inventory_service
    if _inventory_service is None:
        _inventory_service = InventoryService()
    return _inventory_service

def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service

def get_persistent_memory() -> PersistentMemoryService:
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemoryService()
    return _persistent_memory

def get_booking_service() -> BookingService:
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService(persistent_memory=get_persistent_memory())
    return _booking_service

def get_lead_service() -> LeadService:
    global _lead_service
    if _lead_service is None:
        _lead_service = LeadService()
    return _lead_service

def get_chat_orchestrator() -> ChatOrchestrator:
    global _chat_orchestrator
    if _chat_orchestrator is None:
        _chat_orchestrator = ChatOrchestrator(
            inventory_service=get_inventory_service(),
            memory_service=get_memory_service(),
            persistent_memory=get_persistent_memory(),
            booking_service=get_booking_service(),
            lead_service=get_lead_service(),
        )
    return _chat_orchestrator

class _LazyServiceProxy:
    """Proxy object ensuring module-level attributes dynamically resolve to initialized singletons."""
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name):
        return getattr(self._getter(), name)

    def __setattr__(self, name, value):
        if name == "_getter":
            super().__setattr__(name, value)
        else:
            setattr(self._getter(), name, value)

# Module-level exports for backward compatibility with existing tests
inventory_service = _LazyServiceProxy(get_inventory_service)
memory_service = _LazyServiceProxy(get_memory_service)
persistent_memory = _LazyServiceProxy(get_persistent_memory)
booking_service = _LazyServiceProxy(get_booking_service)
lead_service = _LazyServiceProxy(get_lead_service)
chat_orchestrator = _LazyServiceProxy(get_chat_orchestrator)

def create_app(
    db_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    query_interpreter: Optional[QueryInterpreter] = None,
    inventory_service_inst: Optional[InventoryService] = None,
    memory_service_inst: Optional[MemoryService] = None,
    persistent_memory_inst: Optional[PersistentMemoryService] = None,
    booking_service_inst: Optional[BookingService] = None,
    lead_service_inst: Optional[LeadService] = None,
    orchestrator_inst: Optional[ChatOrchestrator] = None,
) -> FastAPI:
    """Application factory providing dependency injection for test isolation and production execution."""
    app_instance = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="FastAPI service exposing endpoints for chat processing, state persistence, and inventory retrieval."
    )

    def _resolve_orch():
        if orchestrator_inst:
            return orchestrator_inst
        if db_path or csv_path or query_interpreter or persistent_memory_inst or memory_service_inst or inventory_service_inst or booking_service_inst or lead_service_inst:
            inv = inventory_service_inst or get_inventory_service()
            mem = memory_service_inst or get_memory_service()
            pmem = persistent_memory_inst or (PersistentMemoryService(db_path=db_path) if db_path else get_persistent_memory())
            book = booking_service_inst or (BookingService(db_path=db_path, persistent_memory=pmem) if db_path else get_booking_service())
            lead = lead_service_inst or (LeadService(csv_path=csv_path) if csv_path else get_lead_service())
            interp = query_interpreter or QueryInterpreter()
            return ChatOrchestrator(
                query_interpreter=interp,
                inventory_service=inv,
                memory_service=mem,
                persistent_memory=pmem,
                booking_service=book,
                lead_service=lead,
            )
        return get_chat_orchestrator()

    @app_instance.get("/health")
    async def health_check():
        """Health check endpoint to verify backend status."""
        return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

    @app_instance.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """Processes natural language chat requests with grounded intent interpretation and inventory retrieval."""
        orch = _resolve_orch()
        try:
            response = orch.process_chat(request)
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

    @app_instance.post("/inventory/search", response_model=List[CarListing])
    async def search_inventory(filters: CarFilter):
        """Deterministic inventory search endpoint filtering dataset cars by criteria."""
        inv = inventory_service_inst or get_inventory_service()
        try:
            results = inv.search(filters)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app_instance.get("/inventory/summary")
    async def inventory_summary():
        """Endpoint returning statistical overview of the loaded car inventory dataset."""
        inv = inventory_service_inst or get_inventory_service()
        return inv.get_summary_statistics()

    return app_instance

# Default application instance
app = create_app()
