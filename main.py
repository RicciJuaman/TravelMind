"""TravelMind — FastAPI application entry point."""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from graph import build_graph  # noqa: E402 — must come after load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_graph = None
_sessions: dict[str, dict] = {}  # session_id → {"thread_config": ..., "status": ...}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = build_graph()
    logger.info("TravelMind graph ready")
    yield
    logger.info("TravelMind shutting down")


app = FastAPI(
    title="TravelMind",
    description="Multi-agent AI travel budget planner",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    destination: str = Field(..., description="Travel destination, e.g. 'Ubud, Bali'")
    usd_budget: float = Field(..., gt=0, description="Total budget in USD")
    duration_days: int = Field(..., ge=1, description="Number of travel days")
    duration_nights: int = Field(..., ge=0, description="Number of nights")


class RespondRequest(BaseModel):
    response: str = Field(..., description="User's priority ranking response")


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


async def _stream_graph(input_value: Any, thread_config: dict) -> tuple[Any, bool, dict]:
    """Stream the graph until it interrupts or completes.

    Returns:
        (interrupt_value, is_complete, state_values)
    """
    from langgraph.types import Command  # noqa: F401 — needed for isinstance check

    async for _ in _graph.astream(input_value, config=thread_config, stream_mode="updates"):
        pass  # consume the stream; we inspect state afterward

    snapshot = _graph.get_state(thread_config)
    is_complete = len(snapshot.next) == 0

    interrupt_value = None
    if not is_complete:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_value = task.interrupts[0].value
                break

    return interrupt_value, is_complete, snapshot.values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/plan")
async def create_plan(request: PlanRequest):
    """Start a new travel plan session.

    Runs the graph until the priority_collector pause point, then returns the
    session_id and the priority-ranking question to ask the user.
    """
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    initial_state: dict = {
        "destination": request.destination,
        "usd_budget": float(request.usd_budget),
        "duration_days": int(request.duration_days),
        "duration_nights": int(request.duration_nights),
        # Defaults — populated by nodes as the graph runs
        "country_code": "",
        "currency_code": "",
        "currency_symbol": "",
        "local_budget": 0.0,
        "exchange_rate": 1.0,
        "priorities": [],
        "buckets_usd": {},
        "buckets_local": {},
        "hotel_recommendation": {},
        "food_recommendation": {},
        "activity_recommendation": {},
        "transport_recommendation": {},
        "free_extras": [],
        "total_spent_usd": 0.0,
        "total_spent_local": 0.0,
        "remaining_usd": 0.0,
        "is_overrun": False,
        "retry_count": 0,
        "itinerary": "",
        "spend_breakdown": {},
        "conversion_note": "",
    }

    try:
        interrupt_value, is_complete, state_values = await _stream_graph(initial_state, thread_config)
    except Exception as e:
        logger.error(f"Graph start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start plan: {e}")

    _sessions[session_id] = {
        "thread_config": thread_config,
        "status": "complete" if is_complete else "waiting_for_input",
    }

    response: dict = {
        "session_id": session_id,
        "status": "complete" if is_complete else "waiting_for_input",
        "destination": state_values.get("destination", request.destination),
        "currency": (
            f"{state_values.get('currency_symbol', '$')}"
            f"{state_values.get('currency_code', 'USD')}"
        ),
        "conversion_note": state_values.get("conversion_note", ""),
    }

    if interrupt_value and not is_complete:
        response["question"] = interrupt_value.get(
            "question",
            "Please rank: Hotel, Food, Activities, Transport (most → least important)",
        )

    if is_complete:
        response["itinerary"] = state_values.get("itinerary", "")

    return response


@app.post("/plan/{session_id}/respond")
async def respond_to_plan(session_id: str, request: RespondRequest):
    """Resume the graph with the user's priority ranking.

    Once resumed, the graph runs all specialist agents and returns the completed
    itinerary.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session["status"] == "complete":
        raise HTTPException(status_code=400, detail="This plan is already complete")

    thread_config = session["thread_config"]

    from langgraph.types import Command

    try:
        interrupt_value, is_complete, final_state = await _stream_graph(
            Command(resume=request.response), thread_config
        )
    except Exception as e:
        logger.error(f"Graph resume failed for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process response: {e}")

    _sessions[session_id]["status"] = "complete" if is_complete else "waiting_for_input"

    if is_complete:
        return {
            "session_id": session_id,
            "status": "complete",
            "destination": final_state.get("destination", ""),
            "duration_days": final_state.get("duration_days", 1),
            "duration_nights": final_state.get("duration_nights", 1),
            "currency_code": final_state.get("currency_code", "USD"),
            "currency_symbol": final_state.get("currency_symbol", "$"),
            "exchange_rate": final_state.get("exchange_rate", 1.0),
            "usd_budget": final_state.get("usd_budget", 0.0),
            "conversion_note": final_state.get("conversion_note", ""),
            "itinerary": final_state.get("itinerary", ""),
            "structured_itinerary": final_state.get("structured_itinerary", {}),
            "spend_breakdown": final_state.get("spend_breakdown", {}),
            "total_spent_usd": final_state.get("total_spent_usd", 0.0),
            "remaining_usd": final_state.get("remaining_usd", 0.0),
            "free_extras": final_state.get("free_extras", []),
        }

    # Unexpected: another interrupt mid-flow
    return {
        "session_id": session_id,
        "status": "waiting_for_input",
        "question": (
            interrupt_value.get("question", "Please provide additional input")
            if interrupt_value
            else "Please provide additional input"
        ),
    }


@app.get("/plan/{session_id}")
async def get_plan(session_id: str):
    """Return the current state for a session (useful for debugging)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    thread_config = _sessions[session_id]["thread_config"]

    try:
        snapshot = _graph.get_state(thread_config)
    except Exception as e:
        logger.error(f"get_state failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id": session_id,
        "status": _sessions[session_id]["status"],
        "next_nodes": list(snapshot.next),
        "state": snapshot.values,
    }


@app.get("/places/photo")
async def places_photo(ref: str):
    """Proxy Google Places photo so the API key never leaves the server."""
    import os
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key or not ref:
        raise HTTPException(status_code=404, detail="Photo not available")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/photo",
                params={"maxwidth": 600, "photo_reference": ref, "key": api_key},
                follow_redirects=True,
                timeout=10.0,
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Photo not found")
        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(content=resp.content, media_type=content_type)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Places request failed: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": _graph is not None}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
