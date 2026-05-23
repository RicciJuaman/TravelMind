"""LangGraph graph definition — wires all 12 nodes with conditional routing."""
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.agents import (
    activities_agent,
    food_agent,
    free_extras_agent,
    hotel_agent,
    transport_agent,
)
from nodes.allocator import budget_allocator, re_allocator
from nodes.collector import priority_collector
from nodes.destination import currency_converter, destination_detector
from nodes.formatter import itinerary_formatter
from nodes.reconciler import budget_reconciler
from state import BudgetState

logger = logging.getLogger(__name__)


def route_after_reconciliation(state: BudgetState) -> str:
    """Conditional router: decides the next node after budget_reconciler runs."""
    is_overrun = state.get("is_overrun", False)
    retry_count = state.get("retry_count", 0)
    remaining_usd = state.get("remaining_usd", 0.0)

    if is_overrun and retry_count < 3:
        logger.info(f"Budget overrun — retrying (attempt {retry_count + 1}/3)")
        return "re_allocator"

    if is_overrun and retry_count >= 3:
        logger.warning("Max retries reached — forcing itinerary output with overrun warning")
        return "itinerary_formatter"

    if remaining_usd > 20:
        logger.info(f"Budget surplus ${remaining_usd:.2f} — adding free extras")
        return "free_extras_agent"

    return "itinerary_formatter"


def build_graph():
    """Build and compile the TravelMind LangGraph with MemorySaver checkpointing."""
    builder = StateGraph(BudgetState)

    # ── Register all nodes ────────────────────────────────────────────────
    builder.add_node("destination_detector", destination_detector)
    builder.add_node("currency_converter", currency_converter)
    builder.add_node("priority_collector", priority_collector)
    builder.add_node("budget_allocator", budget_allocator)
    builder.add_node("hotel_agent", hotel_agent)
    builder.add_node("food_agent", food_agent)
    builder.add_node("activities_agent", activities_agent)
    builder.add_node("transport_agent", transport_agent)
    builder.add_node("budget_reconciler", budget_reconciler)
    builder.add_node("re_allocator", re_allocator)
    builder.add_node("free_extras_agent", free_extras_agent)
    builder.add_node("itinerary_formatter", itinerary_formatter)

    # ── Main sequential flow ──────────────────────────────────────────────
    builder.add_edge(START, "destination_detector")
    builder.add_edge("destination_detector", "currency_converter")
    builder.add_edge("currency_converter", "priority_collector")   # ← conversational pause
    builder.add_edge("priority_collector", "budget_allocator")
    builder.add_edge("budget_allocator", "hotel_agent")
    builder.add_edge("hotel_agent", "food_agent")
    builder.add_edge("food_agent", "activities_agent")
    builder.add_edge("activities_agent", "transport_agent")
    builder.add_edge("transport_agent", "budget_reconciler")

    # ── Conditional routing after reconciliation ──────────────────────────
    builder.add_conditional_edges(
        "budget_reconciler",
        route_after_reconciliation,
        {
            "re_allocator": "re_allocator",
            "free_extras_agent": "free_extras_agent",
            "itinerary_formatter": "itinerary_formatter",
        },
    )

    # ── Loop: re_allocator feeds back into budget_allocator ───────────────
    builder.add_edge("re_allocator", "budget_allocator")

    # ── Free extras then formats ──────────────────────────────────────────
    builder.add_edge("free_extras_agent", "itinerary_formatter")

    # ── Formatter terminates ─────────────────────────────────────────────
    builder.add_edge("itinerary_formatter", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("TravelMind graph compiled successfully")
    return graph
