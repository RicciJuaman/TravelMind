import logging

from state import BudgetState

logger = logging.getLogger(__name__)


def budget_reconciler(state: BudgetState) -> dict:
    """Sum all agent spending and determine whether the plan is within budget."""
    usd_budget = state.get("usd_budget", 0.0)
    exchange_rate = state.get("exchange_rate", 1.0)

    hotel = state.get("hotel_recommendation") or {}
    food = state.get("food_recommendation") or {}
    activities = state.get("activity_recommendation") or {}
    transport = state.get("transport_recommendation") or {}

    hotel_usd = float(hotel.get("total_usd", 0.0))
    food_usd = float(food.get("total_usd", 0.0))
    activities_usd = float(activities.get("total_usd", 0.0))
    transport_usd = float(transport.get("total_usd", 0.0))

    total_spent_usd = round(hotel_usd + food_usd + activities_usd + transport_usd, 2)
    total_spent_local = round(total_spent_usd * exchange_rate, 2)
    remaining_usd = round(usd_budget - total_spent_usd, 2)
    is_overrun = total_spent_usd > usd_budget

    spend_breakdown = {
        "hotel":      {"usd": hotel_usd,      "local": round(hotel_usd * exchange_rate, 2)},
        "food":       {"usd": food_usd,       "local": round(food_usd * exchange_rate, 2)},
        "activities": {"usd": activities_usd, "local": round(activities_usd * exchange_rate, 2)},
        "transport":  {"usd": transport_usd,  "local": round(transport_usd * exchange_rate, 2)},
    }

    logger.info(
        f"Reconciler: total=${total_spent_usd:.2f} vs budget=${usd_budget:.2f} "
        f"| remaining=${remaining_usd:.2f} | overrun={is_overrun}"
    )

    return {
        "total_spent_usd": total_spent_usd,
        "total_spent_local": total_spent_local,
        "remaining_usd": remaining_usd,
        "is_overrun": is_overrun,
        "spend_breakdown": spend_breakdown,
    }
