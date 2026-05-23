import logging

from state import BudgetState

logger = logging.getLogger(__name__)

# Priority 1 gets 40%, priority 2 gets 30%, priority 3 gets 20%, priority 4 gets 10%
PRIORITY_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
_DEFAULT_PRIORITIES = ["hotel", "food", "activities", "transport"]


def budget_allocator(state: BudgetState) -> dict:
    """Deterministically split usd_budget and local_budget into weighted buckets by priority."""
    priorities = state.get("priorities") or list(_DEFAULT_PRIORITIES)
    usd_budget = state.get("usd_budget", 0.0)
    local_budget = state.get("local_budget", 0.0)

    buckets_usd: dict = {}
    buckets_local: dict = {}

    for i, category in enumerate(priorities):
        weight = PRIORITY_WEIGHTS[i] if i < len(PRIORITY_WEIGHTS) else 0.0
        buckets_usd[category] = round(usd_budget * weight, 2)
        buckets_local[category] = round(local_budget * weight, 2)

    logger.info(
        f"Budget allocated: USD {usd_budget:.2f} → "
        + ", ".join(f"{k}=${v:.2f}" for k, v in buckets_usd.items())
    )
    return {"buckets_usd": buckets_usd, "buckets_local": buckets_local}


def re_allocator(state: BudgetState) -> dict:
    """Trim 15% off the lowest-priority bucket, reducing total spend ceiling.

    The trimmed amount is simply removed (not redistributed) so that budget_allocator
    produces tighter buckets on the next pass, forcing specialist agents to recommend
    cheaper options.
    """
    priorities = state.get("priorities") or list(_DEFAULT_PRIORITIES)
    usd_budget = state.get("usd_budget", 0.0)
    local_budget = state.get("local_budget", 0.0)
    exchange_rate = state.get("exchange_rate", 1.0)
    retry_count = state.get("retry_count", 0)
    buckets_usd = state.get("buckets_usd", {})

    lowest_priority = priorities[-1] if priorities else "transport"
    lowest_bucket_usd = buckets_usd.get(lowest_priority, 0.0)
    trim_usd = round(lowest_bucket_usd * 0.15, 2)

    new_usd_budget = round(usd_budget - trim_usd, 2)
    new_local_budget = round(new_usd_budget * exchange_rate, 2)
    new_retry_count = retry_count + 1

    logger.info(
        f"re_allocator (attempt {new_retry_count}/3): trimming ${trim_usd:.2f} "
        f"from '{lowest_priority}' bucket. New budget: ${new_usd_budget:.2f}"
    )

    return {
        "usd_budget": new_usd_budget,
        "local_budget": new_local_budget,
        "retry_count": new_retry_count,
    }
