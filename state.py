from typing import TypedDict


class BudgetState(TypedDict, total=False):
    # User inputs
    destination: str           # e.g. "Ubud, Bali" or "Paris" or "Moscow"
    country_code: str          # e.g. "ID", "FR", "RU" — detected from destination
    currency_code: str         # e.g. "IDR", "EUR", "RUB" — detected from destination
    currency_symbol: str       # e.g. "Rp", "€", "₽"
    usd_budget: float          # user's budget in USD
    local_budget: float        # converted budget in local currency
    exchange_rate: float       # live rate at time of conversion
    duration_days: int         # e.g. 2
    duration_nights: int       # e.g. 1

    # Collected during conversation
    priorities: list[str]      # e.g. ["hotel", "food", "activities", "transport"]

    # Budget buckets in both currencies
    buckets_usd: dict          # e.g. {"hotel": 120.0, "food": 60.0, ...}
    buckets_local: dict        # same but in local currency

    # Agent outputs
    hotel_recommendation: dict
    food_recommendation: dict
    activity_recommendation: dict
    transport_recommendation: dict
    free_extras: list[dict]

    # Reconciliation
    total_spent_usd: float
    total_spent_local: float
    remaining_usd: float
    is_overrun: bool
    retry_count: int           # max 3 retries before hard stop

    # Final output
    itinerary: str
    structured_itinerary: dict  # day-by-day JSON with 2-3 options per time slot
    spend_breakdown: dict
    conversion_note: str       # e.g. "$300 USD = €278 EUR (rate: 0.927 · live)"
