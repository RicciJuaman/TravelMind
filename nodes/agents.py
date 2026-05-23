"""Specialist LLM agents: hotel, food, activities, transport, free_extras."""
import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import BudgetState

logger = logging.getLogger(__name__)


def _llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _parse_json(content: str, default: dict) -> dict:
    """Parse JSON from LLM response; extract JSON block if full parse fails."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    logger.warning("JSON parse failed, using safe default")
    return default


# ---------------------------------------------------------------------------
# Hotel agent
# ---------------------------------------------------------------------------

async def hotel_agent(state: BudgetState) -> dict:
    """Recommend 2-3 accommodation options within the hotel bucket."""
    destination = state.get("destination", "")
    nights = state.get("duration_nights", 1)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    bucket_usd = state.get("buckets_usd", {}).get("hotel", 0.0)
    bucket_local = state.get("buckets_local", {}).get("hotel", 0.0)
    ppn_usd = round(bucket_usd / max(nights, 1), 2)
    ppn_local = round(bucket_local / max(nights, 1), 2)

    prompt = f"""You are a travel accommodation expert. Recommend places to stay for:
- Destination: {destination}
- Duration: {nights} night(s)
- Total hotel budget: ${bucket_usd:.2f} USD ({sym}{bucket_local:,.2f} {code})
- Per-night budget: ${ppn_usd:.2f} USD ({sym}{ppn_local:,.2f} {code})

Return valid JSON with 2-3 real accommodation options that fit within the budget:
{{
  "options": [
    {{
      "name": "accommodation name",
      "price_per_night_usd": 45.0,
      "price_per_night_local": 711000,
      "location": "brief location description",
      "vibe": "budget/mid-range/boutique",
      "amenities": ["wifi", "pool", "breakfast included"],
      "notes": "why this is a good pick"
    }}
  ],
  "recommended": {{
    "name": "best value pick name",
    "price_per_night_usd": 45.0,
    "price_per_night_local": 711000,
    "location": "...",
    "vibe": "...",
    "amenities": ["..."],
    "notes": "..."
  }},
  "total_usd": 45.0,
  "total_local": 711000,
  "booking_tip": "practical tip for booking (platform, timing, etc.)"
}}

CRITICAL: total_usd = recommended.price_per_night_usd × {nights} (must not exceed ${bucket_usd:.2f}).
Use real, well-known accommodations in {destination}. Be specific with names."""

    default = {
        "options": [{
            "name": f"Budget guesthouse in {destination}",
            "price_per_night_usd": ppn_usd,
            "price_per_night_local": ppn_local,
            "location": destination,
            "vibe": "budget",
            "amenities": ["wifi"],
            "notes": "Affordable, central option",
        }],
        "recommended": {
            "name": f"Budget guesthouse in {destination}",
            "price_per_night_usd": ppn_usd,
            "price_per_night_local": ppn_local,
            "location": destination,
            "vibe": "budget",
            "amenities": ["wifi"],
            "notes": "Affordable, central option",
        },
        "total_usd": bucket_usd,
        "total_local": bucket_local,
        "booking_tip": "Book in advance for best rates.",
    }

    for attempt in range(2):
        try:
            resp = await _llm().ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, default)
            data.setdefault("total_usd", bucket_usd)
            data.setdefault("total_local", bucket_local)
            return {"hotel_recommendation": data}
        except Exception as e:
            logger.error(f"hotel_agent attempt {attempt + 1} failed: {e}")

    return {"hotel_recommendation": default}


# ---------------------------------------------------------------------------
# Food agent
# ---------------------------------------------------------------------------

async def food_agent(state: BudgetState) -> dict:
    """Plan a day-by-day meal schedule prioritising local cuisine."""
    destination = state.get("destination", "")
    days = state.get("duration_days", 1)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    bucket_usd = state.get("buckets_usd", {}).get("food", 0.0)
    bucket_local = state.get("buckets_local", {}).get("food", 0.0)
    ppd_usd = round(bucket_usd / max(days, 1), 2)
    ppd_local = round(bucket_local / max(days, 1), 2)

    day_entries = "\n".join(
        f'    {{"day": {i + 1}, "breakfast": {{"description": "...", "cost_usd": 0.0, "cost_local": 0}}, '
        f'"lunch": {{"description": "...", "cost_usd": 0.0, "cost_local": 0}}, '
        f'"dinner": {{"description": "...", "cost_usd": 0.0, "cost_local": 0}}}}'
        for i in range(days)
    )

    prompt = f"""You are a local food expert for {destination}. Plan meals for:
- Duration: {days} day(s)
- Total food budget: ${bucket_usd:.2f} USD ({sym}{bucket_local:,.2f} {code})
- Per-day budget: ${ppd_usd:.2f} USD ({sym}{ppd_local:,.2f} {code})

Prioritise LOCAL restaurants, warungs, street food, and markets over tourist spots.

Return valid JSON:
{{
  "meal_plan": [
{day_entries}
  ],
  "local_tips": [
    "tip about eating locally",
    "recommended local dish or restaurant type"
  ],
  "total_usd": {bucket_usd:.2f},
  "total_local": {bucket_local:.2f}
}}

CRITICAL: Fill in all {days} day(s). Sum of all meal costs must approximately equal ${bucket_usd:.2f}.
Name real local dishes and realistic eating spots in {destination}."""

    def _default():
        return {
            "meal_plan": [
                {
                    "day": i + 1,
                    "breakfast": {"description": "Local breakfast spot", "cost_usd": round(ppd_usd * 0.20, 2), "cost_local": round(ppd_local * 0.20, 2)},
                    "lunch": {"description": "Street food", "cost_usd": round(ppd_usd * 0.35, 2), "cost_local": round(ppd_local * 0.35, 2)},
                    "dinner": {"description": "Local restaurant", "cost_usd": round(ppd_usd * 0.45, 2), "cost_local": round(ppd_local * 0.45, 2)},
                }
                for i in range(days)
            ],
            "local_tips": ["Eat where locals eat", "Try street food for best value"],
            "total_usd": bucket_usd,
            "total_local": bucket_local,
        }

    for attempt in range(2):
        try:
            resp = await _llm().ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, _default())
            data.setdefault("total_usd", bucket_usd)
            data.setdefault("total_local", bucket_local)
            return {"food_recommendation": data}
        except Exception as e:
            logger.error(f"food_agent attempt {attempt + 1} failed: {e}")

    return {"food_recommendation": _default()}


# ---------------------------------------------------------------------------
# Activities agent
# ---------------------------------------------------------------------------

async def activities_agent(state: BudgetState) -> dict:
    """Recommend a mix of paid and free activities for each day."""
    destination = state.get("destination", "")
    days = state.get("duration_days", 1)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    bucket_usd = state.get("buckets_usd", {}).get("activities", 0.0)
    bucket_local = state.get("buckets_local", {}).get("activities", 0.0)

    day_suggestions = "\n".join(
        f'    {{"day": {i + 1}, "morning": "activity name", "afternoon": "activity name", "evening": "activity name"}}'
        for i in range(days)
    )

    prompt = f"""You are a local activities expert for {destination}. Recommend experiences for:
- Duration: {days} day(s)
- Total activities budget: ${bucket_usd:.2f} USD ({sym}{bucket_local:,.2f} {code})

Include a MIX of paid and free activities. Return valid JSON:
{{
  "activities": [
    {{
      "name": "specific activity name",
      "cost_usd": 20.0,
      "cost_local": 316000,
      "duration": "2-3 hours",
      "best_time": "morning/afternoon/evening",
      "type": "paid/free",
      "description": "what makes this worth doing"
    }}
  ],
  "day_suggestions": [
{day_suggestions}
  ],
  "total_usd": {bucket_usd:.2f},
  "total_local": {bucket_local:.2f}
}}

CRITICAL: In total_usd, count ONLY paid activities. Total must not exceed ${bucket_usd:.2f}.
Name real, specific activities in {destination} — temples, rice terraces, cooking classes, markets, etc.
Include day_suggestions covering all {days} day(s)."""

    def _default():
        return {
            "activities": [
                {"name": "Local walking tour", "cost_usd": 0.0, "cost_local": 0, "duration": "2 hours", "best_time": "morning", "type": "free", "description": "Explore the area on foot"},
                {"name": "Main local attraction", "cost_usd": round(bucket_usd * 0.7, 2), "cost_local": round(bucket_local * 0.7, 2), "duration": "3 hours", "best_time": "afternoon", "type": "paid", "description": "Top-rated attraction"},
            ],
            "day_suggestions": [{"day": i + 1, "morning": "Walking tour", "afternoon": "Main attraction", "evening": "Sunset viewpoint"} for i in range(days)],
            "total_usd": bucket_usd,
            "total_local": bucket_local,
        }

    for attempt in range(2):
        try:
            resp = await _llm(temperature=0.8).ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, _default())
            data.setdefault("total_usd", bucket_usd)
            data.setdefault("total_local", bucket_local)
            return {"activity_recommendation": data}
        except Exception as e:
            logger.error(f"activities_agent attempt {attempt + 1} failed: {e}")

    return {"activity_recommendation": _default()}


# ---------------------------------------------------------------------------
# Transport agent
# ---------------------------------------------------------------------------

async def transport_agent(state: BudgetState) -> dict:
    """Recommend realistic local transport options including airport transfer."""
    destination = state.get("destination", "")
    days = state.get("duration_days", 1)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    bucket_usd = state.get("buckets_usd", {}).get("transport", 0.0)
    bucket_local = state.get("buckets_local", {}).get("transport", 0.0)

    prompt = f"""You are a local transport expert for {destination}. Recommend how to get around for:
- Duration: {days} day(s)
- Total transport budget: ${bucket_usd:.2f} USD ({sym}{bucket_local:,.2f} {code})

Cover airport/station arrival AND daily local transport. Return valid JSON:
{{
  "arrival_transfer": {{
    "method": "how to get from airport/main station to accommodation",
    "cost_usd": 10.0,
    "cost_local": 158000,
    "duration": "45 minutes",
    "tip": "practical booking or usage advice"
  }},
  "local_transport": [
    {{
      "method": "transport method (e.g. scooter rental, Grab, ojek, metro)",
      "cost_per_day_usd": 5.0,
      "cost_per_day_local": 79000,
      "best_for": "what this is ideal for",
      "tip": "how to use it safely/effectively"
    }}
  ],
  "recommended_daily": "which single transport method is best for daily use",
  "total_usd": {bucket_usd:.2f},
  "total_local": {bucket_local:.2f}
}}

CRITICAL: total_usd = arrival cost + (daily cost × {days} days). Must not exceed ${bucket_usd:.2f}.
Use real transport options specific to {destination}."""

    def _default():
        arrival_usd = round(bucket_usd * 0.3, 2)
        arrival_local = round(bucket_local * 0.3, 2)
        daily_usd = round((bucket_usd - arrival_usd) / max(days, 1), 2)
        daily_local = round((bucket_local - arrival_local) / max(days, 1), 2)
        return {
            "arrival_transfer": {"method": "Public bus or taxi", "cost_usd": arrival_usd, "cost_local": arrival_local, "duration": "varies", "tip": "Negotiate price or use metered taxi"},
            "local_transport": [{"method": "Public transport / walking", "cost_per_day_usd": daily_usd, "cost_per_day_local": daily_local, "best_for": "getting around the city", "tip": "Use local ride-hailing apps"}],
            "recommended_daily": "Public transport",
            "total_usd": bucket_usd,
            "total_local": bucket_local,
        }

    for attempt in range(2):
        try:
            resp = await _llm(temperature=0.6).ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, _default())
            data.setdefault("total_usd", bucket_usd)
            data.setdefault("total_local", bucket_local)
            return {"transport_recommendation": data}
        except Exception as e:
            logger.error(f"transport_agent attempt {attempt + 1} failed: {e}")

    return {"transport_recommendation": _default()}


# ---------------------------------------------------------------------------
# Free extras agent
# ---------------------------------------------------------------------------

async def free_extras_agent(state: BudgetState) -> dict:
    """Recommend free or near-free experiences when budget surplus > $20."""
    destination = state.get("destination", "")
    remaining_usd = state.get("remaining_usd", 0.0)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    exchange_rate = state.get("exchange_rate", 1.0)
    remaining_local = round(remaining_usd * exchange_rate, 2)

    prompt = f"""You are a local guide for {destination}. A traveller has ${remaining_usd:.2f} USD ({sym}{remaining_local:,.2f} {code}) left over.

Suggest 4-6 FREE or near-free extras they can add to their trip.
Focus on: parks, temples, rice terraces, beaches, markets, viewpoints, cultural walks, sunsets.

Return valid JSON:
{{
  "extras": [
    {{
      "name": "specific place or experience name",
      "cost_usd": 0.0,
      "cost_local": 0,
      "why_worth_it": "what makes this unmissable or special",
      "type": "free/near-free",
      "best_time": "morning/afternoon/evening/any time",
      "duration": "1-2 hours"
    }}
  ],
  "total_cost_usd": 0.0,
  "note": "brief note about why these are great additions to the trip"
}}

Name real, specific spots in {destination}. Near-free means under $3 USD."""

    default_extras = [
        {
            "name": f"Local market in {destination}",
            "cost_usd": 0.0,
            "cost_local": 0,
            "why_worth_it": "Experience authentic local culture and pick up cheap snacks",
            "type": "free",
            "best_time": "morning",
            "duration": "1-2 hours",
        },
        {
            "name": "Sunrise/sunset viewpoint",
            "cost_usd": 0.0,
            "cost_local": 0,
            "why_worth_it": "One of the most memorable experiences, completely free",
            "type": "free",
            "best_time": "morning or evening",
            "duration": "1 hour",
        },
    ]

    for attempt in range(2):
        try:
            resp = await _llm(temperature=0.8).ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, {"extras": default_extras})
            extras = data.get("extras", default_extras)
            if not isinstance(extras, list) or len(extras) == 0:
                extras = default_extras
            return {"free_extras": extras}
        except Exception as e:
            logger.error(f"free_extras_agent attempt {attempt + 1} failed: {e}")

    return {"free_extras": default_extras}
