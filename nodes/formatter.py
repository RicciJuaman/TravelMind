"""Node 12 — itinerary_formatter: structured itinerary agent + plain-text generator."""
import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import BudgetState

logger = logging.getLogger(__name__)

_SEP = "─" * 58


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(content: str, default: dict) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    logger.warning("JSON parse failed in formatter, using default")
    return default


def _fmt_row(label: str, usd: float, sym: str, local_amt: float, code: str) -> str:
    return f"  {label:<14} ${usd:>9.2f} USD  |  {sym}{local_amt:>14,.2f} {code}"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback structured itinerary (built from existing agent outputs)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_structure(state: BudgetState) -> dict:
    """Build a minimal structured itinerary directly from agent output dicts."""
    days = state.get("duration_days", 1)
    destination = state.get("destination", "your destination")
    hotel = state.get("hotel_recommendation") or {}
    food = state.get("food_recommendation") or {}
    activities = state.get("activity_recommendation") or {}
    transport = state.get("transport_recommendation") or {}

    # Hotel options
    hotel_opts = [
        {
            "title": o.get("name", "Accommodation"),
            "price_per_night_usd": o.get("price_per_night_usd", 0),
            "price_per_night_local": o.get("price_per_night_local", 0),
            "vibe": o.get("vibe", "budget"),
            "location": o.get("location", ""),
            "amenities": o.get("amenities", []),
            "best_for": o.get("notes", ""),
        }
        for o in hotel.get("options", [])[:3]
    ]

    # Arrival options
    arr = transport.get("arrival_transfer", {})
    arrival_opts = (
        [{"title": arr.get("method", "Local transport"), "description": arr.get("tip", ""),
          "cost_usd": arr.get("cost_usd", 0), "cost_local": arr.get("cost_local", 0),
          "duration": arr.get("duration", ""), "tip": arr.get("tip", "")}]
        if arr else []
    )

    # Build day slots from food + activity suggestions
    meal_plan = food.get("meal_plan", [])
    act_list = activities.get("activities", [])
    act_map = {a["name"]: a for a in act_list}
    suggestions = activities.get("day_suggestions", [])

    MEAL_TIMES = {"breakfast": "8:00am – 9:00am", "lunch": "12:30pm – 1:30pm", "dinner": "6:30pm – 8:00pm"}
    ACT_TIMES = {"morning": "9:30am – 12:00pm", "afternoon": "2:00pm – 5:00pm", "evening": "7:00pm – 9:00pm"}

    day_structure = []
    for d in range(1, days + 1):
        slots = []
        today_meals = meal_plan[d - 1] if d <= len(meal_plan) else {}
        today_acts = suggestions[d - 1] if d <= len(suggestions) else {}

        for meal_key in ("breakfast", "lunch", "dinner"):
            m = today_meals.get(meal_key, {})
            if m:
                slots.append({
                    "time": MEAL_TIMES[meal_key],
                    "label": meal_key.capitalize(),
                    "options": [{"title": m.get("description", meal_key.capitalize()),
                                 "description": "Local dining option",
                                 "cost_usd": m.get("cost_usd", 0),
                                 "cost_local": m.get("cost_local", 0),
                                 "category": "food", "tip": ""}],
                })
            # Insert activity slot after breakfast and after lunch
            act_period = {"breakfast": "morning", "lunch": "afternoon"}.get(meal_key)
            if act_period and today_acts.get(act_period):
                act_name = today_acts[act_period]
                a = act_map.get(act_name, {})
                if a:
                    slots.append({
                        "time": ACT_TIMES[act_period],
                        "label": f"{act_period.capitalize()} Activity",
                        "options": [{"title": a.get("name", "Local activity"),
                                     "description": a.get("description", ""),
                                     "cost_usd": a.get("cost_usd", 0),
                                     "cost_local": a.get("cost_local", 0),
                                     "category": a.get("type", "activity"), "tip": ""}],
                    })

        day_structure.append({"day": d, "slots": slots})

    return {
        "hotel": {"options": hotel_opts},
        "arrival": {"options": arrival_opts},
        "days": day_structure,
        "local_tips": [
            f"Exchange currency at a reputable bureau in {destination}",
            "Carry small bills for street vendors and markets",
            "Stay hydrated — carry a reusable water bottle",
            "Respect local customs and dress modestly at religious sites",
            "Download offline maps before you go",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Itinerary agent — LLM generates structured JSON with options per slot
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_structured_itinerary(state: BudgetState) -> dict:
    """Ask gpt-4o-mini to build a rich structured itinerary with 2-3 options per time slot."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    destination = state.get("destination", "")
    days = state.get("duration_days", 1)
    nights = state.get("duration_nights", 1)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")

    hotel = state.get("hotel_recommendation") or {}
    food = state.get("food_recommendation") or {}
    activities = state.get("activity_recommendation") or {}
    transport = state.get("transport_recommendation") or {}

    context = {
        "hotel_options": hotel.get("options", [])[:3],
        "hotel_booking_tip": hotel.get("booking_tip", ""),
        "meal_plan": food.get("meal_plan", []),
        "food_local_tips": food.get("local_tips", []),
        "activities": activities.get("activities", []),
        "day_activity_suggestions": activities.get("day_suggestions", []),
        "arrival_transfer": transport.get("arrival_transfer", {}),
        "local_transport_options": transport.get("local_transport", []),
        "recommended_daily_transport": transport.get("recommended_daily", ""),
    }

    prompt = f"""You are an expert travel planner building a detailed itinerary for {destination}.

Trip: {days} day(s), {nights} night(s). Currency: {sym} ({code}).

Data from specialist agents:
{json.dumps(context, indent=2)}

Build a structured day-by-day itinerary. For EVERY time slot, provide 2-3 real OPTIONS so the traveller can choose what suits them.

Return ONLY valid JSON — no markdown, no explanation, just the JSON object:
{{
  "hotel": {{
    "options": [
      {{
        "title": "Exact hotel or guesthouse name",
        "price_per_night_usd": 25.0,
        "price_per_night_local": 1541.0,
        "vibe": "budget",
        "location": "White Beach, Station 3, Boracay",
        "amenities": ["wifi", "pool", "fan room"],
        "best_for": "Budget travellers who want direct beach access"
      }},
      {{ ...second hotel option... }},
      {{ ...third hotel option... }}
    ]
  }},
  "arrival": {{
    "options": [
      {{
        "title": "Transport method name",
        "description": "Step-by-step instructions",
        "cost_usd": 5.0,
        "cost_local": 309.0,
        "duration": "45 min",
        "tip": "Book in advance online to avoid touts"
      }},
      {{ ...second arrival option... }}
    ]
  }},
  "days": [
    {{
      "day": 1,
      "slots": [
        {{
          "time": "8:00am – 9:30am",
          "label": "Breakfast",
          "options": [
            {{
              "title": "Specific restaurant or place name",
              "description": "What it offers and why it is good",
              "cost_usd": 3.0,
              "cost_local": 185.0,
              "category": "food",
              "tip": "Order the house special or arrive early"
            }},
            {{
              "title": "Second breakfast option",
              "description": "...",
              "cost_usd": 2.0,
              "cost_local": 123.0,
              "category": "food",
              "tip": "..."
            }}
          ]
        }},
        {{
          "time": "10:00am – 12:30pm",
          "label": "Morning Activity",
          "options": [
            {{ "title": "Free option name", "description": "...", "cost_usd": 0, "cost_local": 0, "category": "free", "tip": "..." }},
            {{ "title": "Paid option name", "description": "...", "cost_usd": 15.0, "cost_local": 926.0, "category": "activity", "tip": "..." }},
            {{ "title": "Third option", "description": "...", "cost_usd": 8.0, "cost_local": 494.0, "category": "activity", "tip": "..." }}
          ]
        }},
        {{ "time": "12:30pm – 1:30pm", "label": "Lunch", "options": [ ... 2 options ... ] }},
        {{ "time": "2:00pm – 5:00pm", "label": "Afternoon Activity", "options": [ ... 2-3 options ... ] }},
        {{ "time": "5:30pm – 6:30pm", "label": "Sunset / Free Time", "options": [ ... 2 options ... ] }},
        {{ "time": "7:00pm – 8:30pm", "label": "Dinner", "options": [ ... 2 options ... ] }}
      ]
    }}
    {', {{ "day": 2, "slots": [ ... same 6-7 slot structure ... ] }}' if days > 1 else ''}
  ],
  "local_tips": [
    "Specific tip 1 about {destination}",
    "Specific tip 2",
    "Specific tip 3",
    "Specific tip 4",
    "Specific tip 5"
  ]
}}

CRITICAL RULES:
- Include ALL {days} day(s) in the days array
- Each day must have 6-7 time slots covering 8am through 8:30pm
- Each slot must have 2-3 distinct options (not the same place repeated)
- Use REAL, SPECIFIC place names in {destination} — no generic descriptions
- "category" must be exactly one of: food / activity / transport / free
- Costs must be realistic for the budget tier provided by the agents"""

    for attempt in range(2):
        try:
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            data = _parse_json(resp.content, {})
            if isinstance(data.get("days"), list) and len(data["days"]) >= 1:
                logger.info(f"Structured itinerary generated: {len(data['days'])} day(s)")
                return data
            raise ValueError("Missing or empty 'days' array in LLM response")
        except Exception as e:
            logger.error(f"_generate_structured_itinerary attempt {attempt + 1} failed: {e}")

    logger.warning("Falling back to programmatic structured itinerary")
    return _fallback_structure(state)


# ─────────────────────────────────────────────────────────────────────────────
# Text itinerary builder (no LLM — derived from structured data)
# ─────────────────────────────────────────────────────────────────────────────

def _build_text(structured: dict, state: BudgetState) -> str:
    destination = state.get("destination", "")
    usd_budget = state.get("usd_budget", 0.0)
    local_budget = state.get("local_budget", 0.0)
    code = state.get("currency_code", "USD")
    sym = state.get("currency_symbol", "$")
    conversion_note = state.get("conversion_note", "")
    days = state.get("duration_days", 1)
    nights = state.get("duration_nights", 1)
    exchange_rate = state.get("exchange_rate", 1.0)
    total_spent_usd = state.get("total_spent_usd", 0.0)
    total_spent_local = state.get("total_spent_local", 0.0)
    remaining_usd = state.get("remaining_usd", 0.0)
    remaining_local = round(remaining_usd * exchange_rate, 2)
    is_overrun = state.get("is_overrun", False)
    retry_count = state.get("retry_count", 0)
    spend_breakdown = state.get("spend_breakdown") or {}
    free_extras = state.get("free_extras") or []

    lines: list[str] = []

    lines += [
        f"TRAVELMIND — YOUR TRIP TO {destination.upper()}",
        "=" * 58,
        f"Budget:   ${usd_budget:.2f} USD = {sym}{local_budget:,.2f} {code}",
        f"          ({conversion_note})",
        f"Duration: {days} day(s), {nights} night(s)",
    ]

    if is_overrun and retry_count >= 3:
        over = round(total_spent_usd - usd_budget, 2)
        lines.append(f"\n⚠️  BUDGET NOTICE: Spend exceeds budget by ${over:.2f} USD.")

    lines.append("")
    lines.append("BUDGET BREAKDOWN")

    for cat, label in [("hotel", "Hotel:"), ("food", "Food:"),
                        ("activities", "Activities:"), ("transport", "Transport:")]:
        bd = spend_breakdown.get(cat, {})
        lines.append(_fmt_row(label, bd.get("usd", 0), sym, bd.get("local", 0), code))

    lines += [
        _SEP,
        _fmt_row("Total:", total_spent_usd, sym, total_spent_local, code),
        _fmt_row("Remaining:", remaining_usd, sym, remaining_local, code),
        "",
    ]

    # Day-by-day (pick first option per slot as the recommended choice)
    for day_data in structured.get("days", []):
        lines.append(f"DAY {day_data.get('day', '?')}")
        for slot in day_data.get("slots", []):
            opts = slot.get("options", [])
            if opts:
                first = opts[0]
                cost = first.get("cost_usd", 0)
                cost_str = f"${cost:.2f}" if cost > 0 else "Free"
                lines.append(
                    f"  {slot.get('label', '')} ({slot.get('time', '')}): "
                    f"{first.get('title', '')} — {first.get('description', '')} [{cost_str}]"
                )
        lines.append("")

    tips = structured.get("local_tips", [])
    if tips:
        lines.append("LOCAL TIPS")
        for i, tip in enumerate(tips, 1):
            lines.append(f"  {i}. {tip}")
        lines.append("")

    if free_extras:
        lines.append("FREE THINGS TO DO (if time allows)")
        for item in free_extras:
            cost = item.get("cost_usd", 0.0)
            cost_str = "Free" if cost == 0 else f"~${cost:.2f}"
            best_time = item.get("best_time", "")
            time_str = f" · {best_time}" if best_time else ""
            lines.append(
                f"  • {item.get('name', 'Activity')} ({cost_str}{time_str})"
                f" — {item.get('why_worth_it', '')}"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Node 12 — itinerary_formatter
# ─────────────────────────────────────────────────────────────────────────────

async def itinerary_formatter(state: BudgetState) -> dict:
    """Generate structured itinerary (JSON with options) and plain-text summary."""
    import os
    from utils.places import enrich_structured_itinerary

    structured = await _generate_structured_itinerary(state)

    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    destination = state.get("destination", "")
    if api_key and destination:
        logger.info("Enriching itinerary with Google Places photos and Maps links…")
        structured = await enrich_structured_itinerary(structured, destination, api_key)

    text = _build_text(structured, state)
    return {
        "structured_itinerary": structured,
        "itinerary": text,
    }
