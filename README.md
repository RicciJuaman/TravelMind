# TravelMind

Multi-agent AI travel budget planner built with LangGraph and FastAPI.

## How it works

1. **POST /plan** — provide destination, budget (USD), and trip duration. TravelMind detects the local currency, fetches the live exchange rate, and asks you to rank your spending priorities.
2. **POST /plan/{session_id}/respond** — submit your priority ranking (e.g. `"Hotel, Activities, Food, Transport"`). The graph runs 5 specialist agents in sequence, reconciles the budget, re-allocates if over-budget (up to 3 times), and surfaces free extras if a surplus remains.
3. **GET /plan/{session_id}** — inspect the full session state at any point.

## Agent graph

```
START
  → destination_detector   (LLM: detect currency from destination)
  → currency_converter      (httpx: live rate from open.er-api.com)
  → priority_collector      ← conversational pause / user input
  → budget_allocator        (40% / 30% / 20% / 10% by priority)
  → hotel_agent
  → food_agent
  → activities_agent
  → transport_agent
  → budget_reconciler
  → [conditional router]
      overrun + retry < 3   → re_allocator → budget_allocator (loop)
      overrun + retry >= 3  → itinerary_formatter (with warning)
      surplus > $20         → free_extras_agent → itinerary_formatter
      clean                 → itinerary_formatter
  → END
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI key to .env
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Start the server
uvicorn main:app --reload
```

## Example — Ubud, Bali ($300, 2 days / 1 night)

```bash
# Step 1: Start a plan
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"destination": "Ubud, Bali", "usd_budget": 300, "duration_days": 2, "duration_nights": 1}'

# Response includes session_id and priority question

# Step 2: Submit priorities
curl -X POST http://localhost:8000/plan/{session_id}/respond \
  -H "Content-Type: application/json" \
  -d '{"response": "Hotel, Activities, Food, Transport"}'

# Response includes the full itinerary
```

## Budget allocation

| Priority | Weight | Example ($300) |
|----------|--------|----------------|
| 1st      | 40%    | $120           |
| 2nd      | 30%    | $90            |
| 3rd      | 20%    | $60            |
| 4th      | 10%    | $30            |

## Tech stack

- **Python 3.11**
- **LangGraph** — agent orchestration + MemorySaver checkpointing
- **LangChain + langchain-openai** — LLM wrappers (gpt-4o-mini)
- **FastAPI + uvicorn** — REST API
- **httpx** — live currency conversion
- **Pydantic** — request/response validation
