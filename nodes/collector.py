import json
import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from state import BudgetState

logger = logging.getLogger(__name__)

PRIORITY_QUESTION = """Please rank the following spending categories from MOST to LEAST important for your trip:

  1. Hotel       (accommodation)
  2. Food        (meals and drinks)
  3. Activities  (tours, attractions, experiences)
  4. Transport   (getting around)

Type your preferred order, for example:
  • "Hotel, Activities, Food, Transport"
  • "1, 3, 2, 4"
  • "Accommodation first, then sightseeing, then food, transport last"

Your ranking will determine how your budget is split (40% / 30% / 20% / 10%)."""

_VALID_CATEGORIES = {"hotel", "food", "activities", "transport"}
_DEFAULT_PRIORITIES = ["hotel", "food", "activities", "transport"]


def _parse_priorities(user_response: str) -> list[str]:
    """Use LLM to robustly parse any free-form priority input into the canonical 4-item list."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    prompt = f"""Parse this user input into an ordered list of 4 travel spending categories.

User input: "{user_response}"

Map to EXACTLY these category names (lowercase): "hotel", "food", "activities", "transport"

Return JSON:
{{"priorities": ["most_important", "second", "third", "least_important"]}}

Rules:
- All 4 categories must appear exactly once
- Order from most to least important as expressed by the user
- If input is numbers like "1,3,2,4" interpret as: 1=hotel, 2=food, 3=activities, 4=transport
- If input is ambiguous or unclear, default to: ["hotel", "food", "activities", "transport"]"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        priorities = data.get("priorities", [])

        # Validate: must be exactly the 4 known categories
        if isinstance(priorities, list) and set(priorities) == _VALID_CATEGORIES and len(priorities) == 4:
            logger.info(f"Parsed priorities: {priorities}")
            return priorities
    except Exception as e:
        logger.error(f"Priority parsing failed: {e}")

    logger.warning("Could not parse priorities, using default order")
    return list(_DEFAULT_PRIORITIES)


def priority_collector(state: BudgetState) -> dict:
    """Conversational node: ask the user to rank spending categories, then wait for their reply.

    Uses LangGraph interrupt() — the graph pauses here until the user submits a response
    via POST /plan/{session_id}/respond. On resume, interrupt() returns the user's text.
    """
    user_response = interrupt({"question": PRIORITY_QUESTION})
    priorities = _parse_priorities(str(user_response))
    return {"priorities": priorities}
