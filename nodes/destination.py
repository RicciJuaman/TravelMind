import json
import logging

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import BudgetState

logger = logging.getLogger(__name__)

# Approximate fallback rates vs USD (used when live API is unavailable)
FALLBACK_RATES: dict[str, float] = {
    "IDR": 15800.0,
    "EUR": 0.92,
    "JPY": 149.0,
    "RUB": 90.0,
    "GBP": 0.79,
    "AUD": 1.53,
    "CAD": 1.36,
    "SGD": 1.34,
    "THB": 35.5,
    "MYR": 4.72,
    "PHP": 56.5,
    "VND": 24500.0,
    "KRW": 1320.0,
    "INR": 83.5,
    "BRL": 4.97,
    "MXN": 17.2,
    "TRY": 32.5,
    "ZAR": 18.7,
    "AED": 3.67,
    "SAR": 3.75,
    "CHF": 0.90,
    "SEK": 10.5,
    "NOK": 10.6,
    "DKK": 6.9,
    "PLN": 4.0,
    "CZK": 23.0,
    "HUF": 360.0,
    "CNY": 7.24,
    "HKD": 7.82,
    "TWD": 31.8,
    "NZD": 1.62,
    "MXN": 17.2,
    "COP": 3950.0,
    "ARS": 870.0,
    "CLP": 950.0,
    "PEN": 3.8,
    "EGP": 31.0,
    "MAD": 10.0,
    "KES": 130.0,
    "NGN": 1500.0,
    "GHS": 12.5,
    "TZS": 2500.0,
    "ETB": 56.0,
    "PKR": 278.0,
    "BDT": 110.0,
    "LKR": 325.0,
    "NPR": 133.0,
    "MMK": 2100.0,
    "KHR": 4100.0,
    "LAK": 21000.0,
    "USD": 1.0,
}


def destination_detector(state: BudgetState) -> dict:
    """Use LLM to detect country code, currency code, and symbol from destination."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    destination = state.get("destination", "")

    prompt = f"""Detect the country and currency for this travel destination: "{destination}"

Return JSON with exactly these three fields:
{{
    "country_code": "2-letter ISO 3166-1 alpha-2 country code",
    "currency_code": "3-letter ISO 4217 currency code",
    "currency_symbol": "local currency symbol character(s)"
}}

Examples:
- "Ubud, Bali"      → {{"country_code": "ID", "currency_code": "IDR", "currency_symbol": "Rp"}}
- "Paris"           → {{"country_code": "FR", "currency_code": "EUR", "currency_symbol": "€"}}
- "Tokyo"           → {{"country_code": "JP", "currency_code": "JPY", "currency_symbol": "¥"}}
- "Moscow"          → {{"country_code": "RU", "currency_code": "RUB", "currency_symbol": "₽"}}
- "New York"        → {{"country_code": "US", "currency_code": "USD", "currency_symbol": "$"}}
- "Bangkok"         → {{"country_code": "TH", "currency_code": "THB", "currency_symbol": "฿"}}
- "Sydney"          → {{"country_code": "AU", "currency_code": "AUD", "currency_symbol": "A$"}}
- "Dubai"           → {{"country_code": "AE", "currency_code": "AED", "currency_symbol": "د.إ"}}
- "Seoul"           → {{"country_code": "KR", "currency_code": "KRW", "currency_symbol": "₩"}}
- "Mumbai"          → {{"country_code": "IN", "currency_code": "INR", "currency_symbol": "₹"}}

Always return valid JSON. Use the primary local currency of the destination country."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        country_code = data.get("country_code", "US")
        currency_code = data.get("currency_code", "USD")
        currency_symbol = data.get("currency_symbol", "$")
        logger.info(f"Detected: {destination} → {country_code}, {currency_code}, {currency_symbol}")
        return {
            "country_code": country_code,
            "currency_code": currency_code,
            "currency_symbol": currency_symbol,
        }
    except Exception as e:
        logger.error(f"destination_detector failed: {e}. Defaulting to USD.")
        return {
            "country_code": "US",
            "currency_code": "USD",
            "currency_symbol": "$",
        }


async def currency_converter(state: BudgetState) -> dict:
    """Fetch live USD exchange rate and convert the user's budget to local currency."""
    currency_code = state.get("currency_code", "USD")
    usd_budget = state.get("usd_budget", 0.0)
    symbol = state.get("currency_symbol", "$")

    if currency_code == "USD":
        return {
            "exchange_rate": 1.0,
            "local_budget": usd_budget,
            "conversion_note": f"${usd_budget:.2f} USD (destination uses USD, no conversion needed)",
        }

    # Attempt live rate from open.er-api.com
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://open.er-api.com/v6/latest/USD")
            resp.raise_for_status()
            data = resp.json()

        if data.get("result") == "success" and currency_code in data.get("rates", {}):
            rate = float(data["rates"][currency_code])
            local_budget = round(usd_budget * rate, 2)
            conversion_note = (
                f"${usd_budget:.2f} USD = {symbol}{local_budget:,.2f} {currency_code} "
                f"(rate: {rate:.4f} · live)"
            )
            logger.info(f"Live rate: 1 USD = {rate} {currency_code}")
            return {
                "exchange_rate": rate,
                "local_budget": local_budget,
                "conversion_note": conversion_note,
            }
        else:
            logger.warning(f"Rate for {currency_code} not in API response, using fallback")
    except Exception as e:
        logger.warning(f"Currency API call failed ({e}), falling back to hardcoded rate")

    # Fallback to approximate hardcoded rate
    rate = FALLBACK_RATES.get(currency_code, 1.0)
    local_budget = round(usd_budget * rate, 2)
    conversion_note = (
        f"${usd_budget:.2f} USD ≈ {symbol}{local_budget:,.2f} {currency_code} "
        f"(rate: {rate:.4f} · estimated, live rate unavailable)"
    )
    logger.info(f"Fallback rate: 1 USD ≈ {rate} {currency_code}")
    return {
        "exchange_rate": rate,
        "local_budget": local_budget,
        "conversion_note": conversion_note,
    }
