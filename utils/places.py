"""Google Places API — enrich itinerary options with photos and Maps links."""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_FIND_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"


async def _fetch_place_info(
    name: str, destination: str, api_key: str, client: httpx.AsyncClient
) -> dict:
    query = f"{name} {destination}"
    try:
        resp = await client.get(
            _FIND_URL,
            params={
                "input": query,
                "inputtype": "textquery",
                "fields": "place_id,photos",
                "key": api_key,
            },
            timeout=8.0,
        )
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return {}
        place = candidates[0]
        place_id = place.get("place_id", "")
        photos = place.get("photos", [])
        photo_ref = photos[0].get("photo_reference", "") if photos else ""
        maps_url = (
            f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            if place_id
            else f"https://www.google.com/maps/search/?api=1&query={query.replace(' ', '+')}"
        )
        return {"photo_reference": photo_ref, "maps_url": maps_url}
    except Exception as e:
        logger.warning(f"Places API failed for '{name}': {e}")
        return {}


async def enrich_structured_itinerary(
    structured: dict, destination: str, api_key: str
) -> dict:
    """Add photo_reference and maps_url to every option in the structured itinerary."""
    if not api_key:
        return structured

    options: list[dict] = []
    for opt in structured.get("hotel", {}).get("options", []):
        options.append(opt)
    for opt in structured.get("arrival", {}).get("options", []):
        options.append(opt)
    for day in structured.get("days", []):
        for slot in day.get("slots", []):
            for opt in slot.get("options", []):
                options.append(opt)

    sem = asyncio.Semaphore(5)

    async with httpx.AsyncClient() as client:

        async def _enrich(opt: dict) -> None:
            name = opt.get("title", "")
            if not name:
                return
            async with sem:
                info = await _fetch_place_info(name, destination, api_key, client)
                opt.update(info)

        await asyncio.gather(*(_enrich(o) for o in options), return_exceptions=True)

    logger.info(f"Enriched {len(options)} options with Places data")
    return structured
