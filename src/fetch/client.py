"""
Async HTTP client for the Polish Sejm API.
All requests go through fetch_json which handles retries and semaphore limiting.
"""

import asyncio
import logging
from typing import Any

import aiohttp
import orjson

from src.config import BASE_URL, RETRY_ATTEMPTS, RETRY_BASE_DELAY, USER_AGENT

logger = logging.getLogger(__name__)


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> Any | None:
    """Fetch a URL and return parsed JSON. Returns None on permanent failure."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with semaphore:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        raw = await resp.read()
                        return orjson.loads(raw)
                    if resp.status in (429, 503):
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning("HTTP %d for %s, retrying in %.1fs", resp.status, url, wait)
                        await asyncio.sleep(wait)
                        continue
                    logger.error("HTTP %d for %s (non-retriable)", resp.status, url)
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("Network error for %s: %s, retrying in %.1fs", url, exc, wait)
            await asyncio.sleep(wait)
    logger.error("Permanent failure after %d attempts: %s", RETRY_ATTEMPTS, url)
    return None


async def fetch_votings_index(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    term: int,
) -> list[dict[str, Any]]:
    """
    Fetch the votings index for a term.

    Returns a flat list of dicts: [{sitting, voting_num}] for all votings.
    The API returns ProceedingDay objects with votingsNum; we expand to individual pairs.
    """
    url = f"{BASE_URL}/sejm/term{term}/votings"
    data = await fetch_json(session, url, semaphore)
    if data is None:
        return []

    pairs: list[dict[str, Any]] = []
    for day in data:
        sitting = day.get("proceeding") or day.get("sitting")
        n_votings = day.get("votingsNum", 0)
        if sitting is None or n_votings == 0:
            continue
        for num in range(1, n_votings + 1):
            pairs.append({"sitting": int(sitting), "voting_num": num})
    return pairs


async def fetch_voting_detail(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    term: int,
    sitting: int,
    voting_num: int,
) -> dict[str, Any] | None:
    """Fetch details for a single voting including individual vote records."""
    url = f"{BASE_URL}/sejm/term{term}/votings/{sitting}/{voting_num}"
    return await fetch_json(session, url, semaphore)


async def fetch_mps(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    term: int,
) -> list[dict[str, Any]]:
    """Fetch all MPs for a term (handles pagination automatically)."""
    all_mps: list[dict[str, Any]] = []
    offset = 0
    limit = 200
    while True:
        url = f"{BASE_URL}/sejm/term{term}/MP?limit={limit}&offset={offset}"
        data = await fetch_json(session, url, semaphore)
        if not data:
            break
        all_mps.extend(data)
        if len(data) < limit:
            break
        offset += limit
    return all_mps


async def fetch_clubs(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    term: int,
) -> list[dict[str, Any]]:
    """Fetch list of clubs for a term."""
    url = f"{BASE_URL}/sejm/term{term}/clubs"
    data = await fetch_json(session, url, semaphore)
    return data or []
