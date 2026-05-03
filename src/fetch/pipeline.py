"""
Two-phase async fetch pipeline.

Phase 1: enumerate all (sitting, voting_num) pairs from the votings index.
Phase 2: fan-out to individual voting endpoints and save raw JSON files.

Supports resumability: skips files that already exist on disk.
"""

import asyncio
import logging
from pathlib import Path

import aiofiles
import aiohttp
import orjson
import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config import (
    CONCURRENCY,
    PARQUET_DIR,
    RAW_DIR,
)
from src.fetch.client import (
    fetch_clubs,
    fetch_mps,
    fetch_voting_detail,
    fetch_votings_index,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 + 2: fetch and save raw JSON
# ---------------------------------------------------------------------------


async def _save_voting(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    term: int,
    sitting: int,
    voting_num: int,
    out_dir: Path,
) -> None:
    """Fetch one voting and write to {out_dir}/{sitting}_{voting_num}.json."""
    data = await fetch_voting_detail(session, semaphore, term, sitting, voting_num)
    if data is None:
        logger.warning("No data for term%d sitting%d voting%d", term, sitting, voting_num)
        return
    path = out_dir / f"{sitting}_{voting_num}.json"
    async with aiofiles.open(path, "wb") as f:
        await f.write(orjson.dumps(data))


async def run_fetch(term: int, resume: bool = True) -> None:
    """
    Fetch all votings for a given term.

    Args:
        term: Parliamentary term number (e.g. 10).
        resume: If True, skip voting files that already exist on disk.
    """
    out_dir = RAW_DIR / f"term{term}"
    out_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 20)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: get index
        logger.info("Fetching votings index for term %d …", term)
        pairs = await fetch_votings_index(session, semaphore, term)
        logger.info("Found %d (sitting, voting_num) pairs", len(pairs))

        # Fetch and save MPs + clubs (small, do before fan-out)
        logger.info("Fetching MPs …")
        mps = await fetch_mps(session, semaphore, term)
        mps_path = RAW_DIR / f"term{term}" / "mps.json"
        async with aiofiles.open(mps_path, "wb") as f:
            await f.write(orjson.dumps(mps))
        logger.info("Saved %d MPs to %s", len(mps), mps_path)

        clubs = await fetch_clubs(session, semaphore, term)
        clubs_path = RAW_DIR / f"term{term}" / "clubs.json"
        async with aiofiles.open(clubs_path, "wb") as f:
            await f.write(orjson.dumps(clubs))

        # Phase 2: filter already-fetched
        if resume:
            existing = {p.name for p in out_dir.glob("*.json") if p.name not in ("mps.json", "clubs.json")}
            pairs = [p for p in pairs if f"{p['sitting']}_{p['voting_num']}.json" not in existing]
            logger.info("%d pairs remaining after resume filter", len(pairs))

        if not pairs:
            logger.info("Nothing to fetch — all files exist.")
            return

        # Phase 2: fan-out with rich progress bar (works correctly with asyncio)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task(f"Fetching term {term}", total=len(pairs))

            async def _fetch_and_tick(p: dict) -> None:
                await _save_voting(session, semaphore, term, p["sitting"], p["voting_num"], out_dir)
                progress.advance(task)

            await asyncio.gather(*[_fetch_and_tick(p) for p in pairs])

    logger.info("Fetch complete for term %d. Files in %s", term, out_dir)


# ---------------------------------------------------------------------------
# Assembly: raw JSON → parquet
# ---------------------------------------------------------------------------


def assemble_parquet(term: int) -> None:
    """
    Read all raw JSON files for a term and write three parquet files.
    Uses columnar lists (one list per column) instead of list[dict] to avoid
    per-row dict overhead over ~1.8M vote records.
    """
    from datetime import datetime

    raw_dir = RAW_DIR / f"term{term}"
    out_dir = PARQUET_DIR / f"term{term}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MPs ---
    mps_path = raw_dir / "mps.json"
    if mps_path.exists():
        mps_raw = orjson.loads(mps_path.read_bytes())
        mp_ids, first_names, last_names, clubs, actives = [], [], [], [], []
        birth_dates, birth_locs, voivodeships, dist_names, dist_nums = [], [], [], [], []
        edu_levels, professions, n_votes = [], [], []
        for m in mps_raw:
            bd_raw = m.get("birthDate")
            bd = None
            if bd_raw:
                try:
                    p = bd_raw.split("-")
                    from datetime import date as _date
                    bd = _date(int(p[0]), int(p[1]), int(p[2]))
                except Exception:
                    pass
            mp_ids.append(m.get("id"))
            first_names.append(m.get("firstName") or "")
            last_names.append(m.get("lastName") or "")
            clubs.append(m.get("club") or "")
            actives.append(bool(m.get("active", True)))
            birth_dates.append(bd)
            birth_locs.append(m.get("birthLocation") or "")
            voivodeships.append(m.get("voivodeship") or "")
            dist_names.append(m.get("districtName") or "")
            dist_nums.append(m.get("districtNum"))
            edu_levels.append(m.get("educationLevel") or "")
            professions.append(m.get("profession") or "")
            n_votes.append(m.get("numberOfVotes"))
        df_mps = pl.DataFrame({
            "mp_id": pl.Series(mp_ids, dtype=pl.Int32),
            "first_name": first_names,
            "last_name": last_names,
            "club": pl.Series(clubs, dtype=pl.Categorical),
            "active": actives,
            "birth_date": pl.Series(birth_dates, dtype=pl.Date),
            "birth_location": birth_locs,
            "voivodeship": pl.Series(voivodeships, dtype=pl.Categorical),
            "district_name": dist_names,
            "district_num": pl.Series(dist_nums, dtype=pl.Int16),
            "education_level": pl.Series(edu_levels, dtype=pl.Categorical),
            "profession": professions,
            "number_of_votes": pl.Series(n_votes, dtype=pl.Int32),
        })
        df_mps.write_parquet(out_dir / "mps.parquet")
        logger.info("Wrote mps.parquet (%d rows)", len(df_mps))
    else:
        logger.warning("mps.json not found, skipping mps.parquet")

    # --- Votings + Votes (columnar accumulation) ---
    json_files = sorted(
        [p for p in raw_dir.glob("*.json") if p.name not in ("mps.json", "clubs.json")]
    )
    logger.info("Assembling %d voting JSON files …", len(json_files))

    # Voting columns
    v_keys: list[str] = []
    v_sittings: list[int] = []
    v_nums: list[int] = []
    v_dates: list[datetime | None] = []
    v_titles: list[str] = []
    v_topics: list[str] = []
    v_descs: list[str] = []
    v_kinds: list[str] = []
    v_maj_types: list[str] = []
    v_yes: list[int] = []
    v_no: list[int] = []
    v_abs: list[int] = []
    v_notp: list[int] = []
    v_total: list[int] = []
    v_terms: list[int] = []

    # Vote columns
    vt_keys: list[str] = []
    vt_sittings: list[int] = []
    vt_nums: list[int] = []
    vt_dates: list = []
    vt_mp_ids: list[int | None] = []
    vt_clubs: list[str] = []
    vt_votes: list[str] = []
    vt_terms: list[int] = []

    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        task = progress.add_task("Assembling parquet", total=len(json_files))
        for path in json_files:
            try:
                parts = path.stem.split("_")
                sitting = int(parts[0])
                voting_num_file = int(parts[1])
                data = orjson.loads(path.read_bytes())

                date_str = data.get("date") or ""
                dt = None
                date_only = None
                if date_str:
                    try:
                        dt = datetime.fromisoformat(date_str)
                        date_only = dt.date()
                    except Exception:
                        pass

                voting_num = data.get("votingNumber") or voting_num_file
                vkey = f"{sitting}_{voting_num}"

                v_keys.append(vkey)
                v_sittings.append(sitting)
                v_nums.append(voting_num)
                v_dates.append(dt)
                v_titles.append(data.get("title") or "")
                v_topics.append(data.get("topic") or "")
                v_descs.append(data.get("description") or "")
                v_kinds.append(data.get("kind") or "")
                v_maj_types.append(data.get("majorityType") or "")
                v_yes.append(data.get("yes") or 0)
                v_no.append(data.get("no") or 0)
                v_abs.append(data.get("abstain") or 0)
                v_notp.append(data.get("notParticipating") or 0)
                v_total.append(data.get("totalVoted") or 0)
                v_terms.append(term)

                for v in data.get("votes") or []:
                    vt_keys.append(vkey)
                    vt_sittings.append(sitting)
                    vt_nums.append(voting_num)
                    vt_dates.append(date_only)
                    vt_mp_ids.append(v.get("mpCredentialNumber"))
                    vt_clubs.append(v.get("club") or "")
                    vt_votes.append(v.get("vote") or "ABSENT")
                    vt_terms.append(term)
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path.name, exc)
            progress.advance(task)

    # --- votings.parquet ---
    df_votings = pl.DataFrame({
        "voting_key": v_keys,
        "sitting": pl.Series(v_sittings, dtype=pl.Int16),
        "voting_num": pl.Series(v_nums, dtype=pl.Int16),
        "date": pl.Series(v_dates, dtype=pl.Datetime),
        "title": v_titles,
        "topic": v_topics,
        "description": v_descs,
        "kind": pl.Series(v_kinds, dtype=pl.Categorical),
        "majority_type": pl.Series(v_maj_types, dtype=pl.Categorical),
        "yes_count": pl.Series(v_yes, dtype=pl.Int16),
        "no_count": pl.Series(v_no, dtype=pl.Int16),
        "abstain_count": pl.Series(v_abs, dtype=pl.Int16),
        "not_participating": pl.Series(v_notp, dtype=pl.Int16),
        "total_voted": pl.Series(v_total, dtype=pl.Int16),
        "term": pl.Series(v_terms, dtype=pl.Int8),
    })
    df_votings.write_parquet(out_dir / "votings.parquet")
    logger.info("Wrote votings.parquet (%d rows)", len(df_votings))

    # --- votes.parquet ---
    df_votes = pl.DataFrame({
        "voting_key": vt_keys,
        "sitting": pl.Series(vt_sittings, dtype=pl.Int16),
        "voting_num": pl.Series(vt_nums, dtype=pl.Int16),
        "date": pl.Series(vt_dates, dtype=pl.Date),
        "mp_id": pl.Series(vt_mp_ids, dtype=pl.Int32),
        "club": pl.Series(vt_clubs, dtype=pl.Categorical),
        "vote": pl.Series(vt_votes, dtype=pl.Categorical),
        "term": pl.Series(vt_terms, dtype=pl.Int8),
    })
    df_votes.write_parquet(out_dir / "votes.parquet")
    logger.info("Wrote votes.parquet (%d rows)", len(df_votes))
