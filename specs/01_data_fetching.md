# Spec 01 — Data Fetching

## Purpose

Retrieve all voting data from the Polish Sejm public API and store it in a structured,
resumable, analysis-ready format.

---

## API Base URL

```
https://api.sejm.gov.pl
```

No authentication required. No documented rate limit.
Use `User-Agent: sejm-research/1.0 (academic)` header on all requests.

---

## Target Endpoints

### 1. List Votings Index

```
GET /sejm/term{term}/votings
```

Returns an array of `ProceedingDay` objects:
```json
[
  { "proceeding": 1, "date": "2023-11-13", "votingsNum": 42 }
]
```

Used to enumerate all `(sitting, voting_num)` pairs.
`votingsNum` gives the count; individual voting numbers are 1..votingsNum.

### 2. Voting Details (with individual votes)

```
GET /sejm/term{term}/votings/{sitting}/{votingNum}
```

Returns a `VotingDetails` object:
```json
{
  "votingNumber": 1,
  "date": "2023-11-13T10:05:00",
  "title": "...",
  "topic": "...",
  "description": "...",
  "kind": "ELECTRONIC",
  "yes": 231, "no": 189, "abstain": 3, "notParticipating": 37,
  "totalVoted": 423, "present": 460,
  "majorityType": "SIMPLE", "majorityVotes": 212,
  "votes": [
    { "mpCredentialNumber": 1, "firstName": "Jan", "lastName": "Kowalski",
      "club": "KO", "vote": "YES" }
  ]
}
```

### 3. MPs List

```
GET /sejm/term{term}/MP
```

Returns array of MP objects (all MPs, including inactive). Paginated (`limit`, `offset`).

### 4. Clubs List

```
GET /sejm/term{term}/clubs
```

Returns array of club objects with club code and name.

---

## Async Fetch Strategy

### Concurrency

Use `asyncio` + `aiohttp` with `asyncio.Semaphore(CONCURRENCY)` where `CONCURRENCY = 80`.

### Retry Logic

On HTTP 429, 503, or connection error: exponential backoff.
- Attempt 1: wait 1 s
- Attempt 2: wait 2 s
- Attempt 3: wait 4 s
- After 3 failures: log error and skip (do not crash pipeline)

### Two-Phase Fetch

**Phase 1** — enumerate index (1 request):
```
GET /sejm/term{term}/votings → list of (sitting, voting_num) pairs
```

**Phase 2** — fan-out (one request per voting):
```
asyncio.gather(*[fetch_voting(sitting, num) for sitting, num in pairs])
```

### Resumability

Before fetching, build a set of already-existing files in `data/raw/term{N}/`.
Skip any `(sitting, num)` whose file `{sitting}_{num}.json` already exists.
This allows restarting a crashed fetch without duplicate work.

---

## Raw Storage

Each voting is saved as:
```
data/raw/term{N}/{sitting}_{num}.json
```

Content: the raw JSON response from the API (VotingDetails object).

MPs are saved as:
```
data/raw/term{N}/mps.json
```

---

## ON_LIST Voting Handling

Some votings have `kind == "ON_LIST"`. In these, each vote entry has:
- `vote == "VOTE_VALID"` (not YES/NO/ABSTAIN)
- `listVotes: {"1": "YES", "2": "NO", ...}`

**Decision**: Store `vote = "VOTE_VALID"` in the votes table. Filter these out in all
agreement/cohesion calculations using `filter(pl.col("vote") != "VOTE_VALID")`.
Do NOT attempt to flatten `listVotes` — the semantics differ from regular votes.

---

## Parquet Assembly

After raw fetch, run `assemble_parquet(term)` which reads all JSON files and writes three
parquet files. Use `orjson.loads` for parsing.

### `data/parquet/term{N}/mps.parquet`

| Column | Polars type | Source field |
|--------|-------------|-------------|
| `mp_id` | `Int32` | `id` |
| `first_name` | `Utf8` | `firstName` |
| `last_name` | `Utf8` | `lastName` |
| `club` | `Categorical` | `club` |
| `active` | `Boolean` | `active` |
| `birth_date` | `Date` | `birthDate` |
| `birth_location` | `Utf8` | `birthLocation` |
| `voivodeship` | `Categorical` | `voivodeship` |
| `district_name` | `Utf8` | `districtName` |
| `district_num` | `Int16` | `districtNum` |
| `education_level` | `Categorical` | `educationLevel` |
| `profession` | `Utf8` | `profession` |
| `number_of_votes` | `Int32` | `numberOfVotes` |

### `data/parquet/term{N}/votings.parquet`

| Column | Polars type | Source field |
|--------|-------------|-------------|
| `voting_key` | `Utf8` | `"{sitting}_{votingNumber}"` |
| `sitting` | `Int16` | from filename |
| `voting_num` | `Int16` | `votingNumber` |
| `date` | `Datetime` | `date` (ISO 8601) |
| `title` | `Utf8` | `title` |
| `topic` | `Utf8` | `topic` (nullable) |
| `description` | `Utf8` | `description` (nullable) |
| `kind` | `Categorical` | `kind` |
| `majority_type` | `Categorical` | `majorityType` |
| `yes_count` | `Int16` | `yes` |
| `no_count` | `Int16` | `no` |
| `abstain_count` | `Int16` | `abstain` |
| `not_participating` | `Int16` | `notParticipating` |
| `total_voted` | `Int16` | `totalVoted` |
| `term` | `Int8` | from CLI argument |

### `data/parquet/term{N}/votes.parquet`

Largest table (~1.8 M rows for term 10).

| Column | Polars type | Notes |
|--------|-------------|-------|
| `voting_key` | `Utf8` | FK to votings |
| `sitting` | `Int16` | denormalized for fast groupby |
| `voting_num` | `Int16` | denormalized |
| `date` | `Date` | denormalized for time-series groupby |
| `mp_id` | `Int32` | `mpCredentialNumber` |
| `club` | `Categorical` | club AT TIME OF THIS VOTE |
| `vote` | `Categorical` | YES / NO / ABSTAIN / ABSENT / VOTE_VALID |
| `term` | `Int8` | from CLI argument |

**Important**: `club` in `votes.parquet` reflects the club the MP belonged to at the time
of each individual vote. This is the authoritative source for traitor detection.

---

## Performance Notes

- 3,873 voting files for term 10 at ~0.5 s each → serial ≈ 32 min, with 80 concurrent ≈ 1-2 min
- Use `orjson` for JSON parsing (3-5× faster than stdlib)
- Accumulate rows in Python lists, bulk-convert to polars DataFrame once, write parquet once

---

## Edge Cases

- Missing `topic` or `description` fields: fill with `None`
- `votes` array empty (rare): skip, write zero vote rows for that voting
- MP appears in votes but not in mps list: store vote row with mp_id; join will produce nulls
- Duplicate `(sitting, voting_num)` in index: deduplicate before fetching
