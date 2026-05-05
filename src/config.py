"""
Central configuration for all scripts and analysis modules.
Adjust constants here; no need to touch individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
BASE_URL = "https://api.sejm.gov.pl"
USER_AGENT = "sejm-research/1.0 (academic)"

# ---------------------------------------------------------------------------
# Terms to analyse
# ---------------------------------------------------------------------------
TERMS: list[int] = [10]  # extend to [9, 10] for cross-term temporal analysis

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
NETWORKS_DIR = DATA_DIR / "networks"
ANALYSIS_DIR = DATA_DIR / "analysis"

# ---------------------------------------------------------------------------
# Fetch parameters
# ---------------------------------------------------------------------------
CONCURRENCY = 80          # asyncio.Semaphore cap (simultaneous HTTP requests)
RETRY_ATTEMPTS = 3        # number of retries on 429/5xx
RETRY_BASE_DELAY = 1.0    # seconds; doubled each retry

# ---------------------------------------------------------------------------
# Vote encoding (stored in vote_matrix as float32)
# ---------------------------------------------------------------------------
VOTE_ENCODING: dict[str, float] = {
    "YES": 1.0,
    "NO": -1.0,
    "ABSTAIN": 0.0,
    # ABSENT and VOTE_VALID → NaN (handled in store.py)
}
EXCLUDED_VOTES = {"ABSENT", "VOTE_VALID"}

# ---------------------------------------------------------------------------
# Agreement matrix parameters
# ---------------------------------------------------------------------------
MIN_COPRESENCE = 50           # minimum shared votings to include an MP pair
AGREEMENT_THRESHOLDS = [0.30, 0.50, 0.70, 0.90, 0.99]

# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------
MIN_COPRESENCE_MONTHLY = 10   # lower threshold for monthly sub-networks

# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------
LEIDEN_SEED = 42
LEIDEN_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Rebel detection
# ---------------------------------------------------------------------------
MIN_REBEL_VOTES = 100         # minimum votes for rebel score calculation

# ---------------------------------------------------------------------------
# Party switcher detection
# ---------------------------------------------------------------------------
MIN_SWITCH_SITTINGS = 2       # minimum consecutive sittings under new club

# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "mps"       # "mps" for Apple Silicon, "cuda" or "cpu" otherwise
EMBEDDING_BATCH_SIZE = 256
BERTOPIC_MIN_TOPIC_SIZE = 10
UMAP_N_COMPONENTS = 5
UMAP_N_NEIGHBORS = 15
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5

# ---------------------------------------------------------------------------
# Voting blocs
# ---------------------------------------------------------------------------
MIN_TOPIC_VOTINGS = 5         # minimum votings per topic to run bloc analysis
MIN_COPRESENCE_TOPIC = 3      # minimum co-presence for topic sub-networks
BLOC_THRESHOLD = 0.60         # agreement threshold for topic sub-networks
CROSS_PARTY_MAX_FRACTION = 0.70  # a community is cross-party if no club > this fraction
