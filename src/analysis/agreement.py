"""
Vectorized pairwise MP agreement matrix computation using BLAS matmul.

Core idea: instead of O(V * N^2) Python loops, use three float32 matrix
multiplications (O(V * N^2) in BLAS) to count agreements for each vote value.
"""

import logging

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.config import MIN_COPRESENCE

logger = logging.getLogger(__name__)


def compute_agreement_matrix(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    min_copresence: int = MIN_COPRESENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise agreement fraction and co-presence count for all MP pairs.

    Args:
        vote_matrix:     (V, N) float32 — 1=YES, -1=NO, 0=ABSTAIN, NaN=excluded
        presence_matrix: (V, N) bool    — True where MP was present
        min_copresence:  minimum shared votings to compute agreement (else NaN)

    Returns:
        agreement_frac: (N, N) float32 — NaN where copresence < min_copresence
        copresence:     (N, N) int32   — raw co-presence counts
    """
    V, N = vote_matrix.shape
    pres = presence_matrix.astype(np.float32)  # (V, N)

    steps = [
        "Co-presence matrix",
        "YES agreement",
        "NO agreement",
        "ABSTAIN agreement",
        "Normalising",
    ]
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
        task = progress.add_task("Agreement matrix", total=len(steps))

        progress.console.print(f"  → {steps[0]}")
        copresence = (pres.T @ pres).astype(np.int32)
        progress.advance(task)

        progress.console.print(f"  → {steps[1]}")
        v_yes = ((vote_matrix == 1.0) & presence_matrix).astype(np.float32)
        agree_raw = v_yes.T @ v_yes
        progress.advance(task)

        progress.console.print(f"  → {steps[2]}")
        v_no = ((vote_matrix == -1.0) & presence_matrix).astype(np.float32)
        agree_raw += v_no.T @ v_no
        progress.advance(task)

        progress.console.print(f"  → {steps[3]}")
        v_abs = ((vote_matrix == 0.0) & presence_matrix).astype(np.float32)
        agree_raw += v_abs.T @ v_abs
        progress.advance(task)

        progress.console.print(f"  → {steps[4]}")
        agreement_frac = np.where(
            copresence >= min_copresence,
            agree_raw / np.maximum(copresence, 1).astype(np.float32),
            np.nan,
        ).astype(np.float32)
        np.fill_diagonal(agreement_frac, np.nan)
        np.fill_diagonal(copresence, 0)
        progress.advance(task)

    logger.info(
        "Agreement matrix: %d MPs, %d votings. "
        "Valid pairs: %d / %d",
        N, V,
        int(np.sum(~np.isnan(agreement_frac)) // 2),
        N * (N - 1) // 2,
    )
    return agreement_frac, copresence


def apply_threshold(
    agreement_frac: np.ndarray,
    copresence: np.ndarray,
    threshold: float,
    min_copresence: int = MIN_COPRESENCE,
) -> np.ndarray:
    """
    Build a binary adjacency matrix from the agreement matrix.

    Returns:
        adjacency: (N, N) uint8 — 1 where agreement >= threshold AND copresence >= min
    """
    valid = copresence >= min_copresence
    adjacency = (agreement_frac >= threshold) & valid
    return adjacency.astype(np.uint8)
