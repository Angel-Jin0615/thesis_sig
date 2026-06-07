from __future__ import annotations

from typing import Callable, Iterable, List, TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = -1,
    prefer: str = "processes",
) -> List[R]:
    """Run a single-argument function over items in parallel."""
    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(fn)(item) for item in items)
