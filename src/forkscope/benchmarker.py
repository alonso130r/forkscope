"""Public programmatic API for running rollout horizon benchmarks."""

from collections.abc import Callable, Sequence
from pathlib import Path

from forkscope.evaluation.sweep import run_sweep
from forkscope.interface.backend import BackendName, resolve_backend
from forkscope.interface.interfaces import InterfaceEnvironment
from forkscope.planning.planner import Planner
from forkscope.records.records import EpisodeRecord
from forkscope.records.writer import write_jsonl


class RolloutBenchmarker:
    """Run matched-seed horizon sweeps and retain their episode records."""

    def __init__(
        self,
        env_factory: Callable[[], InterfaceEnvironment],
        planner_factory: Callable[[], Planner],
        *,
        seeds: Sequence[int],
        horizons: Sequence[int],
        max_steps: int,
        backend: BackendName = "numpy",
    ) -> None:
        seed_values = tuple(seeds)
        horizon_values = tuple(horizons)
        if not seed_values:
            raise ValueError("seeds must contain at least one value.")
        if any(seed < 0 for seed in seed_values):
            raise ValueError("seeds must be non-negative.")
        if not horizon_values or any(horizon <= 0 for horizon in horizon_values):
            raise ValueError("horizons must contain positive values.")
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        resolve_backend(backend)

        self.env_factory = env_factory
        self.planner_factory = planner_factory
        self.seeds = seed_values
        self.horizons = horizon_values
        self.max_steps = max_steps
        self.backend = backend
        self.results: list[EpisodeRecord] = []

    def run(self) -> list[EpisodeRecord]:
        """Execute the sweep, save results on this instance, and return them."""
        self.results = []
        self.results = run_sweep(
            self.env_factory,
            self.planner_factory,
            seeds=self.seeds,
            horizons=self.horizons,
            max_steps=self.max_steps,
            backend=self.backend,
        )
        return self.results

    def write_results(self, path: str | Path) -> None:
        """Write the current episode results as JSON Lines."""
        write_jsonl(self.results, path, backend=self.backend)
