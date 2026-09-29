import math
from collections.abc import Callable, Sequence
from itertools import product

from forkscope.evaluation.episode import run_episode
from forkscope.interface.backend import BackendName, resolve_backend
from forkscope.interface.interfaces import InterfaceEnvironment
from forkscope.planning.planner import Planner
from forkscope.records.records import EpisodeRecord


def run_sweep(
    env_factory: Callable[[], InterfaceEnvironment],
    planner_factory: Callable[[], Planner],
    *,
    seeds: Sequence[int],
    horizons: Sequence[int],
    max_steps: int,
    backend: BackendName = "numpy",
    rollout_counts: Sequence[int] | None = None,
    temperatures: Sequence[float] | None = None,
) -> list[EpisodeRecord]:
    resolve_backend(backend)
    _validate_optional_ranges(rollout_counts, temperatures)
    rollout_values = tuple(rollout_counts) if rollout_counts is not None else (None,)
    temperature_values = tuple(temperatures) if temperatures is not None else (None,)
    records = []

    for horizon, rollout_count, temperature in product(
        horizons, rollout_values, temperature_values
    ):
        for seed in seeds:
            env = env_factory()
            try:
                records.append(
                    run_episode(
                        env,
                        planner_factory(),
                        seed=seed,
                        horizon=horizon,
                        max_steps=max_steps,
                        backend=backend,
                        rollout_count=rollout_count,
                        temperature=temperature,
                    )
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    return records


def _validate_optional_ranges(
    rollout_counts: Sequence[int] | None, temperatures: Sequence[float] | None
) -> None:
    if rollout_counts is not None and (
        not rollout_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in rollout_counts
        )
    ):
        raise ValueError("rollout_counts must contain positive integers.")
    if temperatures is not None and (
        not temperatures
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in temperatures
        )
    ):
        raise ValueError("temperatures must contain finite positive numbers.")
