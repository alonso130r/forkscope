from collections.abc import Callable, Sequence

from forkscope.evaluation.episode import run_episode
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
) -> list[EpisodeRecord]:
    records = []

    for horizon in horizons:
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
                    )
                )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    return records
