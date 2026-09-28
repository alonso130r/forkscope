"""Small deterministic environment and planner for a horizon sweep."""

from pathlib import Path
from typing import Any

import numpy as np

from forkscope import RolloutBenchmarker


class PointMassEnvironment:
    """One-dimensional point mass with a fixed goal at position 5."""

    observation_space = None
    action_space = None

    def __init__(self, goal: float = 5.0) -> None:
        self.goal = goal
        self.position = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.position = -float(seed or 0)
        return self._observation(), {"seed": seed}

    def step(
        self, action: float
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.position += float(action)
        distance = abs(self.goal - self.position)
        terminated = distance == 0.0
        return self._observation(), -distance, terminated, False, {"distance": distance}

    def close(self) -> None:
        """Release resources (none are used by this example)."""

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "position": np.array([self.position], dtype=np.float64),
            "goal": np.array([self.goal], dtype=np.float64),
        }


class BoundedGreedyPlanner:
    """Move toward the goal by at most the configured planning horizon."""

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def plan(self, observation: dict[str, np.ndarray], *, horizon: int) -> float:
        distance = float(observation["goal"][0] - observation["position"][0])
        return float(np.clip(distance, -horizon, horizon))


def make_environment() -> PointMassEnvironment:
    """Factory used by the CLI and programmatic benchmarker."""
    return PointMassEnvironment()


def make_planner() -> BoundedGreedyPlanner:
    """Factory used by the CLI and programmatic benchmarker."""
    return BoundedGreedyPlanner()


def main() -> None:
    benchmarker = RolloutBenchmarker(
        make_environment,
        make_planner,
        seeds=[0, 1, 2],
        horizons=[1, 2, 4],
        max_steps=10,
    )
    records = benchmarker.run()
    output = Path("outputs/point_mass/programmatic-results.jsonl")
    benchmarker.write_results(output)
    print(f"Wrote {len(records)} episode records to {output}")


if __name__ == "__main__":
    main()
