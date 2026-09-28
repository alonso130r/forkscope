"""Small domain examples used for end-to-end evaluator tests."""

from typing import Any

import numpy as np


class PointMassEnvironment:
    """One-dimensional point mass with bounded control and a fixed goal."""

    observation_space = None
    action_space = None

    def __init__(self, goal: float = 5.0) -> None:
        self.goal = goal
        self.position = 0.0
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.position = -float(seed or 0)
        self.steps = 0
        return self._observation(), {"seed": seed}

    def step(
        self, action: float
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.position += float(action)
        self.steps += 1
        distance = abs(self.goal - self.position)
        terminated = distance == 0.0
        return self._observation(), -distance, terminated, False, {"distance": distance}

    def close(self) -> None:
        pass

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
