"""Small deterministic environment and planner fixtures for integration tests."""

from typing import Any

import numpy as np


class CountingEnvironment:
    instances: list["CountingEnvironment"] = []

    def __init__(self) -> None:
        self.seed: int | None = None
        self.actions: list[Any] = []
        self.closed = False
        self.instances.append(self)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.seed = seed
        self.actions = []
        return np.array([0.0]), {"seed": seed}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        observation = np.array([float(len(self.actions))])
        return observation, 1.0, len(self.actions) == 2, False, {
            "action_count": np.int64(len(self.actions))
        }

    def close(self) -> None:
        self.closed = True


class CountingPlanner:
    instances: list["CountingPlanner"] = []

    def __init__(self) -> None:
        self.seed: int | None = None
        self.calls: list[tuple[Any, int]] = []
        self.instances.append(self)

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def plan(self, observation: Any, *, horizon: int) -> int:
        self.calls.append((observation.copy(), horizon))
        return horizon


def make_environment() -> CountingEnvironment:
    return CountingEnvironment()


def make_planner() -> CountingPlanner:
    return CountingPlanner()
