from typing import Any, Protocol

import numpy as np

Array = np.ndarray


# gymnasium-compatible environment interface
class InterfaceEnvironment(Protocol):
    observation_space: Any
    action_space: Any

    # returns (observation, info)
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]: ...

    # returns (observation, reward, terminated, truncated, info)
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


class DynamicsModel(Protocol):
    def predict_next(self, observations: Array, actions: Array) -> Array: ...


class Objective(Protocol):
    def score(self, trajectories: Array) -> Array: ...
