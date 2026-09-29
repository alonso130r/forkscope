from typing import Any, Protocol


class Planner(Protocol):
    def reset(self, *, seed: int) -> None:
        """Reset planner state and initialize its randomness for one episode."""
        ...

    def plan(
        self,
        observation: Any,
        *,
        horizon: int,
        rollout_count: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Return the action to execute now using supported optional settings."""
        ...
