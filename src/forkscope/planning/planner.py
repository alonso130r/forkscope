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
    ) -> Any:
        """Return the action to execute now."""
        ...
