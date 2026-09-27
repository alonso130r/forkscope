from typing import Any, Protocol


class Planner(Protocol):
    def plan(
        self,
        observation: Any,
        *,
        horizon: int,
        rng_seed: int | None = None,
    ) -> Any:
        """Return the action to execute now."""
        ...
