from dataclasses import dataclass
from typing import Any


@dataclass
class StepRecord:
    index: int
    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeRecord:
    seed: int
    horizon: int
    total_reward: float
    steps: list[StepRecord]
    terminated: bool
    truncated: bool
    step_limit_reached: bool
    rollout_count: int | None = None
    temperature: float | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)
