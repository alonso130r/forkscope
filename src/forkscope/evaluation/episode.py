from typing import Any

from forkscope.interface.interfaces import InterfaceEnvironment
from forkscope.planning.planner import Planner
from forkscope.results.records import EpisodeRecord, StepRecord


def run_episode(
    env: InterfaceEnvironment,
    planner: Planner,
    *,
    seed: int,
    horizon: int,
    max_steps: int,
) -> EpisodeRecord:
    observation, _ = env.reset(seed=seed)
    steps: list[StepRecord] = []
    total_reward = 0.0
    terminated = truncated = False

    for step_index in range(max_steps):
        action = planner.plan(
            observation,
            horizon=horizon,
            rng_seed=seed + step_index,
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        steps.append(
            StepRecord(
                index=step_index,
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        total_reward += reward
        observation = next_observation

        if terminated or truncated:
            break

    return EpisodeRecord(
        seed=seed,
        horizon=horizon,
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )
