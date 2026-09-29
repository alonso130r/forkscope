from forkscope.interface.backend import BackendName, resolve_backend, validate_array_backend
from forkscope.interface.interfaces import InterfaceEnvironment
from forkscope.planning.planner import Planner
from forkscope.records.records import EpisodeRecord, StepRecord


def run_episode(
    env: InterfaceEnvironment,
    planner: Planner,
    *,
    seed: int,
    horizon: int,
    max_steps: int,
    backend: BackendName = "numpy",
    rollout_count: int | None = None,
    temperature: float | None = None,
) -> EpisodeRecord:
    adapter = resolve_backend(backend)
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    planner.reset(seed=seed)
    observation, _ = env.reset(seed=seed)
    validate_array_backend(observation, adapter, name="reset observation")
    steps: list[StepRecord] = []
    total_reward = 0.0
    terminated = truncated = False

    for step_index in range(max_steps):
        planner_options: dict[str, int | float] = {}
        if rollout_count is not None:
            planner_options["rollout_count"] = rollout_count
        if temperature is not None:
            planner_options["temperature"] = temperature
        action = planner.plan(observation, horizon=horizon, **planner_options)
        validate_array_backend(action, adapter, name="planner action")
        next_observation, reward, terminated, truncated, info = env.step(action)
        validate_array_backend(next_observation, adapter, name="step observation")
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

    step_limit_reached = not terminated and not truncated and len(steps) == max_steps
    return EpisodeRecord(
        seed=seed,
        horizon=horizon,
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        step_limit_reached=step_limit_reached,
        rollout_count=rollout_count,
        temperature=temperature,
    )
