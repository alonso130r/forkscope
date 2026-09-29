import json

import numpy as np
import pytest

from forkscope import RolloutBenchmarker
from forkscope.evaluation.episode import run_episode
from forkscope.evaluation.sweep import run_sweep
from forkscope.records.writer import write_jsonl
from tests.helpers import CountingEnvironment, CountingPlanner, make_environment, make_planner


@pytest.fixture(autouse=True)
def clear_fixture_instances() -> None:
    CountingEnvironment.instances.clear()
    CountingPlanner.instances.clear()


def test_episode_resets_with_seed_and_records_transitions() -> None:
    env = make_environment()
    planner = make_planner()

    record = run_episode(env, planner, seed=7, horizon=3, max_steps=10)

    assert env.seed == planner.seed == 7
    assert env.actions == [3, 3]
    assert [step.observation.tolist() for step in record.steps] == [[0.0], [1.0]]
    assert [step.reward for step in record.steps] == [1.0, 1.0]
    assert record.total_reward == 2.0
    assert record.terminated is True
    assert record.truncated is False
    assert record.step_limit_reached is False
    assert record.step_count == 2


def test_episode_marks_runner_step_limit() -> None:
    env = make_environment()

    record = run_episode(env, make_planner(), seed=2, horizon=1, max_steps=1)

    assert record.terminated is False
    assert record.truncated is False
    assert record.step_limit_reached is True


@pytest.mark.parametrize(("horizon", "max_steps"), [(0, 2), (1, 0), (1, -1)])
def test_episode_rejects_nonpositive_horizon_or_step_limit(
    horizon: int, max_steps: int
) -> None:
    with pytest.raises(ValueError):
        run_episode(
            make_environment(),
            make_planner(),
            seed=1,
            horizon=horizon,
            max_steps=max_steps,
        )


def test_sweep_uses_matched_seeds_and_fresh_components() -> None:
    records = run_sweep(
        make_environment,
        make_planner,
        seeds=[4, 9],
        horizons=[1, 5],
        max_steps=3,
    )

    assert [(record.horizon, record.seed) for record in records] == [
        (1, 4),
        (1, 9),
        (5, 4),
        (5, 9),
    ]
    assert len({id(env) for env in CountingEnvironment.instances}) == 4
    assert len({id(planner) for planner in CountingPlanner.instances}) == 4
    assert [planner.seed for planner in CountingPlanner.instances] == [4, 9, 4, 9]
    assert all(env.closed for env in CountingEnvironment.instances)


def test_sweep_uses_cartesian_optional_planner_settings() -> None:
    class ConfigurablePlanner(CountingPlanner):
        def plan(self, observation, *, horizon, rollout_count=None, temperature=None):
            self.configurations = getattr(self, "configurations", [])
            self.configurations.append((horizon, rollout_count, temperature))
            return super().plan(observation, horizon=horizon)

    records = run_sweep(
        make_environment,
        ConfigurablePlanner,
        seeds=[2],
        horizons=[1, 3],
        rollout_counts=[8, 16],
        temperatures=[0.25, 0.5],
        max_steps=1,
    )

    assert len(records) == 8
    assert {
        (record.horizon, record.rollout_count, record.temperature) for record in records
    } == {
        (horizon, rollout_count, temperature)
        for horizon in [1, 3]
        for rollout_count in [8, 16]
        for temperature in [0.25, 0.5]
    }


def test_unconfigured_planner_keeps_existing_plan_signature() -> None:
    records = run_sweep(
        make_environment,
        make_planner,
        seeds=[1],
        horizons=[2],
        max_steps=1,
    )

    assert records[0].rollout_count is None
    assert records[0].temperature is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rollout_counts": []}, "rollout_counts"),
        ({"rollout_counts": [0]}, "rollout_counts"),
        ({"rollout_counts": [1.5]}, "rollout_counts"),
        ({"temperatures": []}, "temperatures"),
        ({"temperatures": [0]}, "temperatures"),
        ({"temperatures": [float("nan")]}, "temperatures"),
    ],
)
def test_benchmarker_rejects_invalid_optional_ranges(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RolloutBenchmarker(
            make_environment,
            make_planner,
            seeds=[1],
            horizons=[1],
            max_steps=1,
            **kwargs,
        )


def test_benchmarker_stores_and_returns_results_and_writes_jsonl(tmp_path) -> None:
    benchmarker = RolloutBenchmarker(
        make_environment,
        make_planner,
        seeds=[3],
        horizons=[2],
        max_steps=3,
    )

    returned = benchmarker.run()
    output_path = tmp_path / "nested" / "episodes.jsonl"
    benchmarker.write_results(output_path)

    assert returned is benchmarker.results
    assert len(benchmarker.results) == 1
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["seed"] == 3
    assert row["horizon"] == 2
    assert row["total_reward"] == 2.0
    assert row["steps"][0]["observation"] == [0.0]


def test_benchmarker_rejects_invalid_run_settings() -> None:
    with pytest.raises(ValueError, match="seeds"):
        RolloutBenchmarker(make_environment, make_planner, seeds=[], horizons=[1], max_steps=1)
    with pytest.raises(ValueError, match="horizons"):
        RolloutBenchmarker(make_environment, make_planner, seeds=[1], horizons=[0], max_steps=1)
    with pytest.raises(ValueError, match="max_steps"):
        RolloutBenchmarker(make_environment, make_planner, seeds=[1], horizons=[1], max_steps=0)


def test_writer_converts_numpy_values_to_json(tmp_path) -> None:
    record = run_episode(make_environment(), make_planner(), seed=1, horizon=2, max_steps=3)
    path = tmp_path / "record.jsonl"

    write_jsonl([record], path)

    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["steps"][0]["observation"] == [0.0]
    assert isinstance(decoded["steps"][0]["reward"], float)
    assert decoded["steps"][0]["info"]["action_count"] == 1


class PositionEnvironment:
    observation_space = None
    action_space = None

    def __init__(self) -> None:
        self.seed = 0
        self.position = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        self.seed = 0 if seed is None else seed
        self.position = 0
        self.steps = 0
        return {"position": np.array([self.position], dtype=np.int64)}, {}

    def step(self, action: int):
        reward = float((self.seed + 1) * action - self.steps)
        self.position += action
        self.steps += 1
        terminated = self.steps == 3
        observation = {"position": np.array([self.position], dtype=np.int64)}
        return observation, reward, terminated, False, {"position": self.position}

    def close(self) -> None:
        pass


class PositionPlanner:
    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def plan(self, observation, *, horizon: int) -> int:
        return horizon + int(observation["position"][0])


def test_benchmark_outputs_match_hand_calculated_episode_records(tmp_path) -> None:
    benchmarker = RolloutBenchmarker(
        PositionEnvironment,
        PositionPlanner,
        seeds=[0, 2],
        horizons=[1, 3],
        max_steps=5,
    )

    results = benchmarker.run()
    output_path = tmp_path / "golden.jsonl"
    benchmarker.write_results(output_path)
    actual_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    expected_rows = [
        _expected_position_episode(seed=0, horizon=1, rewards=[1.0, 1.0, 2.0]),
        _expected_position_episode(seed=2, horizon=1, rewards=[3.0, 5.0, 10.0]),
        _expected_position_episode(seed=0, horizon=3, rewards=[3.0, 5.0, 10.0]),
        _expected_position_episode(seed=2, horizon=3, rewards=[9.0, 17.0, 34.0]),
    ]

    assert len(results) == 4
    assert actual_rows == expected_rows


def _expected_position_episode(seed: int, horizon: int, rewards: list[float]) -> dict:
    positions = [0, horizon, 3 * horizon]
    actions = [horizon, 2 * horizon, 4 * horizon]
    next_positions = [horizon, 3 * horizon, 7 * horizon]
    steps = [
        {
            "index": index,
            "observation": {"position": [position]},
            "action": action,
            "reward": reward,
            "terminated": index == 2,
            "truncated": False,
            "info": {"position": next_position},
        }
        for index, (position, action, reward, next_position) in enumerate(
            zip(positions, actions, rewards, next_positions, strict=True)
        )
    ]
    return {
        "seed": seed,
        "horizon": horizon,
        "rollout_count": None,
        "temperature": None,
        "total_reward": sum(rewards),
        "steps": steps,
        "terminated": True,
        "truncated": False,
        "step_limit_reached": False,
    }
