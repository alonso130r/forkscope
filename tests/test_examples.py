from forkscope import RolloutBenchmarker
from tests.examples import BoundedGreedyPlanner, PointMassEnvironment


def test_point_mass_horizon_sweep_matches_expected_task_outcomes() -> None:
    benchmarker = RolloutBenchmarker(
        PointMassEnvironment,
        BoundedGreedyPlanner,
        seeds=[0, 1],
        horizons=[1, 3],
        max_steps=8,
    )

    records = benchmarker.run()

    actual = [
        (
            record.seed,
            record.horizon,
            record.total_reward,
            [step.action for step in record.steps],
            record.terminated,
            record.step_limit_reached,
        )
        for record in records
    ]
    assert actual == [
        (0, 1, -10.0, [1.0, 1.0, 1.0, 1.0, 1.0], True, False),
        (1, 1, -15.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], True, False),
        (0, 3, -2.0, [3.0, 2.0], True, False),
        (1, 3, -3.0, [3.0, 3.0], True, False),
    ]


def test_point_mass_records_step_limit_when_budget_ends_before_goal() -> None:
    benchmarker = RolloutBenchmarker(
        PointMassEnvironment,
        BoundedGreedyPlanner,
        seeds=[0],
        horizons=[1],
        max_steps=3,
    )

    record = benchmarker.run()[0]

    assert [step.info["distance"] for step in record.steps] == [4.0, 3.0, 2.0]
    assert record.total_reward == -9.0
    assert record.step_count == 3
    assert record.terminated is False
    assert record.step_limit_reached is True
