import json

from typer.testing import CliRunner

from forkscope.cli import app
from tests.helpers import ConfigurableCountingPlanner, CountingEnvironment, CountingPlanner


def test_cli_run_uses_factories_and_writes_results(tmp_path) -> None:
    CountingEnvironment.instances.clear()
    CountingPlanner.instances.clear()
    output = tmp_path / "cli-results.jsonl"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run",
            "--env-factory",
            "tests.helpers:make_environment",
            "--planner-factory",
            "tests.helpers:make_planner",
            "--seeds",
            "0,2",
            "--horizons",
            "1,3",
            "--max-steps",
            "3",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote 4 episode records" in result.output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    assert [planner.seed for planner in CountingPlanner.instances] == [0, 2, 0, 2]


def test_cli_accepts_optional_parameter_ranges(tmp_path) -> None:
    ConfigurableCountingPlanner.instances.clear()
    output = tmp_path / "range-results.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--env-factory", "tests.helpers:make_environment",
            "--planner-factory", "tests.helpers:make_configurable_planner",
            "--seeds", "0",
            "--horizons", "1",
            "--rollout-counts", "8,16",
            "--temperatures", "0.25,0.5",
            "--max-steps", "1",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert len(ConfigurableCountingPlanner.instances) == 4
    expected_configurations = {
        (8, 0.25), (8, 0.5), (16, 0.25), (16, 0.5)
    }
    assert {planner.configurations[0] for planner in ConfigurableCountingPlanner.instances} == (
        expected_configurations
    )
    assert {(row["rollout_count"], row["temperature"]) for row in rows} == (
        expected_configurations
    )


def test_cli_rejects_invalid_temperature_range(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--env-factory", "tests.helpers:make_environment",
            "--planner-factory", "tests.helpers:make_planner",
            "--temperatures", "nan",
            "--output", str(tmp_path / "results.jsonl"),
        ],
    )

    assert result.exit_code != 0
    assert "--temperatures" in result.output


def test_cli_rejects_malformed_factory_path() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--env-factory",
            "missing-separator",
            "--planner-factory",
            "tests.helpers:make_planner",
        ],
    )

    assert result.exit_code != 0
    assert "package.module:factory" in result.output
