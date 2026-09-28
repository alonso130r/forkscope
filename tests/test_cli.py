from typer.testing import CliRunner

from forkscope.cli import app
from tests.helpers import CountingEnvironment, CountingPlanner


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
