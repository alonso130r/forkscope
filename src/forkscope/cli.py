"""Command-line entry point for Forkscope."""

import importlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from forkscope import RolloutBenchmarker

app = typer.Typer(help="Evaluate test-time planning rollouts on your scenarios.")
OUTPUT_OPTION = typer.Option(Path("results.jsonl"), help="JSONL output path.")


@app.callback()
def main() -> None:
    """Run ForkScope commands."""


def _load_factory(spec: str) -> Callable[[], Any]:
    """Load a zero-argument factory from a ``module:attribute`` path."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise typer.BadParameter("Use an import path in the form 'package.module:factory'.")

    try:
        factory = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise typer.BadParameter(f"Could not load factory {spec!r}: {exc}") from exc
    if not callable(factory):
        raise typer.BadParameter(f"Configured factory {spec!r} is not callable.")
    return factory


def _parse_ints(value: str, option: str, *, minimum: int) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be comma-separated integers.") from exc
    if not values or any(item < minimum for item in values):
        comparison = "non-negative" if minimum == 0 else "greater than zero"
        raise typer.BadParameter(f"{option} values must be {comparison}.")
    return values


def _parse_positive_floats(value: str, option: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be comma-separated numbers.") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise typer.BadParameter(f"{option} values must be finite and greater than zero.")
    return values


@app.command()
def run(
    env_factory: str = typer.Option(..., help="Environment factory as package.module:callable."),
    planner_factory: str = typer.Option(..., help="Planner factory as package.module:callable."),
    seeds: str = typer.Option("1,2,3,4,5", help="Comma-separated episode seeds."),
    horizons: str = typer.Option("1,2,4,8,16", help="Comma-separated planning horizons."),
    max_steps: int = typer.Option(500, min=1, help="Maximum environment steps per episode."),
    rollout_counts: str | None = typer.Option(
        None, help="Optional comma-separated planner rollout counts to sweep."
    ),
    temperatures: str | None = typer.Option(
        None, help="Optional comma-separated planner sampling temperatures to sweep."
    ),
    output: Path = OUTPUT_OPTION,
) -> None:
    """Run a matched-seed horizon sweep and write its episode records."""
    benchmarker = RolloutBenchmarker(
        _load_factory(env_factory),
        _load_factory(planner_factory),
        seeds=_parse_ints(seeds, "--seeds", minimum=0),
        horizons=_parse_ints(horizons, "--horizons", minimum=1),
        max_steps=max_steps,
        rollout_counts=(
            _parse_ints(rollout_counts, "--rollout-counts", minimum=1)
            if rollout_counts is not None
            else None
        ),
        temperatures=(
            _parse_positive_floats(temperatures, "--temperatures")
            if temperatures is not None
            else None
        ),
    )
    benchmarker.run()
    benchmarker.write_results(output)
    typer.echo(f"Wrote {len(benchmarker.results)} episode records to {output}")


if __name__ == "__main__":
    app()
