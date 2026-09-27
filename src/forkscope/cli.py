"""Command-line entry point for Forkscope."""

import typer

app = typer.Typer(help="Evaluate test-time planning rollouts on your scenarios.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show the CLI help until runtime commands are implemented."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()

