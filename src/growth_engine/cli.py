"""Command-line entry point: `growth-engine report --out dist`."""

from __future__ import annotations

from pathlib import Path

import typer

from .report import write_report

app = typer.Typer(add_completion=False, help="Growth analytics engine.")


@app.callback()
def main() -> None:
    """Attribution, cohorts, uplift and unit economics on a synthetic event stream."""


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    seed: int = typer.Option(42, help="Seed for the synthetic event stream."),
    users: int = typer.Option(6000, help="Number of prospects to simulate."),
    months: int = typer.Option(18, help="Months of activity to simulate."),
) -> None:
    """Generate the growth report as a static site."""
    path = write_report(out, seed=seed, n_users=users, months=months)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
