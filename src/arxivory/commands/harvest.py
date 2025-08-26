"""Harvest command implementation."""

import sys

import typer
from rich.console import Console

from arxivory.arxiv import compute_window, harvest_raw_papers
from arxivory.constants import (
    DEFAULT_METADATA_PREFIX,
    DEFAULT_SET_SPEC,
    VALID_PRESETS,
)

console = Console()


def harvest_command(
    preset: str = typer.Option(
        None, "--preset", "-p", help="Date preset: yesterday, last-week"
    ),
    from_date: str = typer.Option(
        None, "--from-date", "-f", help="Start date (YYYY-MM-DD)"
    ),
    until_date: str = typer.Option(
        None, "--until-date", "-u", help="End date (YYYY-MM-DD)"
    ),
    set_spec: str = typer.Option(
        DEFAULT_SET_SPEC, "--set", help="arXiv set (default: cs for Computer Science)"
    ),
    metadata_prefix: str = typer.Option(
        DEFAULT_METADATA_PREFIX, "--prefix", help="Metadata format"
    ),
):
    """Harvest arXiv metadata without semantic search (original functionality)."""
    # Determine date range
    if preset:
        if preset not in VALID_PRESETS:
            print(
                f"Error: Invalid preset '{preset}'. Use 'yesterday' or 'last-week'.",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        from_date_computed, until_date_computed = compute_window(preset)
    else:
        if not from_date:
            print(
                "Error: Provide either --preset or --from-date.",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        from_date_computed = from_date
        until_date_computed = until_date or from_date

    # Harvest and output raw JSON (original functionality)
    harvest_raw_papers(
        from_date_computed, until_date_computed, set_spec, metadata_prefix
    )
