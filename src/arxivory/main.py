#!/usr/bin/env python3
"""Main CLI application for ArXiv Semantic Search."""

import typer

from arxivory.commands.harvest import harvest_command
from arxivory.commands.search import search_command

app = typer.Typer(
    help="ArXiv Semantic Search - Find papers using natural language queries"
)

# Register commands
app.command("search")(search_command)
app.command("harvest")(harvest_command)


def main():
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
