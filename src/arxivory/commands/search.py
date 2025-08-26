"""Search command implementation."""

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from arxivory.arxiv import compute_window, harvest_papers
from arxivory.cohere import semantic_search
from arxivory.constants import (
    DEFAULT_RETRIEVAL_K,
    DEFAULT_SET_SPEC,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    PAPERLENS_BASE_URL,
    STRATEGY_EMBED_RERANK,
    VALID_PRESETS,
    VALID_STRATEGIES,
)

console = Console()


def generate_paperlens_url(arxiv_id: str) -> str:
    """Generate PaperLens URL for the given arXiv ID."""
    return f"{PAPERLENS_BASE_URL}/?paper={arxiv_id}&source=arxivory"


def format_results(
    results: list[tuple[dict[str, Any], float]], show_abstract: bool = False
):
    """Format and display search results."""
    if not results:
        console.print("[yellow]No papers found matching your query.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Score", style="green", width=6)
    table.add_column("Title", style="bold")
    table.add_column("Authors", style="dim", width=30)
    table.add_column("arXiv ID", style="blue", width=12)

    if show_abstract:
        table.add_column("Abstract", width=50)

    for i, (paper, score) in enumerate(results, 1):
        authors = ", ".join(
            [author.get("name", "") for author in paper.get("authors", [])][:3]
        )
        if len(paper.get("authors", [])) > 3:
            authors += " et al."

        title = paper.get("title", "N/A")
        arxiv_id = paper.get("arxiv_id", "N/A")

        # Add PaperLens URL under the title
        if arxiv_id and arxiv_id != "N/A":
            paperlens_url = generate_paperlens_url(arxiv_id)
            title_with_link = f"{title}\n📱 [dim]{paperlens_url}[/dim]"
        else:
            title_with_link = title

        row = [str(i), f"{score:.3f}", title_with_link, authors, arxiv_id]

        if show_abstract:
            abstract = paper.get("abstract", "N/A")
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            row.append(abstract)

        table.add_row(*row)

    console.print(table)


def search_command(
    query: str = typer.Argument(..., help="Natural language search query"),
    preset: str = typer.Option(
        None, "--preset", "-p", help="Date preset: yesterday, last-week"
    ),
    from_date: str = typer.Option(
        None, "--from-date", "-f", help="Start date (YYYY-MM-DD)"
    ),
    until_date: str = typer.Option(
        None, "--until-date", "-u", help="End date (YYYY-MM-DD)"
    ),
    top_k: int = typer.Option(
        DEFAULT_TOP_K, "--top-k", "-k", help="Number of top results to return"
    ),
    retrieval_k: int = typer.Option(
        DEFAULT_RETRIEVAL_K,
        "--retrieval-k",
        "-r",
        help="Number of candidates to retrieve before reranking (default: 100)",
    ),
    strategy: str = typer.Option(
        DEFAULT_STRATEGY,
        "--strategy",
        "-s",
        help="Search strategy: 'embed-rerank' (faster, default) or 'rerank-only' (more accurate)",
    ),
    set_spec: str = typer.Option(
        DEFAULT_SET_SPEC, "--set", help="arXiv set (default: cs for Computer Science)"
    ),
    show_abstract: bool = typer.Option(
        False, "--abstract", "-a", help="Show abstracts in results"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Search arXiv papers using natural language queries with semantic search."""
    # Validate strategy
    if strategy not in VALID_STRATEGIES:
        console.print(
            f"[red]Error: Invalid strategy '{strategy}'. Use 'embed-rerank' or 'rerank-only'.[/red]"
        )
        raise typer.Exit(1)

    # Determine date range
    if preset:
        if preset not in VALID_PRESETS:
            console.print(
                f"[red]Error: Invalid preset '{preset}'. Use 'yesterday' or 'last-week'.[/red]"
            )
            raise typer.Exit(1)
        from_date_computed, until_date_computed = compute_window(preset)
    else:
        if not from_date:
            console.print("[red]Error: Provide either --preset or --from-date.[/red]")
            raise typer.Exit(1)
        from_date_computed = from_date
        until_date_computed = until_date or from_date

    console.print(
        f"[bold]Searching arXiv papers from {from_date_computed} to {until_date_computed}[/bold]"
    )
    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[bold]Strategy:[/bold] {strategy}")
    console.print()

    try:
        # Harvest papers
        papers = harvest_papers(from_date_computed, until_date_computed, set_spec)

        if not papers:
            console.print(
                "[yellow]No papers found in the specified date range.[/yellow]"
            )
            return

        strategy_desc = (
            "two-stage semantic search (retrieve-then-rerank)"
            if strategy == STRATEGY_EMBED_RERANK
            else "single-stage reranking"
        )
        console.print(f"Found {len(papers)} papers. Performing {strategy_desc}...")

        # Perform semantic search
        results = semantic_search(papers, query, top_k, retrieval_k, strategy)

        if json_output:
            # Output as JSON
            json_results: list[dict[str, Any]] = []
            for paper, score in results:
                result_obj = paper.copy()
                result_obj["relevance_score"] = score
                json_results.append(result_obj)
            print(json.dumps(json_results, indent=2, ensure_ascii=False))
        else:
            # Format and display results
            format_results(results, show_abstract)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
