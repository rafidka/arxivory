#!/usr/bin/env python3
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Dict, Any, Optional, List
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request

import typer
import cohere
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import numpy as np

BASE_URL = "https://oaipmh.arxiv.org/oai"

app = typer.Typer(
    help="ArXiv Semantic Search - Find papers using natural language queries"
)
console = Console()


def _build_url(params: Dict[str, str]) -> str:
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def _list_records(
    from_date: Optional[str],
    until_date: Optional[str],
    set_spec: str,
    metadata_prefix: str,
) -> Iterable[ET.Element]:
    """
    Iterate OAI-PMH <record> elements for the given window.
    Uses ListRecords with resumptionToken handling.
    """
    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix, "set": set_spec}
    if from_date:
        params["from"] = from_date  # YYYY-MM-DD
    if until_date:
        params["until"] = until_date

    url = _build_url(params)
    token = None

    while True:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        root = ET.fromstring(data)

        # Namespace helpers
        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "arxiv": "http://arxiv.org/OAI/arXiv/",
        }

        # Yield records
        for rec in root.findall(".//oai:record", ns):
            yield rec

        # Handle resumptionToken
        rt = root.find(".//oai:resumptionToken", ns)
        token = (rt.text or "").strip() if rt is not None else ""
        if not token:
            break
        # When using a resumptionToken, you must not pass other params.
        url = _build_url({"verb": "ListRecords", "resumptionToken": token})


def _extract_record(rec: ET.Element) -> Dict[str, Any]:
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "arxiv": "http://arxiv.org/OAI/arXiv/",
    }
    header = rec.find("oai:header", ns)
    meta = rec.find("oai:metadata", ns)
    deleted = header.get("status") == "deleted" if header is not None else False

    out: Dict[str, Any] = {
        "identifier": None,
        "datestamp": None,
        "deleted": deleted,
        "arxiv_id": None,
        "version": None,
        "title": None,
        "abstract": None,
        "authors": [],
        "categories": [],
        "created": None,
        "updated": None,
        "doi": None,
        "license": None,
        "journal_ref": None,
        "comments": None,
        "links": {},
    }

    if header is not None:
        out["identifier"] = (
            header.findtext("oai:identifier", default="", namespaces=ns) or ""
        ).strip()
        out["datestamp"] = (
            header.findtext("oai:datestamp", default="", namespaces=ns) or ""
        ).strip()

    if meta is not None:
        ar = meta.find("arxiv:arXiv", ns)
        if ar is not None:
            out["arxiv_id"] = (ar.findtext("arxiv:id", namespaces=ns) or "").strip()
            out["version"] = (ar.findtext("arxiv:version", namespaces=ns) or "").strip()
            out["title"] = (ar.findtext("arxiv:title", namespaces=ns) or "").strip()
            out["abstract"] = (
                ar.findtext("arxiv:abstract", namespaces=ns) or ""
            ).strip()
            out["created"] = (ar.findtext("arxiv:created", namespaces=ns) or "").strip()
            out["updated"] = (ar.findtext("arxiv:updated", namespaces=ns) or "").strip()
            out["doi"] = (ar.findtext("arxiv:doi", namespaces=ns) or "").strip()
            out["license"] = (ar.findtext("arxiv:license", namespaces=ns) or "").strip()
            out["journal_ref"] = (
                ar.findtext("arxiv:journal-ref", namespaces=ns) or ""
            ).strip()
            out["comments"] = (
                ar.findtext("arxiv:comments", namespaces=ns) or ""
            ).strip()

            # authors
            out["authors"] = []
            for a in ar.findall("arxiv:authors/arxiv:author", ns):
                name = (a.findtext("arxiv:keyname", namespaces=ns) or "").strip()
                given = (a.findtext("arxiv:forenames", namespaces=ns) or "").strip()
                suffix = (a.findtext("arxiv:suffix", namespaces=ns) or "").strip()
                full = " ".join(x for x in [given, name, suffix] if x)
                out["authors"].append({"name": full or name})

            # categories
            primary = (
                ar.findtext("arxiv:primary_category", namespaces=ns) or ""
            ).strip()
            if primary:
                out["categories"].append(primary)
            for c in ar.findall("arxiv:categories", ns):
                cats = (c.text or "").split()
                for cat in cats:
                    if cat and cat not in out["categories"]:
                        out["categories"].append(cat)

            # links
            if out["arxiv_id"]:
                aid = out["arxiv_id"]
                out["links"] = {
                    "abs": f"https://arxiv.org/abs/{aid}",
                    "pdf": f"https://arxiv.org/pdf/{aid}.pdf",
                }

    return out


def compute_window(preset: str) -> tuple[str, str]:
    """Return (from, until) in UTC date format YYYY-MM-DD inclusive."""
    today = datetime.now(timezone.utc).date()
    if preset == "yesterday":
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if preset == "last-week":
        # ISO “last week” = last 7 complete days ending yesterday
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        return start.isoformat(), end.isoformat()
    raise ValueError("unknown preset")


def get_cohere_client() -> cohere.Client:
    """Get Cohere client from environment variable."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        console.print("[red]Error: COHERE_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    return cohere.Client(api_key=api_key)


def harvest_papers(
    from_date: str,
    until_date: str,
    set_spec: str = "cs",
    metadata_prefix: str = "arXiv",
) -> List[Dict[str, Any]]:
    """Harvest papers from arXiv for the given date range."""
    papers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Harvesting papers from arXiv...", total=None)

        for rec in _list_records(from_date, until_date, set_spec, metadata_prefix):
            paper = _extract_record(rec)
            if (
                not paper.get("deleted")
                and paper.get("title")
                and paper.get("abstract")
            ):
                papers.append(paper)

        progress.update(task, description=f"Found {len(papers)} papers")

    return papers


def semantic_search(
    papers: List[Dict[str, Any]], query: str, top_k: int = 10
) -> List[tuple[Dict[str, Any], float]]:
    """Perform semantic search using Cohere embeddings and reranking."""
    if not papers:
        return []

    client = get_cohere_client()

    # Prepare documents for reranking (title + abstract)
    documents = []
    for paper in papers:
        doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        documents.append(doc_text)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Performing semantic search with Cohere...", total=None
        )

        # Use Cohere Rerank 3.5 to find most relevant papers
        response = client.rerank(
            query=query,
            documents=documents,
            model="rerank-v3.5",
            top_n=min(top_k, len(documents)),
        )

        progress.update(task, description="Search completed")

    # Return papers with their relevance scores
    results = []
    for result in response.results:
        paper = papers[result.index]
        results.append((paper, result.relevance_score))

    return results


def format_results(
    results: List[tuple[Dict[str, Any], float]], show_abstract: bool = False
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

        row = [str(i), f"{score:.3f}", title, authors, arxiv_id]

        if show_abstract:
            abstract = paper.get("abstract", "N/A")
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            row.append(abstract)

        table.add_row(*row)

    console.print(table)

    # Show links for top result
    if results:
        top_paper = results[0][0]
        if top_paper.get("links"):
            console.print(f"\n[bold]Top result links:[/bold]")
            console.print(f"Abstract: {top_paper['links'].get('abs', 'N/A')}")
            console.print(f"PDF: {top_paper['links'].get('pdf', 'N/A')}")


@app.command()
def search(
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
        10, "--top-k", "-k", help="Number of top results to return"
    ),
    set_spec: str = typer.Option(
        "cs", "--set", help="arXiv set (default: cs for Computer Science)"
    ),
    show_abstract: bool = typer.Option(
        False, "--abstract", "-a", help="Show abstracts in results"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Search arXiv papers using natural language queries with semantic search."""

    # Determine date range
    if preset:
        if preset not in ["yesterday", "last-week"]:
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
    console.print()

    try:
        # Harvest papers
        papers = harvest_papers(from_date_computed, until_date_computed, set_spec)

        if not papers:
            console.print(
                "[yellow]No papers found in the specified date range.[/yellow]"
            )
            return

        console.print(f"Found {len(papers)} papers. Performing semantic search...")

        # Perform semantic search
        results = semantic_search(papers, query, top_k)

        if json_output:
            # Output as JSON
            json_results = []
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


@app.command()
def harvest(
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
        "cs", "--set", help="arXiv set (default: cs for Computer Science)"
    ),
    metadata_prefix: str = typer.Option("arXiv", "--prefix", help="Metadata format"),
):
    """Harvest arXiv metadata without semantic search (original functionality)."""

    # Determine date range
    if preset:
        if preset not in ["yesterday", "last-week"]:
            console.print(
                f"[red]Error: Invalid preset '{preset}'. Use 'yesterday' or 'last-week'.[/red]",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        from_date_computed, until_date_computed = compute_window(preset)
    else:
        if not from_date:
            console.print(
                "[red]Error: Provide either --preset or --from-date.[/red]",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        from_date_computed = from_date
        until_date_computed = until_date or from_date

    # Harvest and output raw JSON (original functionality)
    for rec in _list_records(
        from_date_computed, until_date_computed, set_spec, metadata_prefix
    ):
        obj = _extract_record(rec)
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    app()


if __name__ == "__main__":
    main()
