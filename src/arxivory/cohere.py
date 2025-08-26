"""Cohere-related functionality for semantic search."""

import os
from typing import Any

import cohere
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from arxivory.constants import (
    DEFAULT_RETRIEVAL_K,
    DEFAULT_TOP_K,
    EMBED_MODEL,
    RERANK_MODEL,
    STRATEGY_EMBED_RERANK,
    STRATEGY_RERANK_ONLY,
)

console = Console()


def get_cohere_client() -> cohere.Client:
    """Get Cohere client from environment variable."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        console.print("[red]Error: COHERE_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
    return cohere.Client(api_key=api_key)


def _rerank_only_search(
    papers: list[dict[str, Any]],
    documents: list[str],
    query: str,
    top_k: int,
    client: cohere.Client,
    strategy: str,
) -> list[tuple[dict[str, Any], float]]:
    """Perform rerank-only semantic search strategy."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        strategy_msg = (
            "user-requested rerank-only strategy"
            if strategy == STRATEGY_RERANK_ONLY
            else "skipping embedding stage for small dataset"
        )
        task = progress.add_task(
            f"Reranking all {len(papers)} papers ({strategy_msg})...", total=None
        )

        response = client.rerank(
            query=query,
            documents=documents,
            model=RERANK_MODEL,
            top_n=min(top_k, len(documents)),
        )

        progress.update(task, description="Search completed")

    # Return papers with their relevance scores
    results: list[tuple[dict[str, Any], float]] = []
    for result in response.results:
        paper = papers[result.index]
        results.append((paper, result.relevance_score))

    return results


def _embed_rerank_search(
    papers: list[dict[str, Any]],
    documents: list[str],
    query: str,
    top_k: int,
    retrieval_k: int,
    client: cohere.Client,
) -> list[tuple[dict[str, Any], float]]:
    """Perform embed-then-rerank semantic search strategy."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Stage 1: Embedding-based retrieval
        embed_task = progress.add_task(
            f"Stage 1: Embedding {len(papers)} papers with Cohere Embed v4...",
            total=None,
        )

        # Get embeddings for query and all documents
        query_embed_response = client.embed(
            texts=[query], model=EMBED_MODEL, input_type="search_query"
        )

        doc_embed_response = client.embed(
            texts=documents, model=EMBED_MODEL, input_type="search_document"
        )

        query_embedding = np.array(query_embed_response.embeddings[0])  # type: ignore
        doc_embeddings = np.array(doc_embed_response.embeddings)  # type: ignore

        progress.update(embed_task, description="Computing cosine similarities...")

        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top candidates for reranking
        candidates_count = min(retrieval_k, len(papers))
        top_indices = np.argsort(similarities)[::-1][:candidates_count]

        progress.update(
            embed_task, description=f"Retrieved top {candidates_count} candidates"
        )

        # Stage 2: Reranking on candidates
        rerank_task = progress.add_task(
            f"Stage 2: Reranking top {candidates_count} candidates...", total=None
        )

        # Prepare candidates for reranking
        candidate_documents: list[str] = [documents[i] for i in top_indices]
        candidate_papers: list[dict[str, Any]] = [papers[i] for i in top_indices]

        # Use Cohere Rerank 3.5 on candidates only
        rerank_response = client.rerank(
            query=query,
            documents=candidate_documents,
            model=RERANK_MODEL,
            top_n=min(top_k, len(candidate_documents)),
        )

        progress.update(rerank_task, description="Search completed")

    # Return final ranked papers with relevance scores
    results: list[tuple[dict[str, Any], float]] = []
    for result in rerank_response.results:
        paper = candidate_papers[result.index]
        results.append((paper, result.relevance_score))

    return results


def semantic_search(
    papers: list[dict[str, Any]],
    query: str,
    top_k: int = DEFAULT_TOP_K,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    strategy: str = STRATEGY_EMBED_RERANK,
) -> list[tuple[dict[str, Any], float]]:
    """Perform semantic search with configurable strategy.

    Strategies:
    - "embed-rerank" (default): Two-stage approach for speed and cost-efficiency
      - Stage 1: Use Cohere Embed v4 to retrieve top candidates via cosine similarity
      - Stage 2: Use Cohere Rerank 3.5 on candidates for final ranking
    - "rerank-only": Single-stage reranking for maximum accuracy (slower, more expensive)
      - Uses Cohere Rerank 3.5 on all documents directly
    """
    if not papers:
        return []

    client = get_cohere_client()

    # Prepare documents for embedding (title + abstract)
    documents: list[str] = []
    for paper in papers:
        doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        documents.append(doc_text)

    # If strategy is rerank-only or we have few documents, use rerank-only approach
    if strategy == STRATEGY_RERANK_ONLY or len(papers) <= retrieval_k:
        return _rerank_only_search(papers, documents, query, top_k, client, strategy)

    # Otherwise use embed-rerank strategy
    return _embed_rerank_search(papers, documents, query, top_k, retrieval_k, client)
