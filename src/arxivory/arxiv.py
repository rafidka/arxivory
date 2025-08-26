"""ArXiv-related functionality for harvesting papers."""

import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from arxivory.constants import (
    BASE_URL,
    DEFAULT_METADATA_PREFIX,
    DEFAULT_SET_SPEC,
    OAI_NAMESPACES,
)

console = Console()


def _build_url(params: dict[str, str]) -> str:
    """Build URL for OAI-PMH requests."""
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def _list_records(
    from_date: str | None,
    until_date: str | None,
    set_spec: str,
    metadata_prefix: str,
) -> Iterable[ET.Element]:
    """Iterate OAI-PMH <record> elements for the given window.

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
        ns = OAI_NAMESPACES

        # Yield records
        yield from root.findall(".//oai:record", ns)

        # Handle resumptionToken
        rt = root.find(".//oai:resumptionToken", ns)
        token = (rt.text or "").strip() if rt is not None else ""
        if not token:
            break
        # When using a resumptionToken, you must not pass other params.
        url = _build_url({"verb": "ListRecords", "resumptionToken": token})


def _extract_record(rec: ET.Element) -> dict[str, Any]:
    """Extract paper metadata from an OAI-PMH record element."""
    ns = OAI_NAMESPACES
    header = rec.find("oai:header", ns)
    meta = rec.find("oai:metadata", ns)
    deleted = header.get("status") == "deleted" if header is not None else False

    out: dict[str, Any] = {
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
            authors_list: list[dict[str, str]] = []
            for a in ar.findall("arxiv:authors/arxiv:author", ns):
                name = (a.findtext("arxiv:keyname", namespaces=ns) or "").strip()
                given = (a.findtext("arxiv:forenames", namespaces=ns) or "").strip()
                suffix = (a.findtext("arxiv:suffix", namespaces=ns) or "").strip()
                full = " ".join(x for x in [given, name, suffix] if x)
                authors_list.append({"name": full or name})
            out["authors"] = authors_list

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
    today = datetime.now(UTC).date()
    if preset == "yesterday":
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if preset == "last-week":
        # ISO "last week" = last 7 complete days ending yesterday
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        return start.isoformat(), end.isoformat()
    raise ValueError("unknown preset")


def harvest_papers(
    from_date: str,
    until_date: str,
    set_spec: str = DEFAULT_SET_SPEC,
    metadata_prefix: str = DEFAULT_METADATA_PREFIX,
) -> list[dict[str, Any]]:
    """Harvest papers from arXiv for the given date range."""
    papers: list[dict[str, Any]] = []
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


def harvest_raw_papers(
    from_date: str,
    until_date: str,
    set_spec: str = DEFAULT_SET_SPEC,
    metadata_prefix: str = DEFAULT_METADATA_PREFIX,
) -> None:
    """Harvest and output raw paper metadata to stdout (original functionality)."""
    for rec in _list_records(from_date, until_date, set_spec, metadata_prefix):
        obj = _extract_record(rec)
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
