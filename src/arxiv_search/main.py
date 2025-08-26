#!/usr/bin/env python3
import json
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Dict, Any, Optional
import argparse
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request

BASE_URL = "https://oaipmh.arxiv.org/oai"

def _build_url(params: Dict[str, str]) -> str:
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"

def _list_records(from_date: Optional[str], until_date: Optional[str], set_spec: str, metadata_prefix: str) -> Iterable[ET.Element]:
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
            "arxiv": "http://arxiv.org/OAI/arXiv/"
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
        "arxiv": "http://arxiv.org/OAI/arXiv/"
    }
    header = rec.find("oai:header", ns)
    meta   = rec.find("oai:metadata", ns)
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
        "links": {}
    }

    if header is not None:
        out["identifier"] = (header.findtext("oai:identifier", default="", namespaces=ns) or "").strip()
        out["datestamp"]  = (header.findtext("oai:datestamp",  default="", namespaces=ns) or "").strip()

    if meta is not None:
        ar = meta.find("arxiv:arXiv", ns)
        if ar is not None:
            out["arxiv_id"]  = (ar.findtext("arxiv:id", namespaces=ns) or "").strip()
            out["version"]   = (ar.findtext("arxiv:version", namespaces=ns) or "").strip()
            out["title"]     = (ar.findtext("arxiv:title", namespaces=ns) or "").strip()
            out["abstract"]  = (ar.findtext("arxiv:abstract", namespaces=ns) or "").strip()
            out["created"]   = (ar.findtext("arxiv:created", namespaces=ns) or "").strip()
            out["updated"]   = (ar.findtext("arxiv:updated", namespaces=ns) or "").strip()
            out["doi"]       = (ar.findtext("arxiv:doi", namespaces=ns) or "").strip()
            out["license"]   = (ar.findtext("arxiv:license", namespaces=ns) or "").strip()
            out["journal_ref"]= (ar.findtext("arxiv:journal-ref", namespaces=ns) or "").strip()
            out["comments"]  = (ar.findtext("arxiv:comments", namespaces=ns) or "").strip()

            # authors
            out["authors"] = []
            for a in ar.findall("arxiv:authors/arxiv:author", ns):
                name = (a.findtext("arxiv:keyname", namespaces=ns) or "").strip()
                given = (a.findtext("arxiv:forenames", namespaces=ns) or "").strip()
                suffix = (a.findtext("arxiv:suffix", namespaces=ns) or "").strip()
                full = " ".join(x for x in [given, name, suffix] if x)
                out["authors"].append({"name": full or name})

            # categories
            primary = (ar.findtext("arxiv:primary_category", namespaces=ns) or "").strip()
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
                    "pdf": f"https://arxiv.org/pdf/{aid}.pdf"
                }

    return out

def compute_window(preset: str) -> (str, str):
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

def main():
    ap = argparse.ArgumentParser(description="Harvest arXiv Computer Science metadata for a date window.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=["yesterday", "last-week"], help="Common windows.")
    g.add_argument("--from-date", dest="from_date", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--until-date", dest="until_date", help="YYYY-MM-DD (UTC). If omitted with --from-date, defaults to same day.")
    ap.add_argument("--set", dest="set_spec", default="cs", help="OAI setSpec for Computer Science (default: 'cs').")
    ap.add_argument("--prefix", dest="metadata_prefix", default="arXiv", choices=["arXiv", "arXivRaw", "oai_dc"], help="Metadata format.")
    args = ap.parse_args()

    if args.preset:
        from_date, until_date = compute_window(args.preset)
    else:
        if not args.from_date:
            ap.error("Provide --preset or --from-date.")
        from_date = args.from_date
        until_date = args.until_date or args.from_date

    for rec in _list_records(from_date, until_date, args.set_spec, args.metadata_prefix):
        obj = _extract_record(rec)
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()