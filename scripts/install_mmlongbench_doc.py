#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REPO = "mayubo2333/MMLongBench-Doc"
DEFAULT_REF = "main"


def default_target_dir() -> Path:
    explicit = os.environ.get("MMLONGBENCH_DOC_DIR")
    if explicit:
        return Path(explicit).expanduser()
    user = os.environ.get("USER")
    if user and Path("/scratch").exists():
        return Path("/scratch") / user / "agenticdocai" / "data" / "MMLongBench-Doc"
    return Path("data") / "MMLongBench-Doc"


def fetch_json(url: str, timeout: int, retries: int):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "AgenticDocAI-mmlongbench-installer"})
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}") from last_error


def download_file(url: str, out_path: Path, timeout: int, retries: int, force: bool = False) -> None:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"[mmlb-install] skip existing: {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "AgenticDocAI-mmlongbench-installer"})
            with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temp_path.replace(out_path)
            print(f"[mmlb-install] downloaded: {out_path}")
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url} to {out_path}: {last_error}") from last_error


def github_api_url(repo: str, path: str, ref: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"


def raw_github_url(repo: str, path: str, ref: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install MMLongBench-Doc data into the official repo layout.")
    parser.add_argument("--target-dir", default=str(default_target_dir()))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug option: download only the first N PDFs.")
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Debug option: download only PDFs required by the first N samples.json rows.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even when they already exist.")
    parser.add_argument("--manifest-only", action="store_true", help="Download samples.json but skip PDF files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()
    data_dir = target_dir / "data"
    documents_dir = data_dir / "documents"
    data_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mmlb-install] repo: {args.repo}@{args.ref}")
    print(f"[mmlb-install] target dir: {target_dir}")

    samples_url = raw_github_url(args.repo, "data/samples.json", args.ref)
    samples_path = data_dir / "samples.json"
    download_file(samples_url, samples_path, timeout=args.timeout, retries=args.retries, force=args.force)
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    docs_api_url = github_api_url(args.repo, "data/documents", args.ref)
    document_items = fetch_json(docs_api_url, timeout=args.timeout, retries=args.retries)
    if not isinstance(document_items, list):
        raise RuntimeError(f"Unexpected GitHub API response for documents: {document_items!r}")

    pdf_items = [
        item
        for item in document_items
        if isinstance(item, dict) and item.get("type") == "file" and str(item.get("name", "")).lower().endswith(".pdf")
    ]
    if args.limit_examples is not None:
        needed_doc_ids: set[str] = set()
        for row in samples[: args.limit_examples]:
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            needed_doc_ids.add(doc_id)
            if not doc_id.lower().endswith(".pdf"):
                needed_doc_ids.add(f"{doc_id}.pdf")
        pdf_items = [item for item in pdf_items if str(item.get("name")) in needed_doc_ids]
    if args.limit_docs is not None:
        pdf_items = pdf_items[: args.limit_docs]

    print(f"[mmlb-install] pdf files to check/download: {len(pdf_items)}")
    if not args.manifest_only:
        for index, item in enumerate(pdf_items, start=1):
            download_url = item.get("download_url") or raw_github_url(args.repo, str(item["path"]), args.ref)
            out_path = documents_dir / str(item["name"])
            print(f"[mmlb-install] [{index}/{len(pdf_items)}] {item['name']}")
            download_file(download_url, out_path, timeout=args.timeout, retries=args.retries, force=args.force)

    installed_pdfs = sorted(documents_dir.glob("*.pdf"))
    summary = {
        "repo": args.repo,
        "ref": args.ref,
        "target_dir": str(target_dir),
        "samples_json": str(data_dir / "samples.json"),
        "documents_dir": str(documents_dir),
        "expected_pdf_count": len(pdf_items),
        "installed_pdf_count": len(installed_pdfs),
        "limit_examples": args.limit_examples,
        "limit_docs": args.limit_docs,
        "manifest_only": args.manifest_only,
    }
    summary_path = target_dir / "install_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[mmlb-install] complete")
    print(json.dumps(summary, indent=2))
    print(f"[mmlb-install] Use this for eval: DATA_DIR={target_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[mmlb-install] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
