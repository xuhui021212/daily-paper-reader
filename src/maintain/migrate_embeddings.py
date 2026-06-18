#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import requests

SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from maintain.sync import (  # noqa: E402
    _base_rest,
    _headers,
    _norm,
    build_embedding_text,
    iter_embedded_row_chunks,
    log,
    upsert_papers,
)


DEFAULT_TABLES = (
    "arxiv_papers",
    "biorxiv_papers",
    "medrxiv_papers",
    "chemrxiv_papers",
    "neurips_openreview_papers",
    "iclr_openreview_papers",
    "icml_openreview_papers",
    "acl_papers",
    "emnlp_papers",
    "aaai_papers",
    "papers",
)


class MissingTableError(RuntimeError):
    pass


def parse_tables(raw: str) -> List[str]:
    text = _norm(raw)
    if not text or text.lower() == "all":
        return list(DEFAULT_TABLES)
    tables = []
    seen = set()
    for part in text.replace("\n", ",").split(","):
        table = _norm(part)
        if table and table not in seen:
            seen.add(table)
            tables.append(table)
    return tables


def fetch_page(
    *,
    url: str,
    service_key: str,
    table: str,
    schema: str,
    limit: int,
    offset: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    endpoint = (
        f"{_base_rest(url)}/{table}"
        "?select=id,title,abstract,embedding_model,embedding_dim"
        "&order=id.asc"
        f"&limit={int(limit)}"
        f"&offset={int(offset)}"
    )
    resp = requests.get(
        endpoint,
        headers=_headers(service_key, schema=schema),
        timeout=max(int(timeout or 30), 1),
    )
    if resp.status_code in {404, 406} or (
        resp.status_code == 400 and "does not exist" in resp.text.lower()
    ):
        raise MissingTableError(f"table not found: {schema}.{table}")
    if resp.status_code >= 300:
        raise RuntimeError(f"fetch {table} failed: HTTP {resp.status_code} {resp.text[:300]}")
    data = resp.json() or []
    if not isinstance(data, list):
        raise RuntimeError(f"fetch {table} returned non-list JSON")
    return [row for row in data if isinstance(row, dict)]


def needs_migration(row: Dict[str, Any], *, model_name: str, expected_dim: int, force: bool) -> bool:
    if force:
        return True
    current_model = _norm(row.get("embedding_model"))
    try:
        current_dim = int(row.get("embedding_dim") or 0)
    except Exception:
        current_dim = 0
    return current_model != model_name or current_dim != expected_dim


def build_update_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "embedding": row["embedding"],
                "embedding_model": row["embedding_model"],
                "embedding_dim": row["embedding_dim"],
                "embedding_updated_at": row["embedding_updated_at"],
            }
        )
    return out


def migrate_table(
    *,
    url: str,
    service_key: str,
    table: str,
    schema: str,
    model_name: str,
    expected_dim: int,
    page_size: int,
    encode_batch_size: int,
    stream_chunk_size: int,
    upsert_batch_size: int,
    request_timeout: int,
    upsert_timeout: int,
    limit: int,
    force: bool,
    dry_run: bool,
) -> Dict[str, int]:
    scanned = 0
    selected = 0
    migrated = 0
    skipped_empty = 0
    offset = 0

    while True:
        rows = fetch_page(
            url=url,
            service_key=service_key,
            table=table,
            schema=schema,
            limit=page_size,
            offset=offset,
            timeout=request_timeout,
        )
        if not rows:
            break
        scanned += len(rows)
        offset += len(rows)

        candidates: List[Dict[str, Any]] = []
        for row in rows:
            row_id = _norm(row.get("id"))
            if not row_id:
                continue
            if not build_embedding_text(row):
                skipped_empty += 1
                continue
            if not needs_migration(row, model_name=model_name, expected_dim=expected_dim, force=force):
                continue
            candidates.append(row)
            if limit > 0 and selected + len(candidates) >= limit:
                candidates = candidates[: max(limit - selected, 0)]
                break

        if candidates:
            selected += len(candidates)
            log(
                f"[EmbeddingMigration] table={table} offset={offset - len(rows)} "
                f"selected={len(candidates)} total_selected={selected}"
            )
            if not dry_run:
                for embedded_rows, dim in iter_embedded_row_chunks(
                    candidates,
                    model_name=model_name,
                    devices=["cpu"],
                    encode_batch_size=encode_batch_size,
                    stream_chunk_size=stream_chunk_size,
                    max_length=0,
                    allow_remote=True,
                ):
                    if expected_dim > 0 and int(dim) != expected_dim:
                        raise RuntimeError(
                            f"embedding dim mismatch: table={table} expected={expected_dim} actual={dim}"
                        )
                    update_rows = build_update_rows(embedded_rows)
                    upsert_papers(
                        url=url,
                        service_key=service_key,
                        table=table,
                        rows=update_rows,
                        schema=schema,
                        batch_size=upsert_batch_size,
                        timeout=upsert_timeout,
                        retries=3,
                        retry_wait=2.0,
                    )
                    migrated += len(update_rows)

        if limit > 0 and selected >= limit:
            break
        if len(rows) < page_size:
            break

    return {
        "scanned": scanned,
        "selected": selected,
        "migrated": migrated,
        "skipped_empty": skipped_empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed Supabase paper tables with a new embedding model.")
    parser.add_argument("--url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--service-key", default=os.getenv("SUPABASE_SERVICE_KEY", ""))
    parser.add_argument("--schema", default=os.getenv("SUPABASE_SCHEMA", "public"))
    parser.add_argument("--tables", default=os.getenv("MIGRATE_EMBED_TABLES", "all"))
    parser.add_argument("--model", default=os.getenv("DPR_EMBED_MODEL", "text-embedding-v2"))
    parser.add_argument("--expected-dim", type=int, default=int(os.getenv("DPR_EMBED_DIM", "1536")))
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--encode-batch-size", type=int, default=25)
    parser.add_argument("--stream-chunk-size", type=int, default=250)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--upsert-timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--allow-local-fallback", action="store_true")
    args = parser.parse_args()

    url = _norm(args.url)
    service_key = _norm(args.service_key)
    if not url:
        raise RuntimeError("missing SUPABASE_URL")
    if not service_key:
        raise RuntimeError("missing SUPABASE_SERVICE_KEY")
    if not _norm(os.getenv("DPR_EMBED_API_URL")):
        raise RuntimeError("missing DPR_EMBED_API_URL")
    if not _norm(os.getenv("DPR_EMBED_API_KEY")):
        raise RuntimeError("missing DPR_EMBED_API_KEY")
    if not args.allow_local_fallback:
        os.environ["DPR_EMBED_REQUIRE_REMOTE"] = "1"

    tables = parse_tables(args.tables)
    started = time.time()
    totals = {"scanned": 0, "selected": 0, "migrated": 0, "skipped_empty": 0}
    log(
        f"[EmbeddingMigration] start tables={tables} model={args.model} "
        f"expected_dim={args.expected_dim} dry_run={bool(args.dry_run)} force={bool(args.force)}"
    )

    for table in tables:
        try:
            stats = migrate_table(
                url=url,
                service_key=service_key,
                table=table,
                schema=_norm(args.schema) or "public",
                model_name=_norm(args.model),
                expected_dim=max(int(args.expected_dim or 0), 0),
                page_size=max(int(args.page_size or 1), 1),
                encode_batch_size=max(int(args.encode_batch_size or 1), 1),
                stream_chunk_size=max(int(args.stream_chunk_size or 1), 1),
                upsert_batch_size=max(int(args.upsert_batch_size or 1), 1),
                request_timeout=max(int(args.request_timeout or 1), 1),
                upsert_timeout=max(int(args.upsert_timeout or 1), 1),
                limit=max(int(args.limit or 0), 0),
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
        except MissingTableError as exc:
            if args.skip_missing:
                log(f"[EmbeddingMigration] skip missing table: {exc}")
                continue
            raise
        for key, value in stats.items():
            totals[key] += int(value or 0)
        log(f"[EmbeddingMigration] table done: {table} {stats}")

    elapsed = time.time() - started
    log(f"[EmbeddingMigration] all done: {totals} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
