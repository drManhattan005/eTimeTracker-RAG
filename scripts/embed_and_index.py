#!/usr/bin/env python3
"""
scripts/embed_and_index.py
──────────────────────────
Standalone embedding and indexing script for the eTimeTracker RAG pipeline.

Reads output/chunks.jsonl, validates the chunk schema, embeds each chunk's
text, and upserts all metadata into the Qdrant vector index.

Does NOT implement query-time retrieval, BM25, hybrid search, reranking,
routing, or any answer-generation logic.

Usage:
    python scripts/embed_and_index.py
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

# ── Project root on sys.path so app.* imports resolve ───────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import settings
from app.infrastructure.embedders import FastEmbedder
from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
from qdrant_client import QdrantClient


# ── Paths ────────────────────────────────────────────────────────────────────
CHUNKS_FILE = _PROJECT_ROOT / "output" / "chunks.jsonl"
CATALOG_FILE = _PROJECT_ROOT / "output" / "commercial_plan_catalog.json"

# ── Qdrant UUID namespace (same as qdrant_store.py) ──────────────────────────
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# ── Canonical commercial facts for validation ─────────────────────────────────
_EXPECTED_STARTER = {"price_inr_per_employee_month": 99, "employee_limit": 100, "support_tier": "email", "location_scope": "one_location"}
_EXPECTED_BUSINESS = {"price_inr_per_employee_month": 249, "employee_limit": 1000, "support_tier": "priority", "location_scope": "multi_branch", "has_geofencing": True}
_EXPECTED_ENTERPRISE = {"price_model": "custom", "employee_capacity": "unlimited", "location_scope": "multi_tenant", "support_tier": "dedicated_account_manager", "has_sla_guarantee": True}

# ── Buyer-fit topics to check ─────────────────────────────────────────────────
_BF_REPORT_TOPICS = [
    ("field/distributed", ["field_and_distributed_workforces"]),
    ("mid-size/500-employee", ["mid_size_enterprise"]),
    ("multi-branch", ["multi_location", "multi_branch"]),
    ("multi-subsidiary", ["multi_subsidiary", "group_organization"]),
    ("large/above-1000", ["growing_organization"]),
    ("mobile-first", ["mobile_first"]),
    ("shift-based", ["shift_based"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Payload sanitisation for Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_for_qdrant(value: Any, field_name: str = "") -> Any:
    """
    Qdrant supports: str, int, float, bool, None, list (of above), dict (of above).
    Arrays of native JSON types are stored as-is.
    None maps to JSON null which Qdrant supports in payload.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_sanitise_for_qdrant(v, field_name) for v in value]
    if isinstance(value, dict):
        return {k: _sanitise_for_qdrant(v, k) for k, v in value.items()}
    # Fallback: stringify
    return str(value)


def build_qdrant_payload(chunk: dict) -> dict:
    """
    Build the Qdrant point payload from a chunk dict.

    All metadata fields are included. Arrays, booleans, nulls, strings, and
    numbers are preserved correctly. Nothing is silently discarded.
    """
    meta = chunk.get("metadata", {})
    payload: dict[str, Any] = {}

    # Carry forward every metadata key
    for k, v in meta.items():
        payload[k] = _sanitise_for_qdrant(v, k)

    # Ensure text is present
    payload["text"] = chunk.get("text", "")

    # Legacy flat fields (for backward compat with retrieval service)
    for legacy_key in ("intent_sphere", "questions", "fit_signals", "supports", "caveats"):
        if legacy_key not in payload and legacy_key in chunk:
            payload[legacy_key] = _sanitise_for_qdrant(chunk[legacy_key], legacy_key)

    if "section" not in payload and "section" in chunk:
        payload["section"] = _sanitise_for_qdrant(chunk["section"], "section")

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_chunks(chunks: list[dict]) -> None:
    """
    Run pre-index validation. Raises SystemExit with actionable errors on failure.
    """
    errors: list[str] = []

    # 1. IDs are present, unique, and string
    chunk_ids = [c.get("chunk_id") or c.get("id", "") for c in chunks]
    if any(not cid for cid in chunk_ids):
        errors.append("ERROR: Some chunks are missing a chunk_id.")
    if len(chunk_ids) != len(set(chunk_ids)):
        dupes = [cid for cid in chunk_ids if chunk_ids.count(cid) > 1]
        errors.append(f"ERROR: Duplicate chunk IDs detected: {list(set(dupes))[:5]}")

    # 2. All chunks are JSON-serialisable
    for i, c in enumerate(chunks):
        try:
            json.dumps(c)
        except (TypeError, ValueError) as exc:
            errors.append(f"ERROR: Chunk {i} ({c.get('chunk_id')}) is not JSON-serialisable: {exc}")

    # 3. Exactly one atomic plan_comparison chunk
    comp_chunks = [
        c for c in chunks
        if c.get("metadata", {}).get("chunk_type") == "plan_comparison"
        and c.get("metadata", {}).get("atomic") is True
        and c.get("metadata", {}).get("plan_tier") == "all"
    ]
    if len(comp_chunks) == 0:
        errors.append("ERROR: No atomic plan_comparison chunk found (expected exactly 1).")
    elif len(comp_chunks) > 1:
        errors.append(f"ERROR: Found {len(comp_chunks)} atomic plan_comparison chunks (expected exactly 1).")

    # 4. At least one plan chunk per tier, with correct canonical values
    def find_plan_chunks(tier: str) -> list[dict]:
        return [
            c for c in chunks
            if c.get("metadata", {}).get("plan_tier") == tier
            and c.get("metadata", {}).get("chunk_type") == "plan"
        ]

    for tier, expected in [
        ("starter", _EXPECTED_STARTER),
        ("business", _EXPECTED_BUSINESS),
        ("enterprise", _EXPECTED_ENTERPRISE),
    ]:
        tier_chunks = find_plan_chunks(tier)
        if not tier_chunks:
            errors.append(f"ERROR: No plan chunk found for tier '{tier}'.")
            continue
        # Validate canonical fields against first chunk of that tier
        sample_meta = tier_chunks[0].get("metadata", {})
        for field, expected_val in expected.items():
            actual = sample_meta.get(field)
            if actual != expected_val:
                errors.append(
                    f"ERROR: {tier} plan chunk metadata mismatch: "
                    f"'{field}' expected {expected_val!r}, got {actual!r}."
                )

    # 5. Commercial catalog exists and parses
    if not CATALOG_FILE.exists():
        errors.append(f"ERROR: Commercial plan catalog not found at {CATALOG_FILE}")
    else:
        try:
            with CATALOG_FILE.open(encoding="utf-8") as fh:
                catalog = json.load(fh)
            for tier in ("starter", "business", "enterprise"):
                if tier not in catalog:
                    errors.append(f"ERROR: Catalog missing tier '{tier}'.")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"ERROR: Could not parse catalog: {exc}")

    # 6. Buyer Fit topic coverage report (warnings only, not errors)
    all_bf_topics: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {})
        if meta.get("intent_sphere") == "buyer_fit":
            all_bf_topics.update(meta.get("buyer_fit_topics", []))
    print("\nBuyer Fit topic pre-index coverage:")
    for label, topic_set in _BF_REPORT_TOPICS:
        status = "found" if any(t in all_bf_topics for t in topic_set) else "ABSENT"
        print(f"  {label}: {status}")

    if errors:
        print("\n" + "\n".join(errors))
        raise SystemExit("Validation failed. Indexing aborted.")

    print("\nValidation passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Post-index summary
# ─────────────────────────────────────────────────────────────────────────────

def print_index_summary(
    chunks: list[dict],
    qdrant_path: str,
    collection_name: str,
) -> None:
    total = len(chunks)
    by_sphere: Counter = Counter()
    by_type: Counter = Counter()
    by_plan_tier: Counter = Counter()
    atomic_comparison = 0

    for c in chunks:
        meta = c.get("metadata", {})
        by_sphere[meta.get("intent_sphere", "unknown")] += 1
        by_type[meta.get("chunk_type", "unknown")] += 1
        tier = meta.get("plan_tier")
        if tier:
            by_plan_tier[tier] += 1
        if (
            meta.get("chunk_type") == "plan_comparison"
            and meta.get("atomic") is True
            and meta.get("plan_tier") == "all"
        ):
            atomic_comparison += 1

    all_bf_topics: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {})
        if meta.get("intent_sphere") == "buyer_fit":
            all_bf_topics.update(meta.get("buyer_fit_topics", []))
    bf_coverage = {
        label: ("found" if any(t in all_bf_topics for t in topic_set) else "absent")
        for label, topic_set in _BF_REPORT_TOPICS
    }

    print(f"\nTotal chunks indexed: {total}")
    print(f"Chunks by intent sphere: {dict(by_sphere)}")
    print(f"Chunks by chunk type: {dict(by_type)}")
    print(f"Commercial chunks by plan tier: {dict(by_plan_tier)}")
    print(f"Atomic comparison chunks: {atomic_comparison}")
    print(f"Buyer Fit topic coverage: {bf_coverage}")
    print(f"Chunk output location: {CHUNKS_FILE}")
    print(f"Commercial plan catalog location: {CATALOG_FILE}")
    print(f"Vector index location: {qdrant_path} (collection: {collection_name})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load chunks ──────────────────────────────────────────────────────────
    if not CHUNKS_FILE.exists():
        raise SystemExit(
            f"chunks.jsonl not found at {CHUNKS_FILE}. "
            "Run the chunking script first:\n"
            "  python -m app.infrastructure.chunking"
        )

    chunks: list[dict] = []
    with CHUNKS_FILE.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON parse error on line {lineno} of {CHUNKS_FILE}: {exc}")
            chunks.append(obj)

    if not chunks:
        raise SystemExit(f"No chunks found in {CHUNKS_FILE}.")

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # ── Validate ─────────────────────────────────────────────────────────────
    validate_chunks(chunks)

    # ── Embed ────────────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(chunks)} chunks with model: {settings.EMBED_MODEL}")
    embedder = FastEmbedder(model_name=settings.EMBED_MODEL)
    texts = [c.get("text", "") for c in chunks]
    vectors = embedder.embed(texts)
    print(f"Embedding complete. Vector size: {len(vectors[0])}")

    # ── Build Qdrant payloads ─────────────────────────────────────────────────
    print("\nBuilding Qdrant payloads...")
    qdrant_points = []
    for chunk, vector in zip(chunks, vectors):
        cid = chunk.get("chunk_id") or chunk.get("id", "")
        point_id = str(uuid.uuid5(_NAMESPACE, cid))
        payload = build_qdrant_payload(chunk)
        qdrant_points.append((point_id, vector, payload))

    # ── Upsert to Qdrant ─────────────────────────────────────────────────────
    qdrant_path = settings.QDRANT_PATH
    collection_name = settings.QDRANT_COLLECTION
    vector_size = settings.QDRANT_VECTOR_SIZE

    print(f"Connecting to Qdrant at: {qdrant_path}")
    client = QdrantClient(path=qdrant_path)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )
    vector_store.ensure_collection()

    from qdrant_client.http import models as qdrant_models
    points = [
        qdrant_models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        for point_id, vector, payload in qdrant_points
    ]

    BATCH_SIZE = 64
    total_upserted = 0
    for batch_start in range(0, len(points), BATCH_SIZE):
        batch = points[batch_start: batch_start + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(points)} chunks...", end="\r")

    print(f"\nSuccessfully upserted {total_upserted} chunks into '{collection_name}'.")

    client.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print_index_summary(chunks, qdrant_path, collection_name)


if __name__ == "__main__":
    main()
