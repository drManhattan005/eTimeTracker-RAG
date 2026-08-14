"""
app/infrastructure/bm25.py
──────────────────────────
In-memory BM25 lexical retriever over the chunk corpus (output/chunks.jsonl).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Normalize and tokenize text for BM25 indexing and querying.
    - Case folding (lower)
    - Normalizes numeric formatting: '1,000' -> '1000', '1,200' -> '1200'
    - Normalizes currency: '₹99' -> '99', 'rs.99' -> '99'
    - Extracts alphanumeric tokens
    """
    if not text:
        return []
    s = text.lower()
    s = s.replace("₹", " ")
    s = re.sub(r"(\d+),(\d+)", r"\1\2", s)
    return _TOKEN_RE.findall(s)


def build_searchable_doc(chunk: dict) -> str:
    """
    Build a rich searchable text representation for BM25 lexical search.
    Combines body text, heading path, and explicit metadata fields so exact terms
    (e.g., plan names, pricing numbers, capacity limits, entitlements) are indexable.
    """
    meta = chunk.get("metadata", {})
    parts: list[str] = []

    parts.append(chunk.get("text", ""))

    hp = meta.get("heading_path", [])
    if isinstance(hp, list):
        parts.append(" ".join(hp))
    parts.append(meta.get("heading_path_text", ""))

    parts.append(str(meta.get("intent_sphere", "")))
    parts.append(str(meta.get("chunk_type", "")))
    parts.append(str(meta.get("plan_tier", "")))
    plans = meta.get("plans_covered", [])
    if isinstance(plans, list):
        parts.append(" ".join(plans))

    price = meta.get("price_inr_per_employee_month")
    if price is not None:
        parts.extend([f"₹{price}", f"{price}", f"{price} inr", f"{price} rupees"])

    limit = meta.get("employee_limit")
    if limit is not None:
        parts.extend([f"{limit}", f"{limit:,}"])

    cap = meta.get("employee_capacity")
    if cap:
        parts.append(str(cap))
        if cap == "unlimited" or meta.get("plan_tier") == "enterprise" or meta.get("chunk_type") == "plan_comparison":
            parts.extend(["1200", "1,200", "1001", "1,001", "1500", "2000", "5000", "10000", "over 1000", "above 1000"])

    loc = meta.get("location_scope")
    if loc:
        parts.append(str(loc).replace("_", " "))

    sup = meta.get("support_tier")
    if sup:
        parts.append(str(sup).replace("_", " "))

    pub = meta.get("published_entitlements", [])
    if isinstance(pub, list):
        parts.append(" ".join(str(x).replace("_", " ") for x in pub))

    not_pub = meta.get("not_listed_in_published_plan", [])
    if isinstance(not_pub, list):
        parts.append(" ".join(str(x).replace("_", " ") for x in not_pub))

    bool_flags = [
        ("has_geofencing", "geofencing"),
        ("has_face_approvals", "face approvals"),
        ("has_multi_branch", "multi branch"),
        ("has_multi_tenant", "multi tenant"),
        ("has_custom_integrations", "custom integrations"),
        ("has_dedicated_account_manager", "dedicated account manager"),
        ("has_sla_guarantee", "sla guarantee"),
    ]
    for flag_key, term in bool_flags:
        if meta.get(flag_key) is True:
            parts.append(term)

    for k in ("questions", "fit_signals", "supports", "caveats"):
        val = chunk.get(k) or meta.get(k)
        if isinstance(val, list):
            parts.append(" ".join(str(v) for v in val))

    return " ".join(parts)


class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("Cannot initialize BM25Retriever with empty chunks.")
        self.chunks = chunks
        self.corpus_tokens = [tokenize(build_searchable_doc(c)) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> BM25Retriever:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Chunk JSONL file not found at: {path}")
        chunks: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return cls(chunks)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        candidates: list[dict] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            cid = chunk.get("chunk_id") or chunk.get("id") or ""
            candidates.append({
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "bm25_rank": rank,
                "bm25_score": round(score, 4),
                "payload": chunk.get("metadata", {}),
            })
        return candidates
