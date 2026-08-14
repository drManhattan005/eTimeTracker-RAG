#!/usr/bin/env python3
"""
app/infrastructure/chunking.py
──────────────────────────────
Markdown-to-chunk pipeline for the eTimeTracker RAG knowledge base.

Key guarantees:
- Hard plan-tier boundaries: Starter / Business / Enterprise chunks are NEVER
  merged with each other or with non-plan chunks.
- Exactly one atomic plan_comparison chunk is emitted (preserved or generated).
- Rich commercial and buyer-fit metadata is attached to every relevant chunk.
- Stable, deterministic chunk IDs (no UUIDs).
- A machine-readable commercial plan catalog is written to output/.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent          # .../Veloit-RAG
SOURCE_DIR = _PROJECT_ROOT / "docs"
OUTPUT_DIR = _PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
CATALOG_FILE = OUTPUT_DIR / "commercial_plan_catalog.json"

# ── Token budget ─────────────────────────────────────────────────────────────
MIN_TOKENS = 140
MAX_TOKENS = 320
TARGET_TOKENS = 220
OVERLAP_TOKENS = 40

# ── Regex helpers ────────────────────────────────────────────────────────────
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
WORD_RE = re.compile(r"\S+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# ── Subsection routing keys ──────────────────────────────────────────────────
QUESTION_KEYS = {"answerable questions"}
FIT_SIGNAL_KEYS = {"fit signals", "selection criteria"}
SUPPORT_KEYS = {
    "capabilities",
    "relevant product capabilities",
    "included capabilities",
    "commercial details",
    "coverage details",
    "capacity details",
    "support details",
}
CAVEAT_KEYS = {
    "qualification questions",
    "not listed for starter",
    "sales qualification topics",
    "upgrade triggers",
}

# ── Plan-tier detection ───────────────────────────────────────────────────────
PLAN_TIER_PATTERNS: dict[str, re.Pattern] = {
    "starter":    re.compile(r"\bstarter\b", re.IGNORECASE),
    "business":   re.compile(r"\bbusiness\b", re.IGNORECASE),
    "enterprise": re.compile(r"\benterprise\b", re.IGNORECASE),
}
COMPARISON_PATTERN = re.compile(
    r"(commercial comparison|public plan entitlements|plan entitlements|plan comparison)",
    re.IGNORECASE,
)

# ── Canonical commercial facts ────────────────────────────────────────────────
COMMERCIAL_CATALOG: dict[str, Any] = {
    "starter": {
        "price_model": "per_employee_monthly",
        "price_inr_per_employee_month": 99,
        "employee_limit": 100,
        "employee_capacity": "limited",
        "location_scope": "one_location",
        "support_tier": "email",
        "published_entitlements": [
            "attendance", "leave", "basic_reports", "email_support", "one_location",
        ],
        "not_listed_in_published_plan": [
            "geofencing", "face_approvals", "multi_branch", "multi_tenant",
            "custom_integrations", "dedicated_account_manager", "sla_guarantee",
        ],
    },
    "business": {
        "price_model": "per_employee_monthly",
        "price_inr_per_employee_month": 249,
        "employee_limit": 1000,
        "employee_capacity": "limited",
        "location_scope": "multi_branch",
        "support_tier": "priority",
        "published_entitlements": [
            "attendance", "leave", "basic_reports", "email_support",
            "shifts", "scheduling", "tour_management", "expense_management",
            "geofencing", "face_approvals", "priority_support", "multi_branch",
        ],
        "not_listed_in_published_plan": [
            "multi_tenant", "custom_integrations",
            "dedicated_account_manager", "sla_guarantee",
        ],
    },
    "enterprise": {
        "price_model": "custom",
        "price_inr_per_employee_month": None,
        "employee_limit": None,
        "employee_capacity": "unlimited",
        "location_scope": "multi_tenant",
        "support_tier": "dedicated_account_manager",
        "published_entitlements": [
            "all_business_features", "multi_tenant", "custom_integrations",
            "dedicated_account_manager", "sla_guarantee",
        ],
        "not_listed_in_published_plan": [],
    },
}

# Per-tier boolean capability flags
_TIER_FLAGS: dict[str, dict[str, Any]] = {
    "starter": {
        "plan_tier": "starter",
        "plans_covered": ["starter"],
        "price_model": "per_employee_monthly",
        "price_inr_per_employee_month": 99,
        "employee_limit": 100,
        "employee_capacity": "limited",
        "location_scope": "one_location",
        "support_tier": "email",
        "has_geofencing": False,
        "has_face_approvals": False,
        "has_multi_branch": False,
        "has_multi_tenant": False,
        "has_custom_integrations": False,
        "has_dedicated_account_manager": False,
        "has_sla_guarantee": False,
        "published_entitlements": COMMERCIAL_CATALOG["starter"]["published_entitlements"],
        "not_listed_in_published_plan": COMMERCIAL_CATALOG["starter"]["not_listed_in_published_plan"],
    },
    "business": {
        "plan_tier": "business",
        "plans_covered": ["business"],
        "price_model": "per_employee_monthly",
        "price_inr_per_employee_month": 249,
        "employee_limit": 1000,
        "employee_capacity": "limited",
        "location_scope": "multi_branch",
        "support_tier": "priority",
        "has_geofencing": True,
        "has_face_approvals": True,
        "has_multi_branch": True,
        "has_multi_tenant": False,
        "has_custom_integrations": False,
        "has_dedicated_account_manager": False,
        "has_sla_guarantee": False,
        "published_entitlements": COMMERCIAL_CATALOG["business"]["published_entitlements"],
        "not_listed_in_published_plan": COMMERCIAL_CATALOG["business"]["not_listed_in_published_plan"],
    },
    "enterprise": {
        "plan_tier": "enterprise",
        "plans_covered": ["enterprise"],
        "price_model": "custom",
        "price_inr_per_employee_month": None,
        "employee_limit": None,
        "employee_capacity": "unlimited",
        "location_scope": "multi_tenant",
        "support_tier": "dedicated_account_manager",
        "has_geofencing": True,
        "has_face_approvals": True,
        "has_multi_branch": True,
        "has_multi_tenant": True,
        "has_custom_integrations": True,
        "has_dedicated_account_manager": True,
        "has_sla_guarantee": True,
        "published_entitlements": COMMERCIAL_CATALOG["enterprise"]["published_entitlements"],
        "not_listed_in_published_plan": COMMERCIAL_CATALOG["enterprise"]["not_listed_in_published_plan"],
    },
}

# ── Buyer-fit topic map ───────────────────────────────────────────────────────
_BUYER_FIT_TOPIC_MAP: dict[str, dict[str, Any]] = {
    "field-and-distributed-workforces": {
        "buyer_fit_topics": ["field_and_distributed_workforces"],
        "workforce_types": ["field_sales", "distributed_workforce"],
        "organization_signals": ["multi_branch"],
        "requires_multi_branch": True,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["geofencing", "location_tracking", "outdoor_duty", "tour_management"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "attendance-sensitive-operations": {
        "buyer_fit_topics": ["attendance_sensitive"],
        "workforce_types": ["office", "factory", "field"],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["biometric_integration", "face_approvals", "attendance_management"],
        "recommended_plan_candidates": ["starter", "business", "enterprise"],
    },
    "shift-based-operations": {
        "buyer_fit_topics": ["shift_based"],
        "workforce_types": ["shift_workers", "factory", "rotating_shifts"],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["shifts", "scheduling", "rotating_shifts"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "mobile-first-workforces": {
        "buyer_fit_topics": ["mobile_first"],
        "workforce_types": ["mobile_workforce", "field_sales"],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["mobile_attendance", "mobile_leave", "mobile_approvals"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "multi-location-organizations": {
        "buyer_fit_topics": ["multi_location", "multi_branch"],
        "workforce_types": [],
        "organization_signals": ["multi_branch"],
        "requires_multi_branch": True,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["multi_branch", "multi_device_sync", "geofencing"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "multi-subsidiary-and-group-organizations": {
        "buyer_fit_topics": ["multi_subsidiary", "group_organization"],
        "workforce_types": [],
        "organization_signals": ["multi_tenant", "multi_subsidiary"],
        "requires_multi_branch": True,
        "requires_multi_tenant": True,
        "relevant_capabilities": ["multi_tenant", "custom_integrations", "dedicated_account_manager"],
        "recommended_plan_candidates": ["enterprise"],
    },
    "mid-size-enterprises": {
        "buyer_fit_topics": ["mid_size_enterprise"],
        "workforce_types": [],
        "organization_signals": ["multi_branch"],
        "employee_range_min": 101,
        "employee_range_max": 1000,
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["shifts", "geofencing", "tour_management", "expense_management"],
        "recommended_plan_candidates": ["business"],
    },
    "growing-organizations": {
        "buyer_fit_topics": ["growing_organization"],
        "workforce_types": [],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["bulk_import", "scalable_workforce_management"],
        "recommended_plan_candidates": ["starter", "business", "enterprise"],
    },
    "hr-teams-seeking-unified-operations": {
        "buyer_fit_topics": ["hr_unified_operations"],
        "workforce_types": ["hr_teams"],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["attendance_management", "leave_management", "approval_inbox"],
        "recommended_plan_candidates": ["starter", "business", "enterprise"],
    },
    "managers-needing-operational-visibility": {
        "buyer_fit_topics": ["operational_visibility"],
        "workforce_types": ["managers"],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["dashboards", "drill_down", "approval_inbox", "location_tracking"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "organizations-requiring-controlled-access": {
        "buyer_fit_topics": ["controlled_access", "rbac"],
        "workforce_types": [],
        "organization_signals": ["multi_tenant"],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["rbac", "audit_logs", "2fa", "permission_sets"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "spreadsheet-replacement": {
        "buyer_fit_topics": ["spreadsheet_replacement"],
        "workforce_types": [],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["bulk_import", "attendance_management", "leave_management"],
        "recommended_plan_candidates": ["starter", "business", "enterprise"],
    },
    "consolidating-multiple-tools": {
        "buyer_fit_topics": ["tool_consolidation"],
        "workforce_types": [],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": ["unified_platform", "approval_inbox", "tour_management"],
        "recommended_plan_candidates": ["business", "enterprise"],
    },
    "confirm-before-positioning": {
        "buyer_fit_topics": ["buyer_fit_boundaries"],
        "workforce_types": [],
        "organization_signals": [],
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": [],
        "recommended_plan_candidates": [],
    },
}

# ── Comparison table text (fallback if source has none) ──────────────────────
_COMPARISON_TABLE_TEXT = """\
## Commercial Comparison

### Public Plan Entitlements

| Commercial Area | Starter | Business | Enterprise |
|---|---|---|---|
| Price | 99 INR per employee/month | 249 INR per employee/month | Custom pricing |
| Employee capacity | Up to 100 employees | Up to 1,000 employees | Unlimited employees |
| Attendance and leave | Included | Included | Included |
| Basic reports | Included | Included | Included |
| Shifts and scheduling | Not listed | Included | Included |
| Tour and expense management | Not listed | Included | Included |
| Geofencing | Not listed | Included | Included |
| Face approvals | Not listed | Included | Included |
| Location or branch coverage | One location | Multi-branch | Multi-tenant |
| Support | Email support | Priority support | Dedicated account manager |
| Multi-tenant administration | Not listed | Not listed | Included |
| Custom integrations | Not listed | Not listed | Included |
| SLA guarantee | Not listed | Not listed | Included |

#### Plan Selection Boundaries

- Starter is the published option for up to 100 employees at one location.
- Business is the published option for up to 1,000 employees needing shifts, tours, expenses, geofencing, face approvals, or multi-branch support.
- Enterprise is the published option for unlimited employees, multi-tenant administration, custom integrations, dedicated account management, or an SLA guarantee.
- Geofencing is listed as a Business entitlement and is not listed among the published Starter entitlements.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Subsection:
    heading: str
    lines: list[str] = field(default_factory=list)

    def bullets(self) -> list[str]:
        return [line[2:].strip() for line in self.lines if line.strip().startswith("- ")]

    def body_text(self) -> str:
        return "\n".join(self.lines).strip()

    def render(self) -> str:
        parts = [f"#### {self.heading}"]
        body = self.body_text()
        if body:
            parts.append(body)
        return "\n\n".join(parts)


@dataclass
class Section:
    h2: str
    h3: str
    intro_lines: list[str] = field(default_factory=list)
    subsections: list[Subsection] = field(default_factory=list)

    def render(self) -> str:
        blocks = [f"### {self.h3}"]
        intro = normalize_text("\n".join(self.intro_lines))
        if intro:
            blocks.append(intro)
        blocks.extend(sub.render() for sub in self.subsections)
        return normalize_text("\n\n".join(blocks))


@dataclass
class Document:
    path: Path
    title: str
    intent_sphere: str
    h1_title: str
    intro_lines: list[str]
    sections: list[Section]


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = blank
    return "\n".join(cleaned).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def estimate_tokens(text: str) -> int:
    return max(1, round(word_count(text) * 1.3))


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    return parts or [text]


def split_large_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    result_chunks: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and estimate_tokens(candidate) > max_tokens:
            result_chunks.append(current)
            overlap: list[str] = []
            running = 0
            for prev_sentence in reversed(current):
                overlap.insert(0, prev_sentence)
                running += estimate_tokens(prev_sentence)
                if running >= overlap_tokens:
                    break
            current = overlap + [sentence]
        else:
            current.append(sentence)
    if current:
        result_chunks.append(current)
    return [" ".join(group).strip() for group in result_chunks]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_frontmatter(raw_text: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        raise ValueError("Missing YAML frontmatter.")
    meta_raw = match.group(1)
    body = raw_text[match.end():]
    meta: dict[str, str] = {}
    for line in meta_raw.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", path.stem)
    intent_sphere = meta.get("intent_sphere", "unknown")
    h1_title = title
    intro_lines: list[str] = []
    sections: list[Section] = []
    current_h2: str | None = None
    current_section: Section | None = None
    current_subsection: Subsection | None = None
    before_first_h3 = True

    for line in body.splitlines():
        if line.startswith("# "):
            h1_title = line[2:].strip()
            continue
        if line.startswith("## "):
            current_h2 = line[3:].strip()
            current_section = None
            current_subsection = None
            continue
        if line.startswith("### "):
            if current_h2 is None:
                raise ValueError(f"{path.name}: found ### before any ##")
            current_section = Section(h2=current_h2, h3=line[4:].strip())
            sections.append(current_section)
            current_subsection = None
            before_first_h3 = False
            continue
        if line.startswith("#### "):
            if current_section is None:
                raise ValueError(f"{path.name}: found #### outside a ### section")
            current_subsection = Subsection(heading=line[5:].strip())
            current_section.subsections.append(current_subsection)
            continue
        if before_first_h3:
            intro_lines.append(line)
        elif current_subsection is not None:
            current_subsection.lines.append(line)
        elif current_section is not None:
            current_section.intro_lines.append(line)

    return Document(
        path=path,
        title=title,
        intent_sphere=intent_sphere,
        h1_title=h1_title,
        intro_lines=intro_lines,
        sections=sections,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plan-tier detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_plan_tier(h2: str, h3: str) -> str | None:
    combined = f"{h2} {h3}".strip()
    if COMPARISON_PATTERN.search(combined):
        return "all"
    for tier, pat in PLAN_TIER_PATTERNS.items():
        if pat.search(combined):
            return tier
    return None


def is_comparison_section(section: Section) -> bool:
    return detect_plan_tier(section.h2, section.h3) == "all"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_from_sections(
    sections: list[Section],
) -> tuple[list[str], list[str], list[str], list[str]]:
    questions: list[str] = []
    fit_signals: list[str] = []
    supports: list[str] = []
    caveats: list[str] = []
    for section in sections:
        for sub in section.subsections:
            key = sub.heading.strip().lower()
            bullets = sub.bullets()
            if key in QUESTION_KEYS:
                questions.extend(bullets)
            elif key in FIT_SIGNAL_KEYS:
                fit_signals.extend(bullets)
            elif key in SUPPORT_KEYS:
                supports.extend(bullets)
            elif key in CAVEAT_KEYS:
                caveats.extend(bullets)
    return dedupe(questions), dedupe(fit_signals), dedupe(supports), dedupe(caveats)


def build_heading_path(sections: list[Section]) -> list[str]:
    h2_vals = list(dict.fromkeys(s.h2 for s in sections))
    h3_vals = list(dict.fromkeys(s.h3 for s in sections))
    if len(h2_vals) == 1 and len(h3_vals) == 1:
        return [h2_vals[0], h3_vals[0]]
    if len(h2_vals) == 1:
        return [h2_vals[0]] + h3_vals
    return h2_vals


def commercial_meta_for_tier(tier: str) -> dict[str, Any]:
    return dict(_TIER_FLAGS[tier])


def buyer_fit_meta_for_h3(h3: str) -> dict[str, Any]:
    slug = slugify(h3)
    base: dict[str, Any] = {
        "buyer_fit_topics": [],
        "workforce_types": [],
        "organization_signals": [],
        "employee_range_min": None,
        "employee_range_max": None,
        "requires_multi_branch": False,
        "requires_multi_tenant": False,
        "relevant_capabilities": [],
        "recommended_plan_candidates": [],
    }
    overrides = _BUYER_FIT_TOPIC_MAP.get(slug, {})
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Text building
# ─────────────────────────────────────────────────────────────────────────────

def build_chunk_text(sections: list[Section]) -> str:
    blocks: list[str] = []
    current_h2: str | None = None
    for section in sections:
        if section.h2 != current_h2:
            blocks.append(f"## {section.h2}")
            current_h2 = section.h2
        blocks.append(section.render())
    return normalize_text("\n\n".join(blocks))


# ─────────────────────────────────────────────────────────────────────────────
# Merge logic — PLAN-TIER SAFE
# ─────────────────────────────────────────────────────────────────────────────

def merge_small_sections(sections: list[Section]) -> list[list[Section]]:
    """
    Group adjacent small sections into single chunks, NEVER across:
    - different plan tiers (starter/business/enterprise/None)
    - plan vs. non-plan
    - comparison ('all') vs. anything else
    """
    groups: list[list[Section]] = []
    i = 0
    while i < len(sections):
        current_group = [sections[i]]
        current_tokens = estimate_tokens(sections[i].render())
        current_tier = detect_plan_tier(sections[i].h2, sections[i].h3)

        # Comparison sections are always alone
        if current_tier == "all":
            groups.append(current_group)
            i += 1
            continue

        while i + 1 < len(sections):
            next_sec = sections[i + 1]
            next_tier = detect_plan_tier(next_sec.h2, next_sec.h3)
            # Hard stops
            if next_tier == "all":
                break
            if next_tier != current_tier:
                break
            if next_sec.h2 != current_group[-1].h2:
                break
            if current_tokens >= MIN_TOKENS:
                break
            next_tokens = estimate_tokens(next_sec.render())
            if current_tokens + next_tokens > TARGET_TOKENS and current_tokens >= MIN_TOKENS:
                break
            current_group.append(next_sec)
            current_tokens += next_tokens
            i += 1

        groups.append(current_group)
        i += 1
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Main chunk builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks_for_document(doc: Document) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    source_stem = slugify(doc.path.stem)

    comparison_sections = [s for s in doc.sections if is_comparison_section(s)]
    non_comparison_sections = [s for s in doc.sections if not is_comparison_section(s)]

    # ── Atomic comparison chunk ───────────────────────────────────────────────
    if doc.intent_sphere == "commercial":
        if comparison_sections:
            comp_section = comparison_sections[-1]
            comp_text = build_chunk_text([comp_section])
            if "Geofencing is listed as a Business entitlement" not in comp_text:
                comp_text += (
                    "\n\nGeofencing is listed as a Business entitlement and is not listed "
                    "among the published Starter entitlements."
                )
            comp_text = normalize_text(comp_text)
            questions, fit_signals, supports, caveats = extract_metadata_from_sections([comp_section])
        else:
            comp_text = normalize_text(_COMPARISON_TABLE_TEXT)
            questions, fit_signals, supports, caveats = [], [], [], []

        comp_id = f"{source_stem}::commercial-comparison::public-plan-entitlements::1"
        heading_path = ["Commercial Comparison", "Public Plan Entitlements"]
        chunks.append(ChunkRecord(
            chunk_id=comp_id,
            text=comp_text,
            metadata={
                "chunk_id": comp_id,
                "source_file": doc.path.name,
                "intent_sphere": "commercial",
                "chunk_type": "plan_comparison",
                "heading_path": heading_path,
                "heading_path_text": " > ".join(heading_path),
                "atomic": True,
                "plan_tier": "all",
                "plans_covered": ["starter", "business", "enterprise"],
                "contains_price_comparison": True,
                "contains_employee_limit_comparison": True,
                "contains_entitlement_comparison": True,
                "contains_support_comparison": True,
                "questions": questions,
                "fit_signals": fit_signals,
                "supports": supports,
                "caveats": caveats,
            },
        ))

    # ── Non-comparison sections ───────────────────────────────────────────────
    for group in merge_small_sections(non_comparison_sections):
        h2 = group[0].h2
        h3_label = group[0].h3 if len(group) == 1 else " + ".join(s.h3 for s in group)

        tiers_in_group = {detect_plan_tier(s.h2, s.h3) for s in group}

        # Safety: cross-tier groups get split apart
        if len(tiers_in_group) > 1:
            for sec in group:
                _emit_single(doc, sec, source_stem, chunks)
            continue

        effective_tier = next(iter(tiers_in_group))
        base_text = build_chunk_text(group)
        questions, fit_signals, supports, caveats = extract_metadata_from_sections(group)
        heading_path = build_heading_path(group)
        heading_path_text = " > ".join(heading_path)

        if effective_tier in ("starter", "business", "enterprise"):
            chunk_type = "plan"
        elif doc.intent_sphere == "commercial":
            chunk_type = "commercial_boundary"
        elif doc.intent_sphere == "buyer_fit":
            chunk_type = "buyer_fit"
        else:
            chunk_type = "feature"

        meta: dict[str, Any] = {
            "chunk_id": "",
            "source_file": doc.path.name,
            "intent_sphere": doc.intent_sphere,
            "chunk_type": chunk_type,
            "heading_path": heading_path,
            "heading_path_text": heading_path_text,
            "atomic": False,
            "questions": questions,
            "fit_signals": fit_signals,
            "supports": supports,
            "caveats": caveats,
        }

        if effective_tier in ("starter", "business", "enterprise"):
            meta.update(commercial_meta_for_tier(effective_tier))

        if doc.intent_sphere == "buyer_fit":
            bf = buyer_fit_meta_for_h3(group[0].h3)
            for k, v in bf.items():
                if k not in ("chunk_id", "source_file", "heading_path", "heading_path_text"):
                    meta[k] = v

        # Plan chunks get a generous max to avoid unwanted splitting
        max_t = MAX_TOKENS * 2 if effective_tier in ("starter", "business", "enterprise") else MAX_TOKENS
        parts = split_large_text(base_text, max_t, OVERLAP_TOKENS)

        for part_index, part_text in enumerate(parts, start=1):
            chunk_id = f"{source_stem}::{slugify(h2)}::{slugify(h3_label)}::{part_index}"
            cm = dict(meta)
            cm["chunk_id"] = chunk_id
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                text=normalize_text(part_text),
                metadata=cm,
            ))

    return chunks


def _emit_single(
    doc: Document, sec: Section, source_stem: str, chunks: list[ChunkRecord]
) -> None:
    tier = detect_plan_tier(sec.h2, sec.h3)
    text = build_chunk_text([sec])
    questions, fit_signals, supports, caveats = extract_metadata_from_sections([sec])
    heading_path = [sec.h2, sec.h3]
    chunk_id = f"{source_stem}::{slugify(sec.h2)}::{slugify(sec.h3)}::1"
    chunk_type = "plan" if tier in ("starter", "business", "enterprise") else "feature"
    meta: dict[str, Any] = {
        "chunk_id": chunk_id,
        "source_file": doc.path.name,
        "intent_sphere": doc.intent_sphere,
        "chunk_type": chunk_type,
        "heading_path": heading_path,
        "heading_path_text": " > ".join(heading_path),
        "atomic": False,
        "questions": questions,
        "fit_signals": fit_signals,
        "supports": supports,
        "caveats": caveats,
    }
    if tier in ("starter", "business", "enterprise"):
        meta.update(commercial_meta_for_tier(tier))
    chunks.append(ChunkRecord(
        chunk_id=chunk_id,
        text=normalize_text(text),
        metadata=meta,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Summary / reporting
# ─────────────────────────────────────────────────────────────────────────────

_BF_REPORT_TOPICS = [
    ("field/distributed", ["field_and_distributed_workforces"]),
    ("mid-size/500-employee", ["mid_size_enterprise"]),
    ("multi-branch", ["multi_location", "multi_branch"]),
    ("multi-subsidiary", ["multi_subsidiary", "group_organization"]),
    ("large/above-1000", ["growing_organization"]),
    ("mobile-first", ["mobile_first"]),
    ("shift-based", ["shift_based"]),
]


def _buyer_fit_coverage(chunks: list[dict]) -> dict[str, str]:
    all_topics: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {})
        if meta.get("intent_sphere") == "buyer_fit":
            all_topics.update(meta.get("buyer_fit_topics", []))
    return {
        label: ("found" if any(t in all_topics for t in topic_set) else "absent")
        for label, topic_set in _BF_REPORT_TOPICS
    }


def print_chunk_summary(chunks: list[dict], catalog_path: Path, output_path: Path) -> None:
    by_source: Counter = Counter()
    by_sphere: Counter = Counter()
    by_type: Counter = Counter()
    plan_found = {"starter": False, "business": False, "enterprise": False}
    atomic_comparison = 0

    for c in chunks:
        meta = c.get("metadata", {})
        by_source[meta.get("source_file", "unknown")] += 1
        by_sphere[meta.get("intent_sphere", "unknown")] += 1
        by_type[meta.get("chunk_type", "unknown")] += 1
        tier = meta.get("plan_tier")
        if tier in plan_found:
            plan_found[tier] = True
        if (
            meta.get("chunk_type") == "plan_comparison"
            and meta.get("atomic") is True
            and meta.get("plan_tier") == "all"
        ):
            atomic_comparison += 1

    bf_coverage = _buyer_fit_coverage(chunks)

    print("\nChunking completed.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks by source file: {dict(by_source)}")
    print(f"Chunks by sphere: {dict(by_sphere)}")
    print(f"Chunks by type: {dict(by_type)}")
    print(
        f"Commercial plan chunks found: "
        f"starter={'yes' if plan_found['starter'] else 'NO'}, "
        f"business={'yes' if plan_found['business'] else 'NO'}, "
        f"enterprise={'yes' if plan_found['enterprise'] else 'NO'}"
    )
    print(f"Atomic commercial comparison chunks: {atomic_comparison}")
    print(f"Buyer Fit topics found: {bf_coverage}")
    print(f"Commercial plan catalog written: {catalog_path}")
    print(f"Chunk artifact written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markdown_files = sorted(SOURCE_DIR.glob("*.md"))
    if not markdown_files:
        raise SystemExit(f"No markdown files found in: {SOURCE_DIR}")

    all_chunks: list[dict] = []

    for path in markdown_files:
        doc = parse_document(path)
        for cr in build_chunks_for_document(doc):
            all_chunks.append({
                "id": cr.chunk_id,
                "chunk_id": cr.chunk_id,
                "text": cr.text,
                "metadata": cr.metadata,
                # Legacy flat fields for backward compat
                "intent_sphere": cr.metadata.get("intent_sphere", ""),
                "section": {
                    "h2": cr.metadata["heading_path"][0] if cr.metadata.get("heading_path") else "",
                    "h3": cr.metadata["heading_path"][1] if len(cr.metadata.get("heading_path", [])) > 1 else "",
                },
                "questions": cr.metadata.get("questions", []),
                "fit_signals": cr.metadata.get("fit_signals", []),
                "supports": cr.metadata.get("supports", []),
                "caveats": cr.metadata.get("caveats", []),
            })

    # Deduplicate IDs
    seen_ids: dict[str, int] = {}
    for chunk in all_chunks:
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            seen_ids[cid] += 1
            new_cid = f"{cid}-dup{seen_ids[cid]}"
            chunk["chunk_id"] = new_cid
            chunk["id"] = new_cid
            chunk["metadata"]["chunk_id"] = new_cid
        else:
            seen_ids[cid] = 0

    # Write JSONL
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Write commercial catalog
    with CATALOG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(COMMERCIAL_CATALOG, fh, indent=2, ensure_ascii=False)

    print_chunk_summary(all_chunks, CATALOG_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
