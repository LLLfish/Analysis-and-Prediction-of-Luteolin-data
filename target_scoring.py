import math
from typing import Any, Dict, Iterable, List, Tuple


UNKNOWN_ORGANISM = "Unknown"

INVALID_TARGET_NAMES = {
    "unchecked",
    "admet",
    "non-protein target",
    "radical scavenging activity",
}

CELL_LINE_TARGETS = {
    "sh-sy5y",
    "rbl-2h3",
    "k562",
    "u-937",
    "hl-60",
    "sw1573",
    "ccrf-cem",
    "raw264.7",
}


def normalize_organism(value: Any) -> str:
    if value is None:
        return UNKNOWN_ORGANISM
    text = str(value).strip()
    return text if text else UNKNOWN_ORGANISM


def classify_target(target: Dict[str, Any]) -> str:
    name = str(target.get("target_name") or "").strip().lower()
    if not name:
        return "unknown"
    if name in INVALID_TARGET_NAMES:
        return "non_protein_or_unchecked"
    if name in CELL_LINE_TARGETS:
        return "cell_line"
    return "protein_or_organism"


def is_chartable_target(target: Dict[str, Any]) -> bool:
    organism = normalize_organism(target.get("organism"))
    return organism != UNKNOWN_ORGANISM and classify_target(target) != "non_protein_or_unchecked"


def target_label(target: Dict[str, Any]) -> str:
    return (
        str(target.get("gene_symbol") or "").strip()
        or str(target.get("target_name") or "").strip()
        or str(target.get("target_chembl_id") or "").strip()
        or "Unknown"
    )


def _unit_to_nm(value: float, unit: str) -> float:
    unit_norm = (unit or "").strip().lower().replace("μ", "u")
    if unit_norm in {"nm", "nanomolar"}:
        return value
    if unit_norm in {"um", "µm", "micromolar"}:
        return value * 1000.0
    if unit_norm in {"mm", "millimolar"}:
        return value * 1_000_000.0
    return value


def activity_to_score(activity_value: Any, activity_unit: Any = "", activity_type: Any = "") -> float:
    try:
        raw_value = float(activity_value)
    except (TypeError, ValueError):
        return 0.0

    if raw_value <= 0:
        return 0.0

    activity_type_norm = str(activity_type or "").strip().lower()
    unit_norm = str(activity_unit or "").strip().lower()

    if activity_type_norm in {"ic50", "ki", "kd", "ec50", "ac50"}:
        value_nm = max(_unit_to_nm(raw_value, unit_norm), 1e-12)
        return max(0.0, 9.0 - math.log10(value_nm))

    if "%" in unit_norm or activity_type_norm in {"inhibition", "increase", "activity", "maximal relaxation"}:
        return max(0.0, min(raw_value, 100.0) / 20.0)

    if activity_type_norm in {"mic", "id50", "ic25"}:
        return max(0.0, 6.0 - math.log10(raw_value + 1.0))

    return max(0.0, 5.0 - math.log10(raw_value + 1.0))


def score_target(target: Dict[str, Any], centrality: Dict[str, float] = None) -> float:
    centrality = centrality or {}
    score = activity_to_score(
        target.get("activity_value"),
        target.get("activity_unit"),
        target.get("activity_type"),
    )

    try:
        pchembl_value = float(target.get("pchembl_value"))
        if pchembl_value > 0:
            score = max(score, pchembl_value)
    except (TypeError, ValueError):
        pass

    organism = normalize_organism(target.get("organism"))
    if organism == "Homo sapiens":
        score += 0.8
    elif organism == UNKNOWN_ORGANISM:
        score -= 0.8

    if target.get("gene_symbol"):
        score += 0.6

    if classify_target(target) == "non_protein_or_unchecked":
        score -= 2.0
    elif classify_target(target) == "cell_line":
        score -= 0.8

    label = target_label(target)
    score += float(centrality.get(label, 0.0)) * 2.0
    return round(max(score, 0.0), 4)


def rank_targets(targets: Iterable[Dict[str, Any]], limit: int = 15,
                 centrality: Dict[str, float] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    best_by_label: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for target in targets:
        if not isinstance(target, dict):
            continue
        label = target_label(target)
        score = score_target(target, centrality)
        previous = best_by_label.get(label)
        if previous is None or score > previous[0]:
            best_by_label[label] = (score, target)

    ranked = sorted(best_by_label.items(), key=lambda item: item[1][0], reverse=True)
    return [(label, score, target) for label, (score, target) in ranked[:limit]]


def organism_distribution(targets: Iterable[Dict[str, Any]], limit: int = 10) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for target in targets:
        if not isinstance(target, dict) or not is_chartable_target(target):
            continue
        organism = normalize_organism(target.get("organism"))
        counts[organism] = counts.get(organism, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
