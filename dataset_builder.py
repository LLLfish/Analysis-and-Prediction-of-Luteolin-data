"""Deterministic, source-bound clustering and strict double-cold split tools."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from data_contracts import ProteinRecord, canonical_json_sha256, validate_protein_record
from evidence_builder import load_entity_bundle


_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "split_policy.json"
_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_ACCESSION = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIRMABLE_SOURCE_KINDS = {"online_snapshot", "synthetic_test"}
_PROTEIN_IDENTITY_STATUSES = {"verified", "mismatch", "invalid", "unverified"}
_PROTEIN_SEQUENCE_STATUSES = {"verified", "missing", "invalid", "error", "unverified"}


class DatasetBuildError(ValueError):
    """Input could not be trusted for clustering or splitting."""

    def __init__(self, message: str, *, code: str = "invalid_input") -> None:
        super().__init__(message)
        self.code = code


class _BuildAbort(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NeighborEvidence:
    entity_a: str
    entity_b: str
    reason: str
    score: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClusterBuildResult:
    status: str
    entity_version: Optional[str]
    entity_bundle_hash: Optional[str]
    policy_version: Optional[str]
    policy_sha256: Optional[str]
    source_provenance: tuple[Mapping[str, Any], ...]
    entities: tuple[Mapping[str, Any], ...]
    clusters: tuple[Cluster, ...]
    neighbors: tuple[NeighborEvidence, ...]
    excluded_entities: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, int]
    error: Optional[str] = None
    test_only: bool = False
    training_ready: bool = False


@dataclass(frozen=True)
class PairSplitAssignment:
    split_group_id: str
    pair_key: str
    molecule_cluster_id: Optional[str]
    protein_cluster_id: Optional[str]
    split: Optional[str]
    labels: tuple[str, ...] = ()
    provenance_partition: str = "unlabeled"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitResult:
    strict_split_status: str
    training_ready: bool
    entity_version: Optional[str]
    entity_bundle_hash: Optional[str]
    policy_sha256: Optional[str]
    label_policy_sha256: Optional[str]
    source_provenance: tuple[Mapping[str, Any], ...]
    molecule_clusters: tuple[Cluster, ...]
    protein_clusters: tuple[Cluster, ...]
    molecule_cluster_assignment: Mapping[str, str]
    protein_cluster_assignment: Mapping[str, str]
    pair_assignments: tuple[PairSplitAssignment, ...]
    crosspair_quarantine: tuple[PairSplitAssignment, ...]
    unassigned_pairs: tuple[PairSplitAssignment, ...]
    test_only_pair_assignments: tuple[PairSplitAssignment, ...]
    class_support: Mapping[str, Mapping[str, int]]
    split_sha256: Optional[str]
    error: Optional[str] = None
    test_only: bool = False


@dataclass(frozen=True)
class _Molecule:
    molecule_id: str
    canonical_smiles: str
    structure_sha256: str
    source_kind: str
    source_provenance: tuple[Mapping[str, Any], ...]
    mol: Any
    scaffold: str
    fingerprint: Any


@dataclass(frozen=True)
class _Protein:
    entity_id: str
    accession: str
    taxonomy_id: int
    full_sequence: str
    sequence_length: int
    sequence_sha256: str
    source_kind: str
    source_provenance: tuple[Mapping[str, Any], ...]


def _reject_json_constant(value: str) -> None:
    raise DatasetBuildError(f"non-finite JSON constant is not allowed: {value}", code="invalid_policy")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetBuildError(f"{label} must be a finite number", code="invalid_policy")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetBuildError(f"{label} must be a finite number", code="invalid_policy")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DatasetBuildError(f"{label} must be a nonnegative integer", code="invalid_policy")
    return value


def _load_policy(policy_path: Any = None) -> tuple[dict[str, Any], str]:
    path = _POLICY_PATH if policy_path is None else Path(policy_path)
    if not isinstance(path, Path):
        raise DatasetBuildError("policy_path must identify a JSON file", code="invalid_policy")
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            policy = json.load(stream, parse_constant=_reject_json_constant)
    except DatasetBuildError:
        raise
    except Exception as exc:
        raise DatasetBuildError(f"split policy could not be loaded: {exc}", code="invalid_policy") from exc
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version", "policy_version", "entity_bundle_schema_version", "molecule",
        "protein", "connected_components", "split", "budgets",
    }:
        raise DatasetBuildError("split policy has unknown or missing top-level fields", code="invalid_policy")
    if (type(policy.get("schema_version")) is not int or policy["schema_version"] != 1
            or type(policy.get("entity_bundle_schema_version")) is not int
            or policy["entity_bundle_schema_version"] != 2
            or policy.get("policy_version") != "double-cold-neighbor-split-v1"):
        raise DatasetBuildError("unsupported split policy or entity bundle schema version", code="invalid_policy")

    molecule = policy.get("molecule")
    fingerprint = molecule.get("fingerprint") if isinstance(molecule, dict) else None
    scaffold = molecule.get("scaffold") if isinstance(molecule, dict) else None
    if (not isinstance(molecule, dict) or set(molecule) != {"identity_fields", "scaffold", "fingerprint"}
            or molecule.get("identity_fields") != ["molecule_chembl_id"]
            or not isinstance(scaffold, dict)
            or scaffold.get("method") != "RDKit Murcko scaffold"
            or scaffold.get("edge_when") != "both nonempty canonical scaffold strings are equal"
            or scaffold.get("empty_scaffold_links") is not False
            or not isinstance(fingerprint, dict)
            or fingerprint.get("method") != "Morgan"
            or type(fingerprint.get("radius")) is not int or fingerprint["radius"] != 2
            or type(fingerprint.get("bits")) is not int or fingerprint["bits"] != 2048
            or fingerprint.get("similarity") != "Tanimoto"
            or fingerprint.get("operator") != ">=" or fingerprint.get("inclusive") is not True
            or _finite_number(fingerprint.get("threshold"), "molecule Tanimoto threshold") != 0.8):
        raise DatasetBuildError("molecule clustering definition differs from the audited v1 contract", code="invalid_policy")

    protein = policy.get("protein")
    alignment = protein.get("alignment") if isinstance(protein, dict) else None
    identity = protein.get("identity") if isinstance(protein, dict) else None
    coverage = protein.get("coverage") if isinstance(protein, dict) else None
    if (not isinstance(protein, dict) or set(protein) != {"identity_fields", "alignment", "identity", "coverage"}
            or protein.get("identity_fields") != ["taxonomy_id", "accession"]
            or not isinstance(alignment, dict)
            or alignment.get("api") != "Bio.Align.PairwiseAligner"
            or alignment.get("mode") != "global"
            or alignment.get("substitution_matrix") != "BLOSUM62"
            or _finite_number(alignment.get("open_gap_score"), "open_gap_score") != -10.0
            or _finite_number(alignment.get("extend_gap_score"), "extend_gap_score") != -0.5
            or _finite_number(alignment.get("epsilon"), "alignment epsilon") != 0.0
            or alignment.get("tie_policy") !=
            "first optimal alignment in PairwiseAligner iteration order after stable sequence identity ordering"
            or alignment.get("aligned_non_gap_columns") !=
            "columns where both complete-sequence indices advance together"
            or alignment.get("gap_columns_in_identity") is not False
            or not isinstance(identity, dict)
            or identity.get("formula") != "exact_residue_matches / aligned_non_gap_columns"
            or identity.get("operator") != ">=" or identity.get("inclusive") is not True
            or _finite_number(identity.get("threshold"), "protein identity threshold") != 0.3
            or not isinstance(coverage, dict)
            or coverage.get("formula") != "aligned_non_gap_columns / each full_sequence_length"
            or coverage.get("operator") != ">=" or coverage.get("inclusive") is not True
            or _finite_number(coverage.get("threshold"), "protein coverage threshold") != 0.8):
        raise DatasetBuildError("protein alignment definition differs from the audited v1 contract", code="invalid_policy")

    split = policy.get("split")
    ratios = split.get("ratios") if isinstance(split, dict) else None
    if (not isinstance(split, dict) or set(split) != {
            "names", "ratios", "seed", "cluster_order", "allocation", "rounding_tie_order",
        }
            or split.get("names") != ["train", "val", "test"]
            or not isinstance(ratios, dict) or set(ratios) != {"train", "val", "test"}
            or any(_finite_number(ratios.get(name), f"{name} ratio") != expected
                   for name, expected in (("train", 0.8), ("val", 0.1), ("test", 0.1)))
            or type(split.get("seed")) is not int or split["seed"] != 42
            or split.get("cluster_order") != "sort stable cluster IDs, then random.Random(seed).shuffle"
            or split.get("allocation") != "largest_remainder"
            or split.get("rounding_tie_order") != ["train", "val", "test"]):
        raise DatasetBuildError("split allocation differs from the audited v1 contract", code="invalid_policy")

    budgets = policy.get("budgets")
    if (not isinstance(budgets, dict) or set(budgets) != {
            "molecule_pairs", "protein_pairs", "alignment_cells", "alignment_cells_measure",
        }
            or budgets.get("alignment_cells_measure") !=
            "sum of len(sequence_a) * len(sequence_b) for every alignment attempted in one build"):
        raise DatasetBuildError("split policy budgets are malformed", code="invalid_policy")
    for name in ("molecule_pairs", "protein_pairs", "alignment_cells"):
        _nonnegative_int(budgets.get(name), f"budget {name}")
    if policy.get("connected_components") != "undirected transitive connected components":
        raise DatasetBuildError("connected-component policy is unsupported", code="invalid_policy")
    return policy, canonical_json_sha256(policy)


def _require_rdkit() -> tuple[Any, Any, Any, Any]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:
        raise _BuildAbort("dependency_missing", f"RDKit is required for molecular identity: {type(exc).__name__}") from exc
    return Chem, DataStructs, rdFingerprintGenerator, MurckoScaffold


def _require_bio() -> tuple[Any, Any, Any]:
    try:
        import Bio
        from Bio import Align
        from Bio.Align import substitution_matrices
        matrix = substitution_matrices.load("BLOSUM62")
    except Exception as exc:
        raise _BuildAbort("dependency_missing", f"Bio.Align and BLOSUM62 are required for protein identity: {type(exc).__name__}") from exc
    return Bio, Align, matrix


def _load_verified_bundle(bundle_path: Any) -> tuple[dict[str, Any], Path]:
    if not isinstance(bundle_path, (str, Path)):
        raise DatasetBuildError("entity_bundle_path must be a filesystem path", code="bundle_rejected")
    path = Path(bundle_path).resolve()
    try:
        bundle = load_entity_bundle(path, verify_sources=True)
    except Exception as exc:
        message = str(exc)
        if "rdkit_unavailable" in message or "No module named 'rdkit'" in message:
            raise _BuildAbort("dependency_missing", "RDKit is required by the source verifier") from exc
        raise DatasetBuildError(f"entity bundle failed source verification: {message}", code="bundle_rejected") from exc
    if not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 2:
        raise DatasetBuildError("unsupported entity bundle schema version", code="bundle_rejected")
    if not isinstance(bundle.get("entity_version"), str) or not isinstance(bundle.get("content_sha256"), str):
        raise DatasetBuildError("entity bundle version/hash is missing", code="bundle_rejected")
    expected_version = f"entities-v2-{bundle['content_sha256'][:16]}"
    if bundle["entity_version"] != expected_version or _SHA256.fullmatch(bundle["content_sha256"]) is None:
        raise DatasetBuildError("entity bundle version/hash does not agree", code="bundle_rejected")
    for collection in ("molecules", "proteins", "activities", "source_provenance"):
        if not isinstance(bundle.get(collection), list):
            raise DatasetBuildError(f"entity bundle {collection} must be a list", code="bundle_rejected")
    return bundle, path


def _bundle_sources(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources = []
    for source in bundle["source_provenance"]:
        if (not isinstance(source, Mapping)
                or not isinstance(source.get("source_id"), str) or not source["source_id"]
                or not isinstance(source.get("source_path"), str)
                or Path(source["source_path"]).is_absolute()
                or ".." in Path(source["source_path"]).parts
                or not isinstance(source.get("raw_sha256"), str)
                or _SHA256.fullmatch(source["raw_sha256"]) is None):
            raise DatasetBuildError("entity bundle has incomplete source provenance", code="bundle_rejected")
        sources.append(deepcopy(dict(source)))
    return tuple(sorted(sources, key=lambda row: (row["source_id"], row["source_path"], row["raw_sha256"])))


def _bundle_is_test_only(bundle: Mapping[str, Any]) -> bool:
    if bundle.get("source_kind") == "synthetic_test":
        return True
    for collection in ("molecules", "proteins", "activities"):
        for row in bundle.get(collection, []):
            if isinstance(row, Mapping) and row.get("source_kind") == "synthetic_test":
                return True
    return False


def _molecule_entities(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in bundle["molecules"]:
        if not isinstance(row, Mapping):
            raise DatasetBuildError("entity bundle molecule must be an object", code="bundle_rejected")
        molecule_id = row.get("molecule_chembl_id")
        label = molecule_id if isinstance(molecule_id, str) else "<missing>"
        if not isinstance(molecule_id, str) or _CHEMBL_ID.fullmatch(molecule_id) is None:
            excluded.append({"entity_id": label, "reason": "invalid_molecule_identity"})
            continue
        if molecule_id in seen:
            raise DatasetBuildError(f"duplicate molecule identity in entity bundle: {molecule_id}", code="bundle_rejected")
        seen.add(molecule_id)
        if row.get("structure_status") != "verified":
            excluded.append({"entity_id": molecule_id, "reason": f"structure_{row.get('structure_status', 'missing')}"})
            continue
        if row.get("source_kind") not in _CONFIRMABLE_SOURCE_KINDS:
            excluded.append({"entity_id": molecule_id, "reason": "molecule_source_kind_unverified"})
            continue
        if not isinstance(row.get("canonical_smiles"), str) or not row["canonical_smiles"]:
            excluded.append({"entity_id": molecule_id, "reason": "canonical_structure_missing"})
            continue
        structure_hash = row.get("structure_sha256")
        expected_hash = hashlib.sha256(row["canonical_smiles"].encode("utf-8")).hexdigest()
        if structure_hash != expected_hash:
            raise DatasetBuildError(f"molecule {molecule_id} structure hash does not match canonical structure", code="bundle_rejected")
        if not isinstance(row.get("source_provenance"), list) or not row["source_provenance"]:
            excluded.append({"entity_id": molecule_id, "reason": "molecule_source_provenance_missing"})
            continue
        result.append({
            "entity_id": molecule_id,
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": row["canonical_smiles"],
            "structure_sha256": structure_hash,
            "source_kind": row["source_kind"],
            "source_provenance": tuple(deepcopy(row["source_provenance"])),
        })
    result.sort(key=lambda row: row["entity_id"])
    excluded.sort(key=lambda row: (row["entity_id"], row["reason"]))
    return result, excluded


def _protein_entities(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in bundle["proteins"]:
        if not isinstance(row, Mapping):
            raise DatasetBuildError("entity bundle protein must be an object", code="bundle_rejected")
        accession = row.get("accession")
        taxonomy_id = row.get("taxonomy_id")
        label = f"{taxonomy_id}|{accession}"
        if (not isinstance(accession, str) or _ACCESSION.fullmatch(accession) is None
                or type(taxonomy_id) is not int or taxonomy_id <= 0):
            excluded.append({"entity_id": label, "reason": "invalid_protein_identity"})
            continue
        identity = (taxonomy_id, accession)
        if identity in seen:
            raise DatasetBuildError(f"duplicate protein identity in entity bundle: {label}", code="bundle_rejected")
        seen.add(identity)
        try:
            record = ProteinRecord(**dict(row))
            validate_protein_record(record)
        except (TypeError, ValueError) as exc:
            raise DatasetBuildError(f"invalid shared ProteinRecord {label}: {exc}", code="bundle_rejected") from exc
        if record.identity_status != "verified" or record.sequence_status != "verified":
            excluded.append({
                "entity_id": label,
                "reason": record.sequence_status_reason or "protein_identity_or_sequence_unverified",
            })
            continue
        if record.source_kind not in _CONFIRMABLE_SOURCE_KINDS:
            excluded.append({"entity_id": label, "reason": "protein_source_kind_unverified"})
            continue
        if not record.source_provenance:
            excluded.append({"entity_id": label, "reason": "protein_source_provenance_missing"})
            continue
        sequence = record.full_sequence
        if (not isinstance(sequence, str) or not sequence
                or type(record.sequence_length) is not int or record.sequence_length != len(sequence)):
            raise DatasetBuildError(f"protein {label} has an invalid complete sequence", code="bundle_rejected")
        sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if record.sequence_sha256 != sequence_hash:
            raise DatasetBuildError(f"protein {label} sequence hash differs from full sequence", code="bundle_rejected")
        result.append({
            "entity_id": f"{taxonomy_id}|{accession}",
            "accession": accession,
            "taxonomy_id": taxonomy_id,
            "full_sequence": sequence,
            "sequence_length": len(sequence),
            "sequence_sha256": sequence_hash,
            "source_kind": record.source_kind,
            "source_provenance": tuple(deepcopy(record.source_provenance)),
        })
    result.sort(key=lambda row: (row["taxonomy_id"], row["accession"]))
    excluded.sort(key=lambda row: (row["entity_id"], row["reason"]))
    return result, excluded


def _make_molecule(Chem: Any, rdFingerprintGenerator: Any, MurckoScaffold: Any,
                   entity: Mapping[str, Any], policy: Mapping[str, Any]) -> _Molecule:
    smiles = entity["canonical_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DatasetBuildError(f"molecule {entity['entity_id']} canonical structure cannot be parsed", code="bundle_rejected")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise DatasetBuildError(f"molecule {entity['entity_id']} is not a single fragment", code="bundle_rejected")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if canonical != smiles:
        raise DatasetBuildError(f"molecule {entity['entity_id']} structure is not canonical", code="bundle_rejected")
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True) if scaffold_mol.GetNumAtoms() else ""
    fingerprint_policy = policy["molecule"]["fingerprint"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprint_policy["radius"], fpSize=fingerprint_policy["bits"],
    )
    fingerprint = generator.GetFingerprint(mol)
    return _Molecule(
        molecule_id=entity["entity_id"],
        canonical_smiles=smiles,
        structure_sha256=entity["structure_sha256"],
        source_kind=entity["source_kind"],
        source_provenance=entity["source_provenance"],
        mol=mol,
        scaffold=scaffold,
        fingerprint=fingerprint,
    )


def _molecule_neighbor(a: _Molecule, b: _Molecule, DataStructs: Any,
                       policy: Mapping[str, Any]) -> tuple[NeighborEvidence, ...]:
    entity_a, entity_b = sorted((a.molecule_id, b.molecule_id))
    if a.molecule_id == b.molecule_id:
        return (NeighborEvidence(entity_a, entity_b, "identity", 1.0, {"identity_field": "molecule_chembl_id"}),)
    records: list[NeighborEvidence] = []
    if a.scaffold and b.scaffold and a.scaffold == b.scaffold:
        records.append(NeighborEvidence(
            entity_a, entity_b, "murcko_scaffold", 1.0,
            {"scaffold": a.scaffold, "empty_scaffold_links": False},
        ))
    similarity = float(DataStructs.TanimotoSimilarity(a.fingerprint, b.fingerprint))
    if not math.isfinite(similarity):
        raise DatasetBuildError("RDKit returned a non-finite Tanimoto score", code="invalid_score")
    fingerprint = policy["molecule"]["fingerprint"]
    if similarity >= float(fingerprint["threshold"]):
        records.append(NeighborEvidence(
            entity_a, entity_b, "morgan_tanimoto", similarity,
            {"radius": fingerprint["radius"], "bits": fingerprint["bits"],
             "threshold": fingerprint["threshold"], "inclusive": True},
        ))
    return tuple(sorted(records, key=lambda row: (row.entity_a, row.entity_b, row.reason, row.score)))


def molecule_neighbor(
    molecule_a: Mapping[str, Any],
    molecule_b: Mapping[str, Any],
    *,
    policy_path: Any = None,
) -> tuple[NeighborEvidence, ...]:
    if not isinstance(molecule_a, Mapping) or not isinstance(molecule_b, Mapping):
        raise DatasetBuildError("molecule_neighbor requires two molecule records")
    entity_a = molecule_a.get("molecule_chembl_id")
    entity_b = molecule_b.get("molecule_chembl_id")
    if (not isinstance(entity_a, str) or _CHEMBL_ID.fullmatch(entity_a) is None
            or not isinstance(entity_b, str) or _CHEMBL_ID.fullmatch(entity_b) is None):
        raise DatasetBuildError("molecule neighbor records require canonical ChEMBL identities")
    policy, _ = _load_policy(policy_path)
    if policy["budgets"]["molecule_pairs"] < 1:
        raise DatasetBuildError("molecule pair budget exceeded", code="size_limit")
    try:
        Chem, DataStructs, rdFingerprintGenerator, MurckoScaffold = _require_rdkit()
    except _BuildAbort as exc:
        raise DatasetBuildError(str(exc), code=exc.code) from exc
    def prepare(row: Mapping[str, Any]) -> _Molecule:
        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            raise DatasetBuildError("molecule neighbor records require canonical_smiles")
        structure_hash = row.get("structure_sha256")
        if structure_hash is None:
            structure_hash = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
        entity = {"entity_id": row["molecule_chembl_id"], "canonical_smiles": smiles,
                  "structure_sha256": structure_hash, "source_kind": row.get("source_kind", "synthetic_test"),
                  "source_provenance": tuple(row.get("source_provenance", ())) }
        return _make_molecule(Chem, rdFingerprintGenerator, MurckoScaffold, entity, policy)
    a = prepare(molecule_a)
    b = prepare(molecule_b)
    return _molecule_neighbor(a, b, DataStructs, policy)


def _protein_record(value: Mapping[str, Any]) -> ProteinRecord:
    if not isinstance(value, Mapping):
        raise DatasetBuildError("protein neighbor requires shared ProteinRecord values")
    try:
        record = ProteinRecord(**dict(value))
        validate_protein_record(record)
    except (TypeError, ValueError) as exc:
        raise DatasetBuildError(f"invalid shared ProteinRecord: {exc}", code="invalid_entity") from exc
    if record.identity_status != "verified" or record.sequence_status != "verified":
        raise DatasetBuildError("protein neighbor requires source-verified full sequences", code="invalid_entity")
    return record


def _alignment_context(policy: Mapping[str, Any]) -> tuple[Any, Any, Any, str]:
    Bio, Align, matrix = _require_bio()
    settings = policy["protein"]["alignment"]
    try:
        aligner = Align.PairwiseAligner(
            mode=settings["mode"],
            substitution_matrix=matrix,
            open_gap_score=float(settings["open_gap_score"]),
            extend_gap_score=float(settings["extend_gap_score"]),
            epsilon=float(settings["epsilon"]),
        )
    except Exception as exc:
        raise _BuildAbort("dependency_missing", f"Bio.Align PairwiseAligner is unavailable: {type(exc).__name__}") from exc
    return Bio, aligner, matrix, str(getattr(aligner, "algorithm", "unknown"))


def _aligned_metrics(sequence_a: str, sequence_b: str, alignment: Any) -> tuple[int, int]:
    coordinates = alignment.coordinates
    aligned_non_gap_columns = 0
    matches = 0
    for column in range(coordinates.shape[1] - 1):
        start_a, end_a = int(coordinates[0, column]), int(coordinates[0, column + 1])
        start_b, end_b = int(coordinates[1, column]), int(coordinates[1, column + 1])
        length_a, length_b = end_a - start_a, end_b - start_b
        if length_a > 0 and length_a == length_b:
            aligned_non_gap_columns += length_a
            for offset in range(length_a):
                if sequence_a[start_a + offset] == sequence_b[start_b + offset]:
                    matches += 1
    return matches, aligned_non_gap_columns


def _protein_neighbor_records(record_a: ProteinRecord, record_b: ProteinRecord, *,
                              policy: Mapping[str, Any], aligner: Any,
                              alignment_algorithm: str, biopython_version: str,
                              cells_remaining: int) -> tuple[tuple[NeighborEvidence, ...], int]:
    key_a = (record_a.taxonomy_id, record_a.accession)
    key_b = (record_b.taxonomy_id, record_b.accession)
    if key_a == key_b:
        entity_a = f"{record_a.taxonomy_id}|{record_a.accession}"
        return ((NeighborEvidence(entity_a, entity_a, "identity", 1.0,
                                  {"identity_fields": ["taxonomy_id", "accession"]}),), 0)
    if key_b < key_a:
        record_a, record_b = record_b, record_a
        key_a, key_b = key_b, key_a
    entity_a = f"{key_a[0]}|{key_a[1]}"
    entity_b = f"{key_b[0]}|{key_b[1]}"
    sequence_a = record_a.full_sequence
    sequence_b = record_b.full_sequence
    cells = len(sequence_a) * len(sequence_b)
    budget = policy["budgets"]["alignment_cells"]
    if cells > cells_remaining or cells > budget:
        raise _BuildAbort("size_limit", f"alignment cell budget exceeded for {entity_a} and {entity_b}: {cells}")
    try:
        alignments = aligner.align(sequence_a, sequence_b)
        alignment = alignments[0]
    except Exception as exc:
        raise _BuildAbort("dependency_missing", f"Bio.Align could not compute a global alignment: {type(exc).__name__}") from exc
    matches, aligned_columns = _aligned_metrics(sequence_a, sequence_b, alignment)
    identity = matches / aligned_columns if aligned_columns else 0.0
    coverage_a = aligned_columns / len(sequence_a) if sequence_a else 0.0
    coverage_b = aligned_columns / len(sequence_b) if sequence_b else 0.0
    identity_policy = policy["protein"]["identity"]
    coverage_policy = policy["protein"]["coverage"]
    records: tuple[NeighborEvidence, ...] = ()
    if (identity >= float(identity_policy["threshold"])
            and coverage_a >= float(coverage_policy["threshold"])
            and coverage_b >= float(coverage_policy["threshold"])):
        details = {
            "matches": matches,
            "aligned_non_gap_columns": aligned_columns,
            "identity": identity,
            "coverage_a": coverage_a,
            "coverage_b": coverage_b,
            "sequence_length_a": len(sequence_a),
            "sequence_length_b": len(sequence_b),
            "alignment_score": float(alignment.score),
            "alignment_algorithm": alignment_algorithm,
            "biopython_version": biopython_version,
            "substitution_matrix": policy["protein"]["alignment"]["substitution_matrix"],
            "open_gap_score": policy["protein"]["alignment"]["open_gap_score"],
            "extend_gap_score": policy["protein"]["alignment"]["extend_gap_score"],
            "epsilon": policy["protein"]["alignment"]["epsilon"],
            "tie_policy": policy["protein"]["alignment"]["tie_policy"],
            "identity_threshold_inclusive": True,
            "coverage_threshold_inclusive": True,
        }
        records = (NeighborEvidence(entity_a, entity_b, "sequence_similarity", identity, details),)
    return records, cells


def protein_neighbor(
    protein_a: Mapping[str, Any],
    protein_b: Mapping[str, Any],
    *,
    policy_path: Any = None,
) -> tuple[NeighborEvidence, ...]:
    record_a = _protein_record(protein_a)
    record_b = _protein_record(protein_b)
    policy, _ = _load_policy(policy_path)
    if policy["budgets"]["protein_pairs"] < 1:
        raise DatasetBuildError("protein pair budget exceeded", code="size_limit")
    try:
        Bio, aligner, _, algorithm = _alignment_context(policy)
    except _BuildAbort as exc:
        raise DatasetBuildError(str(exc), code=exc.code) from exc
    try:
        records, _ = _protein_neighbor_records(
            record_a, record_b, policy=policy, aligner=aligner,
            alignment_algorithm=algorithm, biopython_version=str(Bio.__version__),
            cells_remaining=policy["budgets"]["alignment_cells"],
        )
    except _BuildAbort as exc:
        raise DatasetBuildError(str(exc), code=exc.code) from exc
    return records


def connected_clusters(
    entity_ids: Iterable[str],
    neighbors: Iterable[NeighborEvidence | Mapping[str, Any]],
    *,
    prefix: str = "cluster-v1",
) -> tuple[Cluster, ...]:
    if not isinstance(prefix, str) or not prefix:
        raise DatasetBuildError("cluster prefix must be non-empty text")
    try:
        identifiers = list(entity_ids)
    except TypeError as exc:
        raise DatasetBuildError("entity_ids must be iterable") from exc
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise DatasetBuildError("cluster entity IDs must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise DatasetBuildError("duplicate entity IDs cannot form deterministic clusters")
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            next_identifier = parent[identifier]
            parent[identifier] = root
            identifier = next_identifier
        return root

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        low, high = sorted((root_left, root_right))
        parent[high] = low

    try:
        neighbor_values = list(neighbors)
    except TypeError as exc:
        raise DatasetBuildError("neighbors must be iterable") from exc
    for edge in neighbor_values:
        if isinstance(edge, Mapping):
            left, right = edge.get("entity_a"), edge.get("entity_b")
        elif isinstance(edge, NeighborEvidence):
            left, right = edge.entity_a, edge.entity_b
        else:
            raise DatasetBuildError("neighbor edges must be NeighborEvidence records")
        if left not in parent or right not in parent:
            raise DatasetBuildError("neighbor edge refers to an entity outside the cluster input")
        union(left, right)
    components: dict[str, list[str]] = {}
    for identifier in identifiers:
        components.setdefault(find(identifier), []).append(identifier)
    clusters = []
    for members in components.values():
        ordered = tuple(sorted(members))
        clusters.append(Cluster(cluster_id=f"{prefix}:{ordered[0]}", member_ids=ordered))
    return tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))


def _test_only(bundle: Mapping[str, Any], entities: Iterable[Mapping[str, Any]]) -> bool:
    return (bundle.get("source_kind") == "synthetic_test"
            or any(row.get("source_kind") == "synthetic_test" for row in entities))


def _cluster_result(
    *, status: str, bundle: Optional[Mapping[str, Any]], policy: Mapping[str, Any],
    policy_hash: str, sources: tuple[Mapping[str, Any], ...] = (),
    entities: Iterable[Mapping[str, Any]] = (), clusters: Iterable[Cluster] = (),
    neighbors: Iterable[NeighborEvidence] = (), excluded: Iterable[Mapping[str, Any]] = (),
    error: Optional[str] = None, test_only: bool = False,
) -> ClusterBuildResult:
    entity_values = tuple(deepcopy(dict(row)) for row in entities)
    cluster_values = tuple(clusters)
    neighbor_values = tuple(neighbors)
    excluded_values = tuple(deepcopy(dict(row)) for row in excluded)
    return ClusterBuildResult(
        status=status,
        entity_version=bundle.get("entity_version") if bundle else None,
        entity_bundle_hash=bundle.get("content_sha256") if bundle else None,
        policy_version=policy["policy_version"],
        policy_sha256=policy_hash,
        source_provenance=sources,
        entities=entity_values,
        clusters=cluster_values,
        neighbors=neighbor_values,
        excluded_entities=excluded_values,
        counts={"entities": len(entity_values), "clusters": len(cluster_values),
                "neighbors": len(neighbor_values), "excluded": len(excluded_values)},
        error=error,
        test_only=test_only,
        training_ready=False,
    )


def _molecule_cluster_from_bundle(
    bundle: Mapping[str, Any], policy: Mapping[str, Any], policy_hash: str,
    sources: tuple[Mapping[str, Any], ...],
) -> ClusterBuildResult:
    Chem, DataStructs, rdFingerprintGenerator, MurckoScaffold = _require_rdkit()
    entities, excluded = _molecule_entities(bundle)
    pair_count = len(entities) * (len(entities) - 1) // 2
    pair_budget = policy["budgets"]["molecule_pairs"]
    if pair_count > pair_budget:
        return _cluster_result(
            status="size_limit", bundle=bundle, policy=policy, policy_hash=policy_hash,
            sources=sources, entities=entities, excluded=excluded,
            error=f"molecule pair budget exceeded: {pair_count}>{pair_budget}",
            test_only=_test_only(bundle, entities),
        )
    prepared = [
        _make_molecule(Chem, rdFingerprintGenerator, MurckoScaffold, row, policy)
        for row in entities
    ]
    neighbor_rows: list[NeighborEvidence] = []
    for index, a in enumerate(prepared):
        for b in prepared[index + 1:]:
            neighbor_rows.extend(_molecule_neighbor(a, b, DataStructs, policy))
    neighbors = tuple(sorted(neighbor_rows, key=lambda row: (row.entity_a, row.entity_b, row.reason, row.score)))
    clusters = connected_clusters(
        (row["entity_id"] for row in entities), neighbors, prefix="molecule-cluster-v1",
    )
    return _cluster_result(
        status="partial" if excluded else "complete", bundle=bundle, policy=policy,
        policy_hash=policy_hash, sources=sources, entities=entities, clusters=clusters,
        neighbors=neighbors, excluded=excluded, test_only=_test_only(bundle, entities),
    )


def _protein_cluster_from_bundle(
    bundle: Mapping[str, Any], policy: Mapping[str, Any], policy_hash: str,
    sources: tuple[Mapping[str, Any], ...],
) -> ClusterBuildResult:
    Bio, aligner, _, algorithm = _alignment_context(policy)
    entities, excluded = _protein_entities(bundle)
    pair_count = len(entities) * (len(entities) - 1) // 2
    pair_budget = policy["budgets"]["protein_pairs"]
    if pair_count > pair_budget:
        return _cluster_result(
            status="size_limit", bundle=bundle, policy=policy, policy_hash=policy_hash,
            sources=sources, entities=entities, excluded=excluded,
            error=f"protein pair budget exceeded: {pair_count}>{pair_budget}",
            test_only=_test_only(bundle, entities),
        )
    records = [ProteinRecord(**{
        "accession": row["accession"], "taxonomy_id": row["taxonomy_id"],
        "raw_sequence": row["full_sequence"], "full_sequence": row["full_sequence"],
        "sequence_length": row["sequence_length"], "sequence_sha256": row["sequence_sha256"],
        "sequence_status": "verified", "sequence_status_reason": None,
        "identity_status": "verified", "source_kind": row["source_kind"],
        "source_provenance": list(row["source_provenance"]), "component_evidence": {"references": []},
        "returned_accession": row["accession"], "returned_taxonomy_id": row["taxonomy_id"],
    }) for row in entities]
    neighbor_rows: list[NeighborEvidence] = []
    alignment_cells = 0
    cell_budget = policy["budgets"]["alignment_cells"]
    for index, a in enumerate(records):
        for b in records[index + 1:]:
            remaining = cell_budget - alignment_cells
            neighbors, cells = _protein_neighbor_records(
                a, b, policy=policy, aligner=aligner, alignment_algorithm=algorithm,
                biopython_version=str(Bio.__version__), cells_remaining=remaining,
            )
            alignment_cells += cells
            neighbor_rows.extend(neighbors)
    neighbors = tuple(sorted(neighbor_rows, key=lambda row: (row.entity_a, row.entity_b, row.reason, row.score)))
    clusters = connected_clusters(
        (row["entity_id"] for row in entities), neighbors, prefix="protein-cluster-v1",
    )
    return _cluster_result(
        status="partial" if excluded else "complete", bundle=bundle, policy=policy,
        policy_hash=policy_hash, sources=sources, entities=entities, clusters=clusters,
        neighbors=neighbors, excluded=excluded, test_only=_test_only(bundle, entities),
    )


def _dependency_failure(policy: Mapping[str, Any], policy_hash: str, exc: _BuildAbort) -> ClusterBuildResult:
    return _cluster_result(
        status=exc.code, bundle=None, policy=policy, policy_hash=policy_hash,
        error=str(exc),
    )


def _build_public_cluster_result(entity_bundle_path: Any, *, policy_path: Any,
                                 molecule: bool) -> ClusterBuildResult:
    policy, policy_hash = _load_policy(policy_path)
    try:
        _require_rdkit()
        if not molecule:
            _require_bio()
    except _BuildAbort as exc:
        return _dependency_failure(policy, policy_hash, exc)
    bundle, _ = _load_verified_bundle(entity_bundle_path)
    sources = _bundle_sources(bundle)
    try:
        if molecule:
            return _molecule_cluster_from_bundle(bundle, policy, policy_hash, sources)
        return _protein_cluster_from_bundle(bundle, policy, policy_hash, sources)
    except _BuildAbort as exc:
        return _cluster_result(
            status=exc.code, bundle=bundle, policy=policy, policy_hash=policy_hash,
            sources=sources, error=str(exc), test_only=_bundle_is_test_only(bundle),
        )


def build_molecule_clusters(entity_bundle_path: Any, *, policy_path: Any = None) -> ClusterBuildResult:
    """Cluster only verified molecules loaded from a source-verified schema-v2 bundle."""
    return _build_public_cluster_result(entity_bundle_path, policy_path=policy_path, molecule=True)


def build_protein_clusters(entity_bundle_path: Any, *, policy_path: Any = None) -> ClusterBuildResult:
    """Cluster full, source-verified sequences from the same canonical entity bundle."""
    return _build_public_cluster_result(entity_bundle_path, policy_path=policy_path, molecule=False)


def strict_double_cold_split(
    entity_bundle_path: Any,
    *,
    policy_path: Any = None,
    label_policy_path: Any = None,
    allow_test_only: bool = False,
) -> SplitResult:
    """Split whole molecule/protein clusters and quarantine cross-partition pairs."""
    policy, policy_hash = _load_policy(policy_path)
    raise NotImplementedError("strict double-cold splitting is implemented in Task 2")
