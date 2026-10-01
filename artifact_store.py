"""Immutable, byte-bound publication for verified P0 entity bundles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
from typing import Any, Mapping, Optional

from data_contracts import canonical_json_sha256
from evidence_builder import EvidenceInputError, load_entity_bundle
from leak_audit import AuditReport, audit_split


_VERSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TIME_FIELDS = frozenset({
    "created_at", "updated_at", "generated_at", "retrieved_at", "fetched_at",
    "timestamp", "started_at", "completed_at", "built_at", "audited_at",
    "collected_at", "checked_at", "published_at", "last_modified",
})
_CODE_ROOT = Path(__file__).resolve().parent


class FrozenArtifactError(ValueError):
    """A frozen artifact could not be safely validated or published."""


@dataclass(frozen=True)
class FrozenWriteResult:
    status: str
    exit_code: int
    output_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    semantic_sha256: Optional[str] = None
    audit_report: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "semantic_sha256": self.semantic_sha256,
            "audit_report": dict(self.audit_report) if isinstance(self.audit_report, Mapping) else self.audit_report,
            "error": self.error,
        }


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"manifest contains a non-canonical JSON value: {exc}") from exc


def _without_time_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_time_fields(item)
            for key, item in value.items()
            if str(key).casefold() not in _TIME_FIELDS and str(key) != "semantic_sha256"
        }
    if isinstance(value, (list, tuple)):
        return [_without_time_fields(item) for item in value]
    return value


def semantic_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash manifest meaning while excluding nested timestamps and the digest field."""
    if not isinstance(manifest, Mapping):
        raise FrozenArtifactError("manifest must be an object")
    return canonical_json_sha256(_without_time_fields(manifest))


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _intersects(left: Path, right: Path) -> bool:
    left_key = _path_key(left)
    right_key = _path_key(right)
    try:
        return os.path.commonpath((left_key, right_key)) in {left_key, right_key}
    except ValueError:
        return False


def _is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        import stat
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & reparse_flag)
    except (OSError, AttributeError):
        return False


def _assert_no_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = absolute
    while True:
        if cursor.exists() and _is_reparse_path(cursor):
            raise FrozenArtifactError(f"output path crosses a symlink or junction: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def _protected_paths(apl_root: Path) -> tuple[Path, ...]:
    return tuple(
        apl_root / relative for relative in (
            "data", "framework_output", "offline_data.json", ".model_cache", "features", "output_dashboard.html",
        )
    )


def _validate_output_root(value: Any, *, apl_root: Path) -> Path:
    if value is None or not os.fspath(value).strip():
        raise FrozenArtifactError("an explicit independent output root is required")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise FrozenArtifactError("output root must be an absolute path")
    _assert_no_reparse_components(candidate)
    resolved = candidate.resolve(strict=False)
    for protected in _protected_paths(apl_root):
        if _intersects(resolved, protected):
            raise FrozenArtifactError(f"output root intersects protected history: {protected}")
    return resolved


def _safe_bundle_file(bundle_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FrozenArtifactError("source snapshot path must be non-empty relative text")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise FrozenArtifactError(f"source snapshot path escapes producer bundle: {relative!r}")
    unresolved = bundle_root / path
    _assert_no_reparse_components(unresolved)
    resolved_root = bundle_root.resolve(strict=True)
    resolved = unresolved.resolve(strict=True)
    try:
        common = os.path.normcase(os.path.commonpath((str(resolved_root), str(resolved))))
    except ValueError as exc:
        raise FrozenArtifactError("source snapshot path is on another volume") from exc
    if common != os.path.normcase(str(resolved_root)) or not resolved.is_file():
        raise FrozenArtifactError(f"source snapshot is missing or escapes producer bundle: {relative!r}")
    return resolved


def _git_state(code_root: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(code_root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    git_root = run("rev-parse", "--show-toplevel")
    if not git_root:
        return {"status": "not_initialized", "git_root": None, "branch": None, "commit": None,
                "dirty": None, "dirty_paths": []}
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    if not branch or not commit or status is None:
        return {"status": "git_unavailable", "git_root": git_root, "branch": branch, "commit": commit,
                "dirty": None, "dirty_paths": []}
    dirty_paths = [line[2:].lstrip() for line in status.splitlines() if len(line) >= 3]
    return {
        "status": "git",
        "git_root": git_root,
        "branch": branch,
        "commit": commit,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _exclusive_claim(version_dir: Path) -> tuple[str, Optional[str]]:
    try:
        version_dir.mkdir(parents=False, exist_ok=False)
        return "claimed", None
    except FileExistsError:
        manifest_path = version_dir / "manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            return "exists", str(manifest_path)
        return "busy", None
    except OSError as exc:
        return "incomplete", str(exc)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_atomic(destination: Path, payload: bytes, token: str, temporary_paths: set[Path]) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp.{token}")
    _assert_no_reparse_components(temporary)
    if temporary.exists() or destination.exists():
        raise FrozenArtifactError(f"refusing to replace an existing frozen file: {destination}")
    temporary_paths.add(temporary)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    temporary_paths.discard(temporary)
    _fsync_directory(destination.parent)


def _cleanup_own_temps(version_dir: Path, token: str, temporary_paths: set[Path]) -> None:
    root = version_dir.resolve(strict=False)
    for path in tuple(temporary_paths):
        try:
            resolved = path.resolve(strict=False)
            relative = resolved.relative_to(root)
            if (not relative.parts or token not in resolved.name or not resolved.name.endswith(token)):
                continue
            if path.is_file() and not path.is_symlink():
                path.unlink()
        except OSError:
            continue
        finally:
            temporary_paths.discard(path)


def _validate_split_claim(split: Any, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(split, Mapping):
        raise FrozenArtifactError("split claim must be an object")
    result = deepcopy(dict(split))
    entity_version = bundle.get("entity_version")
    entity_hash = bundle.get("content_sha256")
    for key, expected in (("entity_version", entity_version), ("entity_bundle_hash", entity_hash)):
        claimed = result.get(key)
        if claimed is not None and claimed != expected:
            raise FrozenArtifactError(f"split claim {key} does not match the source-verified entity bundle")
        result[key] = expected
    required = (
        "strict_split_status", "split_policy_sha256", "molecule_memberships", "protein_memberships",
        "pair_assignments", "crosspair_quarantine", "unassigned_pairs", "test_only_pair_assignments",
        "ambiguous_label_quarantine", "class_support", "training_ready",
    )
    if any(key not in result for key in required):
        raise FrozenArtifactError("split claim omits a required assignment/readiness ledger")
    if type(result.get("training_ready")) is not bool:
        raise FrozenArtifactError("split claim training_ready must be boolean")
    return result


def _pair_payload(split: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "pair_assignments", "crosspair_quarantine", "unassigned_pairs",
        "test_only_pair_assignments", "ambiguous_label_quarantine",
    )
    return {field_name: split.get(field_name) for field_name in fields}


class FrozenManifestWriter:
    """Publish one exclusive version directory; manifest.json is the final commit."""

    def __init__(self, output_root: os.PathLike[str] | str, version_id: str, *, code_root: Any = None) -> None:
        self.output_root = Path(output_root).expanduser()
        self.version_id = version_id
        self.code_root = Path(code_root).resolve(strict=False) if code_root is not None else _CODE_ROOT

    def publish(self, entity_bundle_path: os.PathLike[str] | str, *, split: Mapping[str, Any]) -> FrozenWriteResult:
        try:
            if not isinstance(self.version_id, str) or not _VERSION_ID.fullmatch(self.version_id) or self.version_id in {".", ".."}:
                raise FrozenArtifactError("version_id must be a simple filesystem-safe identifier")
            apl_root = Path(__file__).resolve().parent
            output_root = _validate_output_root(self.output_root, apl_root=apl_root)
            requested_entity_path = Path(entity_bundle_path).expanduser()
            _assert_no_reparse_components(requested_entity_path)
            entity_path = requested_entity_path.resolve(strict=True)
            if not entity_path.is_file():
                raise FrozenArtifactError("entity_bundle_path must be a regular file")
            bundle_root = entity_path.parent
            bundle_bytes = entity_path.read_bytes()
            bundle = load_entity_bundle(entity_path, verify_sources=True)
            split_payload = _validate_split_claim(split, bundle)
            version_dir = output_root / self.version_id
            if _intersects(version_dir, bundle_root):
                raise FrozenArtifactError("frozen output version intersects its source bundle")
            output_root.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_components(output_root)
            claim, existing_manifest = _exclusive_claim(version_dir)
            if claim == "exists":
                return FrozenWriteResult("exists", 0, str(version_dir), existing_manifest)
            if claim == "busy":
                return FrozenWriteResult("busy", 2, str(version_dir), error="version directory is already claimed without a committed manifest")
            if claim != "claimed":
                return FrozenWriteResult("incomplete", 2, str(version_dir), error=existing_manifest or "could not claim output version")
        except (FrozenArtifactError, EvidenceInputError, OSError, TypeError, ValueError) as exc:
            return FrozenWriteResult("incomplete", 2, error=str(exc))

        token = uuid.uuid4().hex
        temporary_paths: set[Path] = set()
        manifest_path = version_dir / "manifest.json"
        try:
            # Read every referenced raw or metadata receipt, bind its exact bytes,
            # and copy it under the same relative path as the producer bundle.
            snapshots: dict[str, tuple[Path, bytes, str]] = {}
            for source in bundle.get("source_provenance", []):
                if not isinstance(source, Mapping):
                    raise FrozenArtifactError("source provenance contains a non-object entry")
                relative = source.get("source_path")
                expected_hash = source.get("raw_sha256")
                source_path = _safe_bundle_file(bundle_root, relative)
                raw = source_path.read_bytes()
                actual_hash = hashlib.sha256(raw).hexdigest()
                if actual_hash != expected_hash:
                    raise FrozenArtifactError(f"source snapshot changed before freeze: {relative}")
                if relative in snapshots and snapshots[relative][2] != actual_hash:
                    raise FrozenArtifactError(f"source path has conflicting receipts: {relative}")
                snapshots[relative] = (source_path, raw, actual_hash)

            _write_atomic(version_dir / "entities.json", bundle_bytes, token, temporary_paths)
            for relative in sorted(snapshots):
                destination = version_dir / Path(relative)
                try:
                    destination.resolve(strict=False).relative_to(version_dir.resolve(strict=True))
                except ValueError as exc:
                    raise FrozenArtifactError(f"frozen snapshot destination escapes claimed version: {relative}") from exc
                _assert_no_reparse_components(destination.parent)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _assert_no_reparse_components(destination.parent)
                _write_atomic(destination, snapshots[relative][1], token, temporary_paths)

            activity_rows = bundle.get("activities", [])
            pair_payload = _pair_payload(split_payload)
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "version_id": self.version_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "entities": {
                    "path": "entities.json",
                    "schema_version": bundle["schema_version"],
                    "entity_version": bundle["entity_version"],
                    "content_sha256": bundle["content_sha256"],
                    "file_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
                },
                "source_provenance": deepcopy(bundle.get("source_provenance", [])),
                "activities": {
                    "entity_version": bundle["entity_version"],
                    "entity_bundle_hash": bundle["content_sha256"],
                    "record_count": len(activity_rows),
                    "semantic_sha256": canonical_json_sha256(activity_rows),
                },
                "pairs": {
                    "entity_version": bundle["entity_version"],
                    "entity_bundle_hash": bundle["content_sha256"],
                    "semantic_sha256": canonical_json_sha256(pair_payload),
                    "ledger_counts": {key: len(value) for key, value in pair_payload.items()},
                },
                "split": split_payload,
                "code_version_control": _git_state(self.code_root),
                "training_ready": bool(split_payload.get("training_ready")),
            }
            manifest["semantic_sha256"] = semantic_manifest_hash(manifest)
            provisional = version_dir / f".manifest.audit.tmp.{token}"
            temporary_paths.add(provisional)
            with provisional.open("xb") as stream:
                stream.write(_canonical_bytes(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            audit: AuditReport = audit_split(provisional)
            provisional.unlink()
            temporary_paths.discard(provisional)
            if audit.exit_code != 0:
                status = "audit_violation" if audit.exit_code == 3 else "not_auditable" if audit.exit_code == 4 else "incomplete"
                _cleanup_own_temps(version_dir, token, temporary_paths)
                return FrozenWriteResult(
                    status,
                    audit.exit_code,
                    str(version_dir),
                    audit_report=audit.to_dict(),
                    error=audit.error or "independent audit did not return clean",
                )

            # Verify that neither the producer's inputs nor our staged copies
            # changed while the independent audit was reading them.
            if entity_path.read_bytes() != bundle_bytes or hashlib.sha256((version_dir / "entities.json").read_bytes()).hexdigest() != manifest["entities"]["file_sha256"]:
                raise FrozenArtifactError("entities.json changed during frozen publication")
            for relative, (source_path, original_bytes, expected_hash) in snapshots.items():
                if source_path.read_bytes() != original_bytes:
                    raise FrozenArtifactError(f"source input changed while audit ran: {relative}")
                frozen_path = version_dir / Path(relative)
                if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != expected_hash:
                    raise FrozenArtifactError(f"frozen raw or metadata bytes changed before manifest commit: {relative}")

            manifest["audit"] = {"status": "clean", "exit_code": 0, "finding_count": 0}
            manifest["training_ready"] = audit.training_ready
            split_payload["training_ready"] = audit.training_ready
            manifest["semantic_sha256"] = semantic_manifest_hash(manifest)
            _write_atomic(manifest_path, _canonical_bytes(manifest), token, temporary_paths)
            return FrozenWriteResult(
                "complete", 0, str(version_dir), str(manifest_path), manifest["semantic_sha256"], audit.to_dict(),
            )
        except (FrozenArtifactError, EvidenceInputError, OSError, TypeError, ValueError) as exc:
            _cleanup_own_temps(version_dir, token, temporary_paths)
            return FrozenWriteResult("incomplete", 2, str(version_dir), error=str(exc))
