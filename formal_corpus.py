"""Versioned, fail-closed preflight for the first formal ChEMBL cohort.

Collection is intentionally not implemented here until a source-only candidate
selection receipt is sealed.  This module provides the resource gate used by
that later entry point and keeps the development pilot controller untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import artifact_store
import dataset_builder
import evidence_builder
import leak_audit


APL_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APL_ROOT / "configs" / "corpus_formal_v1.json"
MAX_ESM_TOKENS = 1024
SPECIAL_TOKEN_RESERVE = 2
MAX_SUPPORTED_RESIDUES = MAX_ESM_TOKENS - SPECIAL_TOKEN_RESERVE


class FormalCorpusError(ValueError):
    """A formal protocol or resource gate failed closed."""


class FormalSizeLimit(FormalCorpusError):
    """Actual resource use crossed a registered formal batch limit."""


class RuntimeComputeBudget:
    """Measure non-overlapping wall time and stop before the next full-neighbor pass."""

    def __init__(self, caps: Mapping[str, Any], clock=time.perf_counter):
        self.caps = caps
        self.clock = clock
        self.started = clock()
        self.records: list[dict[str, Any]] = []
        self.exceeded_reason: str | None = None

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started)

    def check_total(self) -> None:
        if self.elapsed() > self.caps["pipeline_total_wall_seconds"]:
            self.exceeded_reason = "actual_pipeline_compute_wall_limit_exceeded"
            raise FormalSizeLimit(self.exceeded_reason)

    def run_pass(self, name: str, function: Any, wall_cap: float, *args: Any, **kwargs: Any) -> Any:
        self.check_total()
        started = self.clock()
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            elapsed = max(0.0, self.clock() - started)
            within_pass = elapsed <= wall_cap
            record = {"pass": name, "wall_seconds": elapsed, "cap_seconds": wall_cap,
                      "status": "failed" if within_pass else "size_limit",
                      "error_type": type(exc).__name__}
            self.records.append(record)
            if not within_pass:
                self.exceeded_reason = f"actual_{name}_wall_limit_exceeded_after_failure"
                raise FormalSizeLimit(self.exceeded_reason) from exc
            try:
                self.check_total()
            except FormalSizeLimit as limit_exc:
                record["status"] = "size_limit"
                record["limit_reason"] = str(limit_exc)
                raise limit_exc from exc
            raise
        elapsed = max(0.0, self.clock() - started)
        record = {"pass": name, "wall_seconds": elapsed, "cap_seconds": wall_cap,
                  "status": "within_budget" if elapsed <= wall_cap else "size_limit"}
        self.records.append(record)
        if elapsed > wall_cap:
            self.exceeded_reason = f"actual_{name}_wall_limit_exceeded"
            raise FormalSizeLimit(self.exceeded_reason)
        self.check_total()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"global_cap_seconds": self.caps["pipeline_total_wall_seconds"],
                "actual_elapsed_seconds": self.elapsed(), "status": "size_limit" if self.exceeded_reason else "within_budget",
                "exceeded_reason": self.exceeded_reason, "passes": list(self.records),
                "overlapping_elapsed_not_summed": True}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _validate_ledger_shape(ledger: Mapping[str, Any], limits: Mapping[str, Any],
                           ledger_root: Path | None = None) -> None:
    if (type(ledger.get("schema_version")) is not int or ledger.get("schema_version") != 1
            or ledger.get("protocol_version") != "corpus-formal-v1.0"
            or not isinstance(ledger.get("attempts"), list)):
        raise FormalCorpusError("prior request ledger schema/protocol is invalid")
    attempts = ledger["attempts"]
    for index, item in enumerate(attempts, 1):
        if (not isinstance(item, Mapping) or type(item.get("attempt_id")) is not int
                or item.get("attempt_id") != index or item.get("service") not in {"chembl", "uniprot"}
                or item.get("phase") not in {"discovery", "capture"}
                or item.get("status") not in {"pending", "succeeded", "failed", "recovered_success", "recovered_unknown"}
                or type(item.get("activity_page")) is not bool):
            raise FormalCorpusError(f"prior ledger attempt {index} has invalid identity/status fields")
        parsed = urlparse(item.get("url", ""))
        host = "rest.uniprot.org" if item.get("service") == "uniprot" else "www.ebi.ac.uk"
        if parsed.scheme != "https" or parsed.hostname != host:
            raise FormalCorpusError(f"prior ledger attempt {index} URL is outside the registered official host")
        numeric = ("request_timeout_seconds", "elapsed_seconds", "charged_wall_seconds")
        for field in numeric:
            value = item.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not __import__("math").isfinite(value) or value < 0):
                raise FormalCorpusError(f"prior ledger attempt {index} {field} is not finite/non-negative")
        for field in ("response_bytes", "charged_bytes", "activity_rows", "charged_rows"):
            if field in item and (type(item[field]) is not int or item[field] < 0):
                raise FormalCorpusError(f"prior ledger attempt {index} {field} is not a non-negative strict integer")
        if item["status"] in {"succeeded", "recovered_success"} and (
                not isinstance(item.get("raw_path"), str) or not isinstance(item.get("raw_sha256"), str)
                or len(item["raw_sha256"]) != 64):
            raise FormalCorpusError(f"prior ledger attempt {index} success lacks raw response path/SHA")
        if item["activity_page"] and (item["service"] != "chembl" or not parsed.path.endswith("/activity.json")):
            raise FormalCorpusError(f"prior ledger attempt {index} activity-page binding is invalid")
        if ledger_root is not None:
            batch_root = Path(item.get("prior_batch_root", ledger_root)).resolve(strict=True)
            for path_key, hash_key in (("raw_path", "raw_sha256"), ("partial_path", "partial_sha256")):
                relative = item.get(path_key)
                if relative is None:
                    continue
                if path_key == "raw_path" and item.get("status") not in {"succeeded", "recovered_success"}:
                    possible = (batch_root / relative).resolve()
                    if not possible.exists():
                        continue
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise FormalCorpusError(f"prior ledger attempt {index} {path_key} is unsafe")
                payload_path = (batch_root / relative_path).resolve(strict=True)
                if batch_root not in payload_path.parents or _sha(payload_path.read_bytes()) != item.get(hash_key):
                    raise FormalCorpusError(f"prior ledger attempt {index} {path_key} bytes/hash mismatch")
    chembl = [item for item in attempts if item["service"] == "chembl"]
    uniprot = [item for item in attempts if item["service"] == "uniprot"]
    pages = [item for item in attempts if item["activity_page"]]
    charged = lambda rows: sum(int(item.get("charged_bytes", 8_388_608 if item["status"] in {"pending", "recovered_unknown"} else 0)) for item in rows)
    wall = sum(float(item.get("charged_wall_seconds", 30.0 if item["status"] in {"pending", "recovered_unknown"} else 0.0)) for item in attempts)
    row_count = sum(int(item.get("charged_rows", 100 if item["activity_page"] and item["status"] in {"pending", "recovered_unknown"} else item.get("activity_rows", 0))) for item in pages)
    if (len(chembl) > limits["chembl_requests"] or len(uniprot) > limits["uniprot_requests"]
            or len(pages) > limits["chembl_activity_pages"] or row_count > limits["chembl_activity_rows"]
            or charged(chembl) > limits["chembl_response_bytes_total"]
            or charged(uniprot) > limits["uniprot_response_bytes_total"]
            or wall > limits["chembl_wall_seconds"]):
        raise FormalCorpusError("prior request ledger counters exceed a registered batch budget")


class BatchRegistry:
    """Repository-local guard preventing a new GUID from resetting a formal budget."""

    @classmethod
    def validate_start(cls, config: Mapping[str, Any], config_sha256: str,
                       resume_from: Path | None, *, registry_path: Path | None = None) -> Path | None:
        path = registry_path or (APL_ROOT / "configs" / "corpus_formal_v1.batch-ledger.json")
        if not path.exists():
            if resume_from is not None:
                raise FormalCorpusError("--resume-from was supplied but no matching formal batch registry exists")
            return None
        registry, _ = _read_object(path)
        if (registry.get("protocol_version") != config["protocol_version"]
                or registry.get("decision_sha256") != config["budget"]["decision_sha256"]
                or registry.get("config_sha256") != config_sha256):
            raise FormalCorpusError("formal batch registry is bound to another protocol; a new decision/version and explicit history link is required")
        if registry.get("status") in {"complete", "candidate_audited_not_ready", "selection_window_exhausted", "size_limit"}:
            raise FormalCorpusError("this formal budget batch is terminal; a new GUID cannot reset its caps")
        latest = Path(registry.get("latest_root", "")).resolve(strict=True)
        if resume_from is None or resume_from.expanduser().resolve(strict=True) != latest:
            raise FormalCorpusError("same formal batch retry requires --resume-from the registered latest ledger root")
        prior_ledger, prior_raw = _read_object((latest / "request_ledger.json").resolve(strict=True))
        if _sha(prior_raw) != registry.get("latest_ledger_sha256"):
            raise FormalCorpusError("registered latest ledger SHA mismatch; batch resume is blocked to prevent budget reset")
        if (prior_ledger.get("config_sha256") != config_sha256
                or prior_ledger.get("decision_sha256") != config["budget"]["decision_sha256"]):
            raise FormalCorpusError("registered request ledger is bound to another raw config/decision hash")
        decision, _ = _read_object(APL_ROOT / "configs" / "corpus_formal_budget_decision_v1.json")
        _validate_ledger_shape(prior_ledger, decision["caps"], ledger_root=latest)
        receipt_sha = registry.get("selection_receipt_sha256")
        if receipt_sha is not None:
            selection_root = Path(registry.get("selection_root", "")).resolve(strict=True)
            _verify_selection_receipt(selection_root / "selection_receipt.json", config, config_sha256,
                                      expected_sha256=receipt_sha)
        return latest

    def __init__(self, config: Mapping[str, Any], config_sha256: str, output_root: Path,
                 resume_from: Path | None, *, registry_path: Path | None = None):
        self.path = registry_path or (APL_ROOT / "configs" / "corpus_formal_v1.batch-ledger.json")
        self.protocol = config["protocol_version"]
        self.decision_sha256 = config["budget"]["decision_sha256"]
        self.config_sha256 = config_sha256
        self.root = output_root.resolve()
        previous_registry = _read_object(self.path)[0] if self.path.exists() else {}
        self.prior_root = self.validate_start(config, config_sha256, resume_from, registry_path=self.path)
        if not self.root.is_dir():
            raise FormalCorpusError("batch output root must be created after preflight and before registry registration")
        self._write({"schema_version": 1, "protocol_version": self.protocol,
                     "decision_sha256": self.decision_sha256, "config_sha256": self.config_sha256,
                     "status": "in_progress", "latest_root": str(self.root),
                     "prior_root": str(self.prior_root) if self.prior_root else None,
                     "latest_ledger_sha256": None,
                     "selection_receipt_sha256": previous_registry.get("selection_receipt_sha256"),
                     "selection_root": previous_registry.get("selection_root"),
                     "started_utc": datetime.now(timezone.utc).isoformat()})

    def _write(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f"{self.path.name}.{os.urandom(8).hex()}.tmp")
        with temp.open("xb") as stream:
            stream.write(_json_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)

    def update_ledger(self, root: Path, ledger_sha256: str) -> None:
        current, _ = _read_object(self.path)
        current.update(latest_root=str(root.resolve()), latest_ledger_sha256=ledger_sha256,
                       updated_utc=datetime.now(timezone.utc).isoformat())
        self._write(current)

    def set_selection(self, root: Path, receipt_sha256: str) -> None:
        current, _ = _read_object(self.path)
        current.update(selection_root=str(root.resolve()), selection_receipt_sha256=receipt_sha256,
                       updated_utc=datetime.now(timezone.utc).isoformat())
        self._write(current)

    def finish(self, status: str, receipt_sha256: str | None = None) -> None:
        current, _ = _read_object(self.path)
        current.update(status=status, result_receipt_sha256=receipt_sha256,
                       completed_utc=datetime.now(timezone.utc).isoformat())
        self._write(current)


class RequestLedger:
    """Durable whole-batch request accounting with conservative crash recovery."""

    def __init__(self, root: Path, config: Mapping[str, Any], decision: Mapping[str, Any], *,
                 prior_root: Path | None = None, registry: BatchRegistry | None = None,
                 config_sha256: str | None = None):
        self.root = root
        self.path = root / "request_ledger.json"
        self.config_sha256 = config_sha256 or _sha(_json_bytes(dict(config)))
        self.registry = registry
        self.decision_sha256 = config["budget"]["decision_sha256"]
        self.limits = dict(decision["caps"])
        self.phase = "discovery"
        self.attempts: list[dict[str, Any]] = []
        self.prior_ledger_sha256 = None
        if prior_root is not None:
            prior_path = prior_root / "request_ledger.json"
            prior, prior_raw = _read_object(prior_path.resolve(strict=True))
            if (prior.get("protocol_version") != config["protocol_version"]
                    or prior.get("decision_sha256") != self.decision_sha256
                    or prior.get("config_sha256") != self.config_sha256):
                raise FormalCorpusError("resume ledger does not match the frozen formal protocol/budget")
            _validate_ledger_shape(prior, self.limits)
            self.prior_ledger_sha256 = _sha(prior_raw)
            for original in prior.get("attempts", []):
                item = dict(original)
                if item.get("status") == "pending":
                    relative = item.get("raw_path")
                    raw_path = (prior_root / relative).resolve() if isinstance(relative, str) else None
                    if raw_path is not None and prior_root.resolve() in raw_path.parents and raw_path.is_file():
                        raw = raw_path.read_bytes()
                        if item.get("raw_sha256") in (None, _sha(raw)):
                            item.update(status="recovered_success", response_bytes=len(raw), charged_bytes=len(raw),
                                        raw_sha256=_sha(raw), elapsed_seconds=item.get("elapsed_seconds", 0.0))
                        else:
                            item.update(status="recovered_unknown", charged_bytes=8_388_608,
                                        charged_wall_seconds=30.0)
                    else:
                        item.update(status="recovered_unknown", charged_bytes=8_388_608,
                                    charged_wall_seconds=30.0)
                elif item.get("status") in {"succeeded", "recovered_success"} and isinstance(item.get("raw_path"), str):
                    raw_path = (prior_root / item["raw_path"]).resolve()
                    if prior_root.resolve() not in raw_path.parents or not raw_path.is_file():
                        raise FormalCorpusError("resume ledger raw receipt is missing or escapes its batch root")
                    if _sha(raw_path.read_bytes()) != item.get("raw_sha256"):
                        raise FormalCorpusError("resume ledger raw receipt hash mismatch")
                elif item.get("status") == "failed" and isinstance(item.get("partial_path"), str):
                    partial_path = (prior_root / item["partial_path"]).resolve()
                    if prior_root.resolve() not in partial_path.parents or not partial_path.is_file():
                        raise FormalCorpusError("resume ledger partial response receipt is missing or escapes its batch root")
                    if _sha(partial_path.read_bytes()) != item.get("partial_sha256"):
                        raise FormalCorpusError("resume ledger partial response hash mismatch")
                item["prior_batch_root"] = str(prior_root.resolve())
                self.attempts.append(item)
        self._persist()

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "protocol_version": "corpus-formal-v1.0",
                   "config_sha256": self.config_sha256, "decision_sha256": self.decision_sha256,
                   "prior_ledger_sha256": self.prior_ledger_sha256,
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "attempts": self.attempts}
        temp = self.root / f"request_ledger-{os.urandom(8).hex()}.tmp"
        with temp.open("xb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)
        if self.registry is not None:
            self.registry.update_ledger(self.root, _sha(self.path.read_bytes()))

    def _used(self, service: str | None = None) -> dict[str, float]:
        rows = [item for item in self.attempts if service is None or item.get("service") == service]
        return {
            "requests": float(len(rows)),
            "pages": float(sum(bool(item.get("activity_page")) for item in rows)),
            "rows": float(sum(int(item.get("charged_rows", 100 if item.get("activity_page") and item.get("status") in {"pending", "recovered_unknown"} else item.get("activity_rows", 0))) for item in rows)),
            "bytes": float(sum(int(item.get("charged_bytes", 8_388_608 if item.get("status") in {"pending", "recovered_unknown"} else item.get("response_bytes", 0))) for item in rows)),
            "wall": float(sum(float(item.get("charged_wall_seconds", item.get("request_timeout_seconds", 30.0) if item.get("status") in {"pending", "recovered_unknown"} else item.get("elapsed_seconds", 0.0))) for item in rows)),
        }

    def before(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        service = "uniprot" if parsed.hostname == "rest.uniprot.org" else "chembl"
        activity_page = parsed.hostname == "www.ebi.ac.uk" and parsed.path.endswith("/activity.json")
        usage = self._used()
        service_usage = self._used(service)
        timeout = min(30.0, max(0.001, self.limits["chembl_wall_seconds"] - usage["wall"]))
        limits = self.limits
        if service == "chembl" and service_usage["requests"] + 1 > limits["chembl_requests"]:
            raise FormalCorpusError("whole formal ChEMBL request budget exhausted")
        if service == "uniprot" and service_usage["requests"] + 1 > limits["uniprot_requests"]:
            raise FormalCorpusError("whole formal UniProt request budget exhausted")
        if usage["wall"] + timeout > limits["chembl_wall_seconds"]:
            timeout = limits["chembl_wall_seconds"] - usage["wall"]
        if timeout <= 0:
            raise FormalCorpusError("whole formal wall-time budget exhausted")
        if service == "uniprot":
            if service_usage["requests"] + 1 > limits["uniprot_requests"]:
                raise FormalCorpusError("whole formal UniProt request budget exhausted")
            max_service_bytes = limits["uniprot_response_bytes_total"]
        else:
            if service_usage["requests"] + 1 > limits["chembl_requests"]:
                raise FormalCorpusError("whole formal ChEMBL request budget exhausted")
            max_service_bytes = limits["chembl_response_bytes_total"]
        if service_usage["bytes"] + limits["chembl_response_bytes_per_request"] > max_service_bytes:
            raise FormalCorpusError(f"whole formal {service} response-byte budget exhausted")
        if service == "chembl":
            phase_bytes = sum(int(item.get("charged_bytes", 8_388_608 if item.get("status") in {"pending", "recovered_unknown"} else item.get("response_bytes", 0)))
                              for item in self.attempts if item.get("service") == "chembl" and item.get("phase") == self.phase)
            phase_cap = 134_217_728 if self.phase == "discovery" else 402_653_184
            if phase_bytes + limits["chembl_response_bytes_per_request"] > phase_cap:
                raise FormalCorpusError(f"formal {self.phase} response-byte reserve exhausted")
        if activity_page:
            if usage["pages"] + 1 > limits["chembl_activity_pages"] or usage["rows"] + 100 > limits["chembl_activity_rows"]:
                raise FormalCorpusError("whole formal Activity page/row budget exhausted")
            if self.phase == "discovery":
                phase_usage = self._used()
                discovery_pages = sum(bool(item.get("activity_page")) and item.get("phase") == "discovery" for item in self.attempts)
                discovery_rows = sum(int(item.get("charged_rows", 100 if item.get("activity_page") and item.get("status") in {"pending", "recovered_unknown"} else item.get("activity_rows", 0))) for item in self.attempts if item.get("phase") == "discovery")
                if discovery_pages + 1 > 10 or discovery_rows + 100 > 1000:
                    raise FormalCorpusError("registered source discovery window exhausted")
        if self.phase == "discovery" and service == "chembl":
            discovery_requests = sum(item.get("phase") == "discovery" and item.get("service") == "chembl" for item in self.attempts)
            if discovery_requests + 1 > 512:
                raise FormalCorpusError("registered source discovery request budget exhausted")
        if self.phase == "capture" and service == "chembl":
            capture_usage = [item for item in self.attempts if item.get("phase") == "capture" and item.get("service") == "chembl"]
            if len(capture_usage) + 1 > 1536:
                raise FormalCorpusError("registered capture request reserve exhausted")
            capture_pages = sum(bool(item.get("activity_page")) for item in capture_usage)
            capture_rows = sum(int(item.get("charged_rows", 100 if item.get("activity_page") and item.get("status") in {"pending", "recovered_unknown"} else item.get("activity_rows", 0))) for item in capture_usage)
            if activity_page and (capture_pages + 1 > 246 or capture_rows + 100 > 19000):
                raise FormalCorpusError("registered capture page/row reserve exhausted")
        attempt = {"attempt_id": len(self.attempts) + 1, "phase": self.phase, "service": service,
                   "url": url, "activity_page": bool(activity_page), "status": "pending",
                   "request_timeout_seconds": timeout, "started_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": 0.0,
                   "charged_bytes": 8_388_608, "charged_wall_seconds": timeout,
                   "charged_rows": 100 if activity_page else 0,
                   "raw_path": f"raw/http/attempt-{len(self.attempts) + 1:05d}.bin"}
        self.attempts.append(attempt)
        self._persist()  # durable before the actual network request
        return attempt

    def finish(self, attempt: dict[str, Any], *, status: str, elapsed: float,
               response_bytes: int = 0, raw_path: str | None = None, raw_sha256: str | None = None,
               partial_path: str | None = None, partial_sha256: str | None = None,
               charged_bytes: int | None = None,
               error: str | None = None) -> None:
        attempt.update(status=status, elapsed_seconds=max(0.0, float(elapsed)),
                       response_bytes=max(0, int(response_bytes)),
                       charged_bytes=max(0, int(response_bytes if charged_bytes is None else charged_bytes)),
                       charged_wall_seconds=max(0.0, float(elapsed)), raw_path=raw_path,
                       raw_sha256=raw_sha256, error=error)
        if partial_path is not None:
            attempt["partial_path"] = partial_path
            attempt["partial_sha256"] = partial_sha256
        if attempt.get("activity_page"):
            attempt["activity_rows"] = 100  # query page is charged by its fixed 100-row limit
            attempt["charged_rows"] = 100
        self._persist()

    def reconcile_activity_pages(self) -> None:
        for item in self.attempts:
            if not item.get("activity_page") or item.get("status") not in {"succeeded", "recovered_success"}:
                continue
            batch_root = Path(item.get("prior_batch_root", self.root))
            raw_path = (batch_root / item["raw_path"]).resolve()
            raw = raw_path.read_bytes()
            if _sha(raw) != item.get("raw_sha256"):
                raise FormalCorpusError("Activity request raw receipt changed before page reconciliation")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FormalCorpusError("Activity raw response is not valid JSON") from exc
            rows = payload.get("activities") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list) or len(rows) > 100:
                raise FormalCorpusError("Activity response rows differ from the fixed 100-row query bound")
            item["activity_rows"] = len(rows)
            item["charged_rows"] = len(rows)
        self._persist()

    def summary(self) -> dict[str, Any]:
        all_usage = self._used()
        actual_bytes = lambda service: sum(int(item.get("response_bytes", 0)) for item in self.attempts
                                            if item.get("service") == service)
        return {
            "chembl_requests": int(self._used("chembl")["requests"]),
            "chembl_activity_pages": int(all_usage["pages"]),
            "chembl_activity_rows": int(all_usage["rows"]),
            "chembl_response_bytes_actual": actual_bytes("chembl"),
            "chembl_response_bytes_charged": int(self._used("chembl")["bytes"]),
            "uniprot_requests": int(self._used("uniprot")["requests"]),
            "uniprot_response_bytes_actual": actual_bytes("uniprot"),
            "uniprot_response_bytes_charged": int(self._used("uniprot")["bytes"]),
            "wall_seconds_charged": all_usage["wall"],
            "caps": self.limits,
        }

    def settle_pending(self, reason: str) -> None:
        for item in self.attempts:
            if item.get("status") != "pending":
                continue
            batch_root = Path(item.get("prior_batch_root", self.root))
            part = batch_root / "raw" / "http" / f"attempt-{item['attempt_id']:05d}.part"
            actual = part.stat().st_size if part.is_file() else 0
            item.update(status="recovered_unknown", response_bytes=actual, charged_bytes=8_388_608,
                        elapsed_seconds=item.get("request_timeout_seconds", 30.0),
                        charged_wall_seconds=item.get("request_timeout_seconds", 30.0),
                        error=f"pending attempt conservatively settled: {reason}")
            if part.is_file():
                item["partial_path"] = str(part.relative_to(batch_root))
                item["partial_sha256"] = _sha(part.read_bytes())
        self._persist()

    def http_get(self, url: str, **kwargs: Any) -> Any:
        attempt = self.before(url)
        started = time.monotonic()
        try:
            kwargs["timeout"] = min(float(kwargs.get("timeout", 30.0)), attempt["request_timeout_seconds"])
            kwargs["stream"] = True
            kwargs["allow_redirects"] = False
            response = evidence_builder.requests.get(url, **kwargs)
            return _LedgerResponse(response, self, attempt, started)
        except BaseException as exc:
            self.finish(attempt, status="failed", elapsed=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}")
            raise


class _LedgerResponse:
    def __init__(self, response: Any, ledger: RequestLedger, attempt: dict[str, Any], started: float):
        self._response, self._ledger, self._attempt, self._started = response, ledger, attempt, started
        self.status_code = response.status_code
        self.headers = response.headers
        self.url = response.url
        self._body = bytearray()
        self._finished = False
        self._refused = False
        self._raw_dir = ledger.root / "raw" / "http"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._part = self._raw_dir / f"attempt-{attempt['attempt_id']:05d}.part"
        self._raw = ledger.root / attempt["raw_path"]
        length = self.headers.get("Content-Length")
        if length is not None:
            try:
                if int(length) > ledger.limits["chembl_response_bytes_per_request"]:
                    self._refused = True
                    self._fail("declared response exceeds 8 MiB cap",
                               charged_bytes=ledger.limits["chembl_response_bytes_per_request"])
                    response.close()
            except (TypeError, ValueError):
                self._fail("invalid Content-Length", charged_bytes=ledger.limits["chembl_response_bytes_per_request"])

    def raise_for_status(self) -> None:
        try:
            self._response.raise_for_status()
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

    def iter_content(self, chunk_size: int = 65536):
        if self._refused or self._finished:
            raise FormalCorpusError("HTTP response was already refused or finalized")
        try:
            with self._part.open("xb") as stream:
                for chunk in self._response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    if not isinstance(chunk, bytes):
                        raise FormalCorpusError("HTTP response yielded non-bytes")
                    self._body.extend(chunk)
                    if len(self._body) > self._ledger.limits["chembl_response_bytes_per_request"]:
                        raise FormalCorpusError("HTTP response exceeded 8 MiB per-response cap")
                    stream.write(chunk)
                    yield chunk
                stream.flush()
                os.fsync(stream.fileno())
            if self.status_code < 200 or self.status_code >= 300:
                raise FormalCorpusError(f"HTTP status {self.status_code}")
            os.replace(self._part, self._raw)
            digest = _sha(self._raw.read_bytes())
            self._ledger.finish(self._attempt, status="succeeded", elapsed=time.monotonic() - self._started,
                                response_bytes=len(self._body), raw_path=str(self._raw.relative_to(self._ledger.root)),
                                raw_sha256=digest)
            self._finished = True
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

    def _fail(self, reason: str, *, charged_bytes: int | None = None) -> None:
        if not self._finished:
            part_path = self._part if hasattr(self, "_part") else None
            partial_bytes = max(part_path.stat().st_size if part_path is not None and part_path.exists() else 0,
                                len(self._body))
            if charged_bytes is None and partial_bytes == 0:
                declared = self.headers.get("Content-Length")
                try:
                    charged_bytes = min(max(0, int(declared)), self._ledger.limits["chembl_response_bytes_per_request"]) if declared is not None else self._ledger.limits["chembl_response_bytes_per_request"]
                except (TypeError, ValueError):
                    charged_bytes = self._ledger.limits["chembl_response_bytes_per_request"]
            self._ledger.finish(self._attempt, status="failed", elapsed=time.monotonic() - self._started,
                                response_bytes=partial_bytes,
                                partial_path=str(part_path.relative_to(self._ledger.root)) if part_path is not None and part_path.exists() else None,
                                partial_sha256=_sha(part_path.read_bytes()) if part_path is not None and part_path.exists() else None,
                                charged_bytes=charged_bytes,
                                error=reason)
            self._finished = True

    def close(self) -> None:
        if not self._finished:
            self._fail("response stream was abandoned before completion")
        self._response.close()


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FormalCorpusError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FormalCorpusError(f"expected a JSON object: {path}")
    return value, raw


def _verify_selection_receipt(path: Path, config: Mapping[str, Any], config_sha256: str,
                              *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    receipt, raw = _read_object(path.resolve(strict=True))
    digest = _sha(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FormalCorpusError("registered selection receipt SHA mismatch")
    if (receipt.get("schema_version") != 1 or type(receipt.get("schema_version")) is not int
            or receipt.get("selection_status") != "sealed"
            or receipt.get("protocol_version") != config.get("protocol_version")
            or receipt.get("config_sha256") != config_sha256
            or receipt.get("config_semantic_sha256") != _sha(_json_bytes(dict(config)))
            or receipt.get("base_corpus_sha256") != config.get("base_corpus_sha256")
            or receipt.get("decision_sha256") != config.get("budget", {}).get("decision_sha256")
            or receipt.get("global_population_claim") is not False):
        raise FormalCorpusError("selection receipt is incomplete or does not bind the frozen protocol")
    snapshot = receipt.get("ledger_snapshot")
    if not isinstance(snapshot, Mapping):
        raise FormalCorpusError("selection receipt omitted its immutable request-ledger snapshot")
    snapshot_root = Path(snapshot.get("batch_root", "")).resolve(strict=True)
    snapshot_path = (snapshot_root / str(snapshot.get("path", ""))).resolve(strict=True)
    if (snapshot_root not in snapshot_path.parents
            or _sha(snapshot_path.read_bytes()) != snapshot.get("sha256")):
        raise FormalCorpusError("selection request-ledger snapshot is missing, escaped, or changed")
    molecule_ids, target_ids = receipt.get("selected_molecule_ids"), receipt.get("selected_target_chembl_ids")
    if (not isinstance(molecule_ids, list) or not molecule_ids
            or any(not isinstance(value, str) for value in molecule_ids)
            or molecule_ids != sorted(set(molecule_ids))
            or len(molecule_ids) > config["selection_window"]["molecule_limit"]
            or not isinstance(target_ids, list) or not target_ids
            or any(not isinstance(value, str) for value in target_ids)
            or target_ids != sorted(set(target_ids))
            or len(target_ids) > config["selection_window"]["target_limit"]):
        raise FormalCorpusError("selection receipt IDs are empty, duplicated, unsorted, or exceed pre-registered limits")
    for molecule_id in molecule_ids:
        if not isinstance(molecule_id, str) or evidence_builder.CHEMBL_ID.fullmatch(molecule_id) is None:
            raise FormalCorpusError("selection receipt contains an invalid molecule ID")
    for target_id in target_ids:
        if not isinstance(target_id, str) or evidence_builder.CHEMBL_ID.fullmatch(target_id) is None:
            raise FormalCorpusError("selection receipt contains an invalid target ID")
    if receipt.get("forbidden_selection_fields_read") != []:
        raise FormalCorpusError("selection receipt indicates forbidden numeric/class selection")
    rows = receipt.get("selected_relation_evidence")
    forbidden = {"standard_value", "pchembl_value", "class", "split", "support_count"}
    if (not isinstance(rows, list) or any(not isinstance(row, Mapping) or forbidden.intersection(row) for row in rows)):
        raise FormalCorpusError("selection relation evidence contains invalid or forbidden fields")
    source_receipts = receipt.get("raw_source_receipts")
    if not isinstance(source_receipts, list) or not source_receipts:
        raise FormalCorpusError("selection receipt omitted raw source receipts")
    for source in source_receipts:
        if not isinstance(source, Mapping) or source.get("source_kind") != "online_snapshot":
            raise FormalCorpusError("selection raw receipt source kind is invalid")
        batch_root = Path(source.get("batch_root", "")).resolve(strict=True)
        relative = Path(source.get("raw_path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise FormalCorpusError("selection raw receipt path is invalid")
        raw_path = (batch_root / relative).resolve(strict=True)
        if batch_root not in raw_path.parents or _sha(raw_path.read_bytes()) != source.get("raw_sha256"):
            raise FormalCorpusError("selection raw receipt bytes/hash failed verification")
    return receipt, digest


def _safe_new_root(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise FormalCorpusError("output path must be an absolute GUID directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FormalCorpusError("output path must be absolute")
    try:
        artifact_store._assert_no_reparse_components(path)
    except Exception as exc:
        raise FormalCorpusError("output path contains a symlink/reparse point") from exc
    path = path.resolve()
    if path.name != path.name.lower() or len(path.name) != 32:
        raise FormalCorpusError("output directory name must be a lowercase 32-character GUID")
    try:
        import uuid
        if uuid.UUID(hex=path.name).hex != path.name:
            raise ValueError
    except ValueError as exc:
        raise FormalCorpusError("output directory name must be a lowercase 32-character GUID") from exc
    protected = [APL_ROOT / name for name in (
        "data", "framework_output", "offline_data.json", ".model_cache", "model_cache",
        "features", "history", "output_dashboard.html",
    )]
    for item in protected:
        target = item.resolve()
        if path == target or target in path.parents or path in target.parents:
            raise FormalCorpusError("output root intersects a protected project path")
    if path.exists() or path.is_symlink():
        raise FormalCorpusError("output root already exists")
    # Reject any existing symlink/reparse component to avoid escaping the requested root.
    cursor = path
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise FormalCorpusError("output path contains a symlink/reparse point")
        cursor = cursor.parent
    return path


def _verify_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    config, _ = _read_object(path.resolve(strict=True))
    base_path = APL_ROOT / "configs" / "corpus_v1.json"
    base, base_raw = _read_object(base_path)
    if config.get("protocol_version") != "corpus-formal-v1.0" or config.get("base_corpus_sha256") != _sha(base_raw):
        raise FormalCorpusError("formal config protocol/base corpus binding mismatch")
    if config.get("base_corpus_path") != "configs/corpus_v1.json":
        raise FormalCorpusError("formal config must bind the immutable corpus_v1.json")
    bindings = {
        "label_policy": base["label_policy"]["path"],
        "split_policy": base["split_policy"]["path"],
        "support_policy": base["support_policy"]["path"],
    }
    for key, relative in bindings.items():
        target = APL_ROOT / relative
        if not target.is_file() or config.get(f"{key}_sha256") != _sha(target.read_bytes()):
            raise FormalCorpusError(f"formal config {key} hash mismatch")
        if config.get(f"{key}_path") != relative:
            raise FormalCorpusError(f"formal config {key} path mismatch")
    budget = config.get("budget")
    if (not isinstance(budget, Mapping) or budget.get("status") != "measured_pre_registered"
            or budget.get("formal_collection_allowed") is not True):
        raise FormalCorpusError("formal budget decision is not registered")
    decision_path = APL_ROOT / str(budget.get("decision_path", ""))
    if not decision_path.is_file() or _sha(decision_path.read_bytes()) != budget.get("decision_sha256"):
        raise FormalCorpusError("formal budget decision missing or hash mismatch")
    decision, _ = _read_object(decision_path)
    if decision.get("decision_version") != config["protocol_version"] or decision.get("base_corpus_sha256") != config["base_corpus_sha256"]:
        raise FormalCorpusError("budget decision is not bound to the formal corpus protocol")
    caps = decision.get("caps")
    expected = {"molecules": 128, "targets": 64, "chembl_requests": 2048,
                "chembl_activity_pages": 256, "chembl_activity_rows": 20000,
                "chembl_response_bytes_per_request": 8_388_608,
                "chembl_response_bytes_total": 536_870_912,
                "chembl_wall_seconds": 1800, "request_timeout_seconds": 30,
                "uniprot_requests": 128, "uniprot_response_bytes_per_request": 8_388_608,
                "uniprot_response_bytes_total": 134_217_728,
                "builder_molecule_pairs": 8128, "builder_protein_pairs": 2016,
                "builder_alignment_cells": 2_200_000_000, "builder_wall_seconds": 900,
                "auditor_molecule_pairs": 8128, "auditor_protein_pairs": 2016,
                "auditor_alignment_cells": 2_200_000_000, "auditor_wall_seconds": 900,
                "development_neighbor_pairs": 129, "development_neighbor_alignment_cells": 256_000_000,
                "development_neighbor_wall_seconds": 120, "development_molecule_neighbor_pairs": 512,
                "development_molecule_neighbor_wall_seconds": 10,
                "eligibility_cluster_molecule_pairs": 8128, "eligibility_cluster_protein_pairs": 2016,
                "eligibility_cluster_alignment_cells": 2_200_000_000, "eligibility_cluster_wall_seconds": 900,
                "publisher_audit_passes": 3, "publisher_audit_alignment_cells": 2_200_000_000,
                "publisher_audit_wall_seconds": 900,
                "pipeline_total_alignment_cells": 6_856_000_000, "pipeline_total_wall_seconds": 2820}
    if caps != expected:
        raise FormalCorpusError("formal budget caps differ from the signed decision")
    expected_window = {
        "resource": "ChEMBL Activity REST JSON",
        "filters": {"standard_type": "Ki", "target_organism": "Homo sapiens"},
        "projection": ["activity_id", "molecule_chembl_id", "target_chembl_id", "assay_chembl_id",
                       "document_chembl_id", "standard_type", "standard_relation", "target_organism"],
        "forbid_fields": ["standard_value", "pchembl_value", "class", "split", "support_count"],
        "target_tax_id_required_on_activity_row": False,
        "taxonomy_verification": "target and component source metadata plus complete UniProt evidence",
        "api_order_by": "molecule_chembl_id", "stable_local_tiebreak": ["molecule_chembl_id", "activity_id"],
        "page_size": 100, "max_pages": 10, "max_projection_rows": 1000,
        "max_discovery_chembl_requests": 512, "target_limit": 64, "molecule_limit": 128,
        "stop_rule": "select first stable eligible targets/molecules from the fixed contiguous page window; if fewer than one eligible entity of either kind, stop with selection_window_exhausted; never expand window based on labels or support",
        "window_semantics": "contiguous bounded source window; page_meta next and exact page chain recorded; never represented as global ChEMBL cohort",
        "capture_reserve": {"activity_pages": 246, "activity_rows": 19000, "chembl_requests": 1536,
                            "chembl_response_bytes": 402653184},
        "shared_caps": "discovery and capture share total requests/pages/rows/response-bytes/wall caps; capture cannot reset discovery usage",
    }
    if config.get("selection_window") != expected_window:
        raise FormalCorpusError("formal selection window differs from the pre-registered fixed rule")
    expected_luteolin_lookup = {
        "resource": "ChEMBL Molecule REST JSON", "filter": {"pref_name__iexact": "luteolin"},
        "order_by": "molecule_chembl_id", "local_stable_order": "ascending molecule_chembl_id",
        "only": ["molecule_chembl_id", "pref_name", "molecule_structures", "molecule_type"],
        "page_size": 100, "max_pages": 10, "max_rows": 1000,
        "completion": "follow and hash every next link until next=null and accumulated rows equal stable page_meta.total_count; if cap reached first, incomplete and no unseen-case ID is selected",
        "ambiguity": "after complete exact-name result set, select lowest stable ChEMBL ID and preserve every exact-name candidate ID",
    }
    if config.get("luteolin_case", {}).get("source_lookup_rule") != expected_luteolin_lookup:
        raise FormalCorpusError("luteolin identity lookup differs from the locked exact-name window rule")
    development = config.get("development_exclusion")
    if (not isinstance(development, Mapping)
            or development.get("molecule_ids") != ["CHEMBL1", "CHEMBL10"]
            or development.get("protein_accessions") != ["P08183", "P35916"]
            or development.get("exclude_exact_entities") is not True
            or development.get("exclude_Murcko_scaffolds_and_molecule_neighbor_clusters") is not True
            or development.get("exclude_protein_neighbor_clusters") is not True
            or development.get("source_bundle_path") != "../P1_02_02_Artifacts_c0de81f2a51849b089548f420f211177/pilot-v1.2-root-accepted/entities.json"
            or development.get("cluster_threshold_bound_to_split_policy_sha256") != config.get("split_policy_sha256")
            or development.get("protein_neighbor_rule") != "exclude any candidate protein adjacent to either development protein under the exact double-cold split policy thresholds and full-sequence global alignment"
            or development.get("protein_thresholds") != {"identity": 0.3, "coverage": 0.8,
                                                          "semantics": "both complete-sequence coverages >=0.8; identity >=0.3"}):
        raise FormalCorpusError("formal development-neighbor exclusions differ from the locked rule")
    if config.get("sequence_policy") != {
        "esm_max_tokens": MAX_ESM_TOKENS,
        "reserved_special_tokens": SPECIAL_TOKEN_RESERVE,
        "maximum_supported_full_sequence_residues": MAX_SUPPORTED_RESIDUES,
        "tokenizer_validation": "not_claimed; conservative max_tokens minus BOS/EOS reserve",
        "retention": "keep complete source sequence and receipts; record unsupported_length; never truncate",
    }:
        raise FormalCorpusError("formal sequence applicability policy mismatch")
    return config, decision


def _require_formal_online_snapshot(bundle: Mapping[str, Any]) -> None:
    if bundle.get("source_kind") != "online_snapshot":
        raise FormalCorpusError("formal preflight requires an online_snapshot entity bundle")
    for collection in ("molecules", "proteins", "activities", "source_provenance"):
        rows = bundle.get(collection)
        if not isinstance(rows, list):
            raise FormalCorpusError(f"formal source bundle omitted {collection}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("source_kind") != "online_snapshot":
                raise FormalCorpusError(f"formal source bundle contains non-online_snapshot {collection}[{index}]")
    if not bundle["molecules"] or not bundle["proteins"] or not bundle["activities"] or not bundle["source_provenance"]:
        raise FormalCorpusError("formal source bundle must contain molecules, proteins, activities, and raw receipts")


def _sequence_applicability(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in bundle.get("proteins", []):
        if not isinstance(row, Mapping):
            continue
        accession, sequence = row.get("accession"), row.get("full_sequence")
        label = f"{row.get('taxonomy_id')}|{accession}"
        length = len(sequence) if isinstance(sequence, str) else None
        item = {"entity_id": label, "accession": accession, "sequence_length": length,
                "sequence_sha256": row.get("sequence_sha256"), "source_kind": row.get("source_kind")}
        if row.get("sequence_status") != "verified" or row.get("identity_status") != "verified":
            excluded.append({**item, "reason": row.get("sequence_status_reason") or "sequence_or_identity_unverified"})
        elif length is None or length == 0:
            excluded.append({**item, "reason": "full_sequence_missing"})
        elif length > MAX_SUPPORTED_RESIDUES:
            excluded.append({**item, "reason": "unsupported_length_full_sequence_retained"})
        else:
            supported.append(item)
    return supported, excluded


def _development_bundle(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    rule = config["development_exclusion"]
    path = (APL_ROOT / str(rule["source_bundle_path"])).resolve(strict=True)
    bundle = evidence_builder.load_entity_bundle(path, verify_sources=True)
    _require_formal_online_snapshot(bundle)
    if bundle.get("content_sha256") != rule.get("source_bundle_content_sha256"):
        raise FormalCorpusError("development exclusion bundle changed from its pre-registered content hash")
    molecule_ids = sorted(item["molecule_chembl_id"] for item in bundle.get("molecules", [])
                          if isinstance(item, Mapping) and item.get("molecule_chembl_id"))
    protein_ids = sorted(item["accession"] for item in bundle.get("proteins", [])
                         if isinstance(item, Mapping) and item.get("accession"))
    if molecule_ids != rule.get("molecule_ids") or protein_ids != rule.get("protein_accessions"):
        raise FormalCorpusError("development molecule/protein identities differ from the locked exclusion rule")
    return bundle, path


def _molecule_profile(smiles: str, generator: Any, Chem: Any, MurckoScaffold: Any) -> tuple[Any, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise FormalCorpusError("source molecule canonical SMILES cannot be parsed")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""
    return generator.GetFingerprint(molecule), scaffold_smiles


def _molecule_exclusion_index(bundle: Mapping[str, Any], luteolin: Mapping[str, Any]) -> tuple[Any, Any, list[dict[str, Any]]]:
    Chem, DataStructs, fingerprint_generator, MurckoScaffold = dataset_builder._require_rdkit()
    generator = fingerprint_generator.GetMorganGenerator(radius=2, fpSize=2048)
    references = []
    for row in bundle["molecules"]:
        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            raise FormalCorpusError("development molecule lacks verified structure for neighbor exclusion")
        fingerprint, scaffold = _molecule_profile(smiles, generator, Chem, MurckoScaffold)
        references.append({"entity_id": row["molecule_chembl_id"], "fingerprint": fingerprint,
                           "scaffold": scaffold, "role": "development"})
    luteolin_id = luteolin.get("molecule_chembl_id")
    luteolin_structures = luteolin.get("molecule_structures")
    smiles = luteolin_structures.get("canonical_smiles") if isinstance(luteolin_structures, Mapping) else None
    if not isinstance(luteolin_id, str) or not isinstance(smiles, str) or not smiles:
        raise FormalCorpusError("source-resolved luteolin identity lacks a canonical structure")
    fingerprint, scaffold = _molecule_profile(smiles, generator, Chem, MurckoScaffold)
    references.append({"entity_id": luteolin_id, "fingerprint": fingerprint, "scaffold": scaffold, "role": "unseen_case"})
    return (DataStructs, (generator, Chem, MurckoScaffold), references)


def _exclude_molecule(molecule: Mapping[str, Any], references: Sequence[Mapping[str, Any]],
                      profile_tools: tuple[Any, Any, Any]) -> dict[str, Any] | None:
    structures = molecule.get("molecule_structures")
    smiles = structures.get("canonical_smiles") if isinstance(structures, Mapping) else None
    if not isinstance(smiles, str) or not smiles:
        return {"reason": "canonical_structure_missing"}
    DataStructs, (generator, Chem, MurckoScaffold), _ = profile_tools
    try:
        fingerprint, scaffold = _molecule_profile(smiles, generator, Chem, MurckoScaffold)
    except FormalCorpusError:
        return {"reason": "canonical_structure_invalid"}
    molecule_id = molecule.get("molecule_chembl_id")
    for reference in references:
        if molecule_id == reference["entity_id"]:
            return {"reason": f"exact_{reference['role']}_molecule", "neighbor_id": reference["entity_id"]}
        if scaffold and reference["scaffold"] and scaffold == reference["scaffold"]:
            return {"reason": f"shared_{reference['role']}_murcko_scaffold", "neighbor_id": reference["entity_id"]}
        similarity = float(DataStructs.TanimotoSimilarity(fingerprint, reference["fingerprint"]))
        if similarity >= 0.8:
            return {"reason": f"{reference['role']}_molecule_similarity_ge_0.8",
                    "neighbor_id": reference["entity_id"], "tanimoto": similarity}
    return None


def _raw_attempt_for_url(ledger: RequestLedger, url: str) -> dict[str, Any]:
    candidates = [item for item in ledger.attempts if item.get("url") == url and item.get("service") == "chembl"]
    if not candidates or candidates[-1].get("status") not in {"succeeded", "recovered_success"}:
        raise FormalCorpusError("source query has no successful persisted raw receipt")
    return candidates[-1]


def _fetch_json(ledger: RequestLedger, url: str) -> tuple[dict[str, Any], bytes, str]:
    payload, body, digest = evidence_builder._bounded_get_json(
        url, timeout=30, max_response_bytes=8_388_608, http_get=ledger.http_get,
    )
    attempt = _raw_attempt_for_url(ledger, url)
    if attempt.get("raw_sha256") != digest:
        raise FormalCorpusError("persisted raw response hash disagrees with parsed source bytes")
    return payload, body, digest


def _validate_activity_window_url(url: str, *, expected_offset: int, projection: Sequence[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk" or parsed.path != "/chembl/api/data/activity.json":
        raise FormalCorpusError("ChEMBL Activity next link escaped the registered endpoint")
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected = {"standard_type": ["Ki"], "target_organism": ["Homo sapiens"],
                "order_by": ["molecule_chembl_id"], "limit": ["100"],
                "offset": [str(expected_offset)], "only": [",".join(projection)]}
    if query != expected:
        raise FormalCorpusError("Activity next link changed the fixed filter, projection, order, or page boundary")


def _validate_activity_projection_page(payload: Mapping[str, Any], *, expected_offset: int,
                                       projection: Sequence[str], expected_total: int | None = None
                                       ) -> tuple[list[dict[str, Any]], str | None, int]:
    try:
        normalized, rows, next_url = evidence_builder._validate_activity_page(payload, expected_offset)
    except Exception as exc:
        raise FormalCorpusError(f"Activity page/pagination contract failed: {exc}") from exc
    meta = normalized["page_meta"]
    if type(meta.get("limit")) is not int or meta["limit"] != 100:
        raise FormalCorpusError("Activity source page size changed from registered 100")
    total = meta.get("total_count")
    if type(total) is not int or total < 0:
        raise FormalCorpusError("Activity page total_count must be a strict non-negative integer")
    if expected_total is not None and total != expected_total:
        raise FormalCorpusError("Activity page total_count changed within the registered source window")
    for row in rows:
        if row.get("standard_type") != "Ki" or row.get("target_organism") != "Homo sapiens":
            raise FormalCorpusError("Activity row changed the registered Ki/human query scope")
        if any(field in row for field in ("standard_value", "pchembl_value", "class", "split", "support_count")):
            raise FormalCorpusError("Activity projection unexpectedly contains forbidden result/value fields")
    return [dict(row) for row in rows], next_url, total


def _validate_activity_window_end(total_count: int, observed_rows: int, next_url: str | None) -> None:
    if next_url is None and observed_rows != total_count:
        raise FormalCorpusError("Activity pagination ended before total_count; source window is incomplete")
    if next_url is not None and observed_rows >= total_count:
        raise FormalCorpusError("Activity next link contradicts total_count; source window is malformed")


def _discover_window(ledger: RequestLedger, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window = config["selection_window"]
    projection = window["projection"]
    status_url = f"{evidence_builder.CHEMBL_API_ROOT}/status.json"
    status, _, status_hash = _fetch_json(ledger, status_url)
    snapshot = status.get("chembl_db_version") or status.get("database_version")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise FormalCorpusError("ChEMBL release boundary missing from status response")
    params = {"standard_type": "Ki", "target_organism": "Homo sapiens",
              "order_by": "molecule_chembl_id", "limit": window["page_size"], "offset": 0,
              "only": ",".join(projection)}
    next_url: str | None = f"{evidence_builder.CHEMBL_API_ROOT}/activity.json?{urlencode(params)}"
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expected_offset = 0
    total_count = None
    while next_url is not None and len(pages) < window["max_pages"]:
        _validate_activity_window_url(next_url, expected_offset=expected_offset, projection=projection)
        payload, body, digest = _fetch_json(ledger, next_url)
        page_rows, advertised_next, total_count = _validate_activity_projection_page(
            payload, expected_offset=expected_offset, projection=projection, expected_total=total_count,
        )
        page_meta = payload["page_meta"]
        attempt = _raw_attempt_for_url(ledger, next_url)
        pages.append({"offset": expected_offset, "row_count": len(page_rows), "url": next_url,
                      "raw_path": attempt["raw_path"], "raw_sha256": digest,
                      "page_meta": page_meta})
        rows.extend(dict(row) for row in page_rows)
        expected_offset += len(page_rows)
        _validate_activity_window_end(total_count, expected_offset, advertised_next)
        if advertised_next is None:
            next_url = None
        else:
            next_url = evidence_builder._validate_chembl_next(advertised_next, next_url)
            _validate_activity_window_url(next_url, expected_offset=expected_offset, projection=projection)
    coverage = "complete_window" if next_url is None else "bounded_window_truncated_at_registered_page_cap"
    rows.sort(key=lambda row: (str(row.get("molecule_chembl_id", "")), str(row.get("activity_id", ""))))
    return rows, {"source_snapshot": snapshot, "status_raw_sha256": status_hash,
                  "status_receipt": _raw_attempt_for_url(ledger, status_url),
                  "query_window": {"first_offset": 0, "last_offset": pages[-1]["offset"] if pages else None,
                                   "pages": pages, "row_count": len(rows), "page_meta_total_count": total_count,
                                   "coverage": coverage,
                                   "complete_global_source": False,
                                   "source_receipts": [{"raw_path": row["raw_path"], "raw_sha256": row["raw_sha256"]} for row in pages]},
                  "stable_order": ["molecule_chembl_id", "activity_id"],
                  "projection_fields": list(projection),
                  "selection_request_count": sum(item.get("phase") == "discovery" and item.get("service") == "chembl" for item in ledger.attempts)}


def _validate_molecule_lookup_url(url: str, *, expected_offset: int, lookup: Mapping[str, Any]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk" or parsed.path != "/chembl/api/data/molecule.json":
        raise FormalCorpusError("luteolin lookup next link escaped the registered molecule endpoint")
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected = {"pref_name__iexact": ["luteolin"], "order_by": ["molecule_chembl_id"],
                "limit": [str(lookup["page_size"])], "offset": [str(expected_offset)],
                "only": [",".join(lookup["only"])]}
    if query != expected:
        raise FormalCorpusError("luteolin lookup next link changed the exact-name scope, projection, or page boundary")


def _validate_molecule_lookup_next(advertised: Any, *, expected_offset: int,
                                   lookup: Mapping[str, Any]) -> str:
    if not isinstance(advertised, str) or not advertised.strip() or advertised.startswith("//"):
        raise FormalCorpusError("luteolin molecule pagination next link is invalid")
    absolute = urljoin("https://www.ebi.ac.uk", advertised)
    _validate_molecule_lookup_url(absolute, expected_offset=expected_offset, lookup=lookup)
    return absolute


def _lookup_luteolin(ledger: RequestLedger, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lookup = config["luteolin_case"]["source_lookup_rule"]
    params = {**lookup["filter"], "order_by": lookup["order_by"], "limit": lookup["page_size"],
              "offset": 0, "only": ",".join(lookup["only"])}
    next_url: str | None = f"{evidence_builder.CHEMBL_API_ROOT}/molecule.json?{urlencode(params)}"
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    offset = 0
    total_count = None
    while next_url is not None and len(pages) < lookup["max_pages"]:
        _validate_molecule_lookup_url(next_url, expected_offset=offset, lookup=lookup)
        payload, _, digest = _fetch_json(ledger, next_url)
        entries, meta = payload.get("molecules"), payload.get("page_meta")
        if not isinstance(entries, list) or not isinstance(meta, Mapping):
            raise FormalCorpusError("exact-name molecule lookup omitted records or page_meta")
        if type(meta.get("offset")) is not int or meta["offset"] != offset or type(meta.get("limit")) is not int or meta["limit"] != lookup["page_size"]:
            raise FormalCorpusError("exact-name molecule pagination offset/limit mismatch")
        current_total = meta.get("total_count")
        if type(current_total) is not int or current_total < 0 or (total_count is not None and current_total != total_count):
            raise FormalCorpusError("exact-name molecule total_count is invalid or changed between pages")
        total_count = current_total
        for item in entries:
            if not isinstance(item, Mapping) or str(item.get("pref_name", "")).strip().casefold() != "luteolin":
                raise FormalCorpusError("exact-name molecule query returned an out-of-scope preferred name")
            molecule_id = item.get("molecule_chembl_id")
            if not isinstance(molecule_id, str) or evidence_builder.CHEMBL_ID.fullmatch(molecule_id) is None:
                raise FormalCorpusError("exact-name molecule query returned an invalid ChEMBL ID")
            rows.append(dict(item))
        offset += len(entries)
        attempt = _raw_attempt_for_url(ledger, next_url)
        pages.append({"offset": meta["offset"], "row_count": len(entries), "total_count": current_total,
                      "batch_root": str(ledger.root.resolve()),
                      "source_url": next_url, "raw_path": attempt["raw_path"], "raw_sha256": digest})
        advertised = meta.get("next")
        if advertised is not None and not isinstance(advertised, str):
            raise FormalCorpusError("molecule page_meta.next is malformed")
        next_url = (_validate_molecule_lookup_next(advertised, expected_offset=offset, lookup=lookup)
                    if advertised is not None else None)
        if next_url is not None:
            _validate_molecule_lookup_url(next_url, expected_offset=offset, lookup=lookup)
        elif offset != total_count:
            raise FormalCorpusError("exact-name molecule lookup ended before its total_count")
        if offset > lookup["max_rows"]:
            raise FormalCorpusError("exact-name molecule lookup exceeded its registered row cap")
    if next_url is not None:
        raise FormalCorpusError("exact-name luteolin lookup window incomplete at registered page cap")
    if total_count != len(rows) or not rows:
        raise FormalCorpusError("exact-name luteolin lookup did not produce a complete non-empty result set")
    rows.sort(key=lambda item: item["molecule_chembl_id"])
    selected = rows[0]
    structures = selected.get("molecule_structures")
    if not isinstance(structures, Mapping) or not structures.get("canonical_smiles"):
        raise FormalCorpusError("source-resolved luteolin has no canonical structure for holdout exclusion")
    detail_url = f"{evidence_builder.CHEMBL_API_ROOT}/molecule/{selected['molecule_chembl_id']}.json"
    detail, _, detail_sha = _fetch_json(ledger, detail_url)
    detail_structures = detail.get("molecule_structures")
    if (detail.get("molecule_chembl_id") != selected["molecule_chembl_id"]
            or str(detail.get("pref_name", "")).strip().casefold() != "luteolin"
            or not isinstance(detail_structures, Mapping)
            or detail_structures.get("canonical_smiles") != structures.get("canonical_smiles")):
        raise FormalCorpusError("ChEMBL Molecule detail identity/structure disagrees with the exact-name lookup")
    detail_attempt = _raw_attempt_for_url(ledger, detail_url)
    return selected, {"lookup_complete": True, "total_count": total_count,
                      "exact_name_candidate_ids": [row["molecule_chembl_id"] for row in rows],
                      "selected_chembl_id": selected["molecule_chembl_id"],
                      "pages": pages,
                      "detail_receipt": {"batch_root": str(ledger.root.resolve()),
                                         "source_url": detail_url, "raw_path": detail_attempt["raw_path"],
                                         "raw_sha256": detail_sha, "native_molecule_chembl_id": detail.get("molecule_chembl_id")},
                      "complete_source_receipts": pages + [{"source_url": detail_url,
                                          "raw_path": detail_attempt["raw_path"], "raw_sha256": detail_sha}],
                      "selection_rule": lookup}


def _select_from_window(ledger: RequestLedger, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                        development_bundle: Mapping[str, Any], luteolin: Mapping[str, Any],
                        luteolin_receipt: Mapping[str, Any]) -> dict[str, Any]:
    allowed_relations = set(config["scope"]["relation_allowlist"])
    projected = []
    for row in rows:
        molecule_id, target_id, assay_id, activity_id = (row.get("molecule_chembl_id"), row.get("target_chembl_id"),
                                                         row.get("assay_chembl_id"), row.get("activity_id"))
        if (row.get("standard_type") == "Ki" and row.get("target_organism") == "Homo sapiens"
                and row.get("standard_relation") in allowed_relations
                and isinstance(activity_id, (str, int))
                and isinstance(molecule_id, str) and evidence_builder.CHEMBL_ID.fullmatch(molecule_id)
                and isinstance(target_id, str) and evidence_builder.CHEMBL_ID.fullmatch(target_id)
                and isinstance(assay_id, str) and evidence_builder.CHEMBL_ID.fullmatch(assay_id)
                and isinstance(row.get("document_chembl_id"), str)
                and evidence_builder.CHEMBL_ID.fullmatch(row["document_chembl_id"])):
            projected.append(dict(row))
    projected.sort(key=lambda row: (row["molecule_chembl_id"], str(row["activity_id"])))
    rows_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in projected:
        rows_by_target.setdefault(row["target_chembl_id"], []).append(row)
    targets: list[str] = []
    qualified_pairs: set[tuple[str, str]] = set()
    target_evidence: list[dict[str, Any]] = []
    assay_cache: dict[str, dict[str, Any] | None] = {}
    for target_id in sorted(rows_by_target):
        target_url = f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json"
        target, _, target_sha = _fetch_json(ledger, target_url)
        if (target.get("target_chembl_id") != target_id or target.get("target_type") != "SINGLE PROTEIN"
                or target.get("organism") != "Homo sapiens" or target.get("tax_id") not in (9606, "9606")):
            continue
        target_attempt = _raw_attempt_for_url(ledger, target_url)
        resolved = evidence_builder.resolve_target_components(
            target, source_kind="online_snapshot", source_path=target_attempt["raw_path"],
            source_sha256=target_sha, source_url=target_url,
        )
        if (len(resolved) != 1 or resolved[0].get("resolution_status") != "verified"
                or resolved[0].get("taxonomy_id") != 9606):
            continue
        for assay_id in sorted({row["assay_chembl_id"] for row in rows_by_target[target_id]}):
            if assay_id not in assay_cache:
                assay_url = f"{evidence_builder.CHEMBL_API_ROOT}/assay/{assay_id}.json"
                assay, _, _ = _fetch_json(ledger, assay_url)
                assay_cache[assay_id] = assay if isinstance(assay, dict) else None
            assay = assay_cache[assay_id]
            confidence = assay.get("confidence_score") if isinstance(assay, Mapping) else None
            if isinstance(confidence, str) and confidence.isdigit():
                confidence = int(confidence)
            if (isinstance(assay, Mapping) and assay.get("assay_chembl_id") == assay_id
                    and assay.get("target_chembl_id") == target_id
                    and type(confidence) is int and confidence >= 8):
                qualified_pairs.add((target_id, assay_id))
        if any(target_id == pair[0] for pair in qualified_pairs):
            targets.append(target_id)
            target_evidence.append({"target_chembl_id": target_id,
                                    "accession": resolved[0].get("accession"),
                                    "taxonomy_id": resolved[0].get("taxonomy_id"),
                                    "target_raw_sha256": target_sha,
                                    "target_receipt": target_attempt})
        if len(targets) >= config["selection_window"]["target_limit"]:
            break

    profiles = _molecule_exclusion_index(development_bundle, luteolin)
    selected_ids: list[str] = []
    excluded_molecules: list[dict[str, Any]] = []
    molecule_receipts: list[dict[str, Any]] = []
    selected_rows = []
    molecule_ids = sorted({row["molecule_chembl_id"] for row in projected
                           if (row["target_chembl_id"], row["assay_chembl_id"]) in qualified_pairs})
    for molecule_id in molecule_ids:
        molecule_url = f"{evidence_builder.CHEMBL_API_ROOT}/molecule/{molecule_id}.json"
        molecule, _, molecule_sha = _fetch_json(ledger, molecule_url)
        if molecule.get("molecule_chembl_id") != molecule_id:
            excluded_molecules.append({"molecule_chembl_id": molecule_id, "reason": "molecule_identity_mismatch"})
            continue
        structures = molecule.get("molecule_structures")
        if not isinstance(structures, Mapping) or not (structures.get("canonical_smiles") or structures.get("standard_inchi")):
            excluded_molecules.append({"molecule_chembl_id": molecule_id, "reason": "source_structure_missing"})
            continue
        exclusion = _exclude_molecule(molecule, profiles[2], profiles)
        attempt = _raw_attempt_for_url(ledger, molecule_url)
        if exclusion is not None:
            excluded_molecules.append({"molecule_chembl_id": molecule_id, **exclusion,
                                       "source_raw_sha256": molecule_sha, "source_receipt": attempt})
            continue
        selected_ids.append(molecule_id)
        molecule_receipts.append({"molecule_chembl_id": molecule_id, "raw_sha256": molecule_sha,
                                  "source_receipt": attempt})
        for row in projected:
            if (row["molecule_chembl_id"] == molecule_id and row["target_chembl_id"] in targets
                    and (row["target_chembl_id"], row["assay_chembl_id"]) in qualified_pairs):
                selected_rows.append({field: row.get(field) for field in (
                    "activity_id", "molecule_chembl_id", "target_chembl_id", "assay_chembl_id",
                    "document_chembl_id", "standard_type", "standard_relation")})
        if len(selected_ids) >= config["selection_window"]["molecule_limit"]:
            break
    selected_targets = sorted({row["target_chembl_id"] for row in selected_rows})
    receipt = {
        "schema_version": 1, "purpose": "formal_bounded_candidate_selection",
        "protocol_version": config["protocol_version"], "config_semantic_sha256": _sha(_json_bytes(dict(config))),
        "base_corpus_sha256": config["base_corpus_sha256"], "decision_sha256": config["budget"]["decision_sha256"],
        "source_snapshot": None, "selection_status": "sealed" if selected_ids and selected_targets else "selection_window_exhausted",
        "global_population_claim": False, "selection_window": None,
        "selection_rule": config["selection_window"],
        "luteolin_case": dict(luteolin_receipt),
        "development_bundle_content_sha256": config["development_exclusion"]["source_bundle_content_sha256"],
        "selected_molecule_ids": selected_ids, "selected_target_chembl_ids": selected_targets,
        "selected_relation_evidence": sorted(selected_rows, key=lambda row: (row["molecule_chembl_id"], str(row["activity_id"]))),
        "target_quality_evidence": target_evidence, "molecule_metadata_receipts": molecule_receipts,
        "excluded_molecules": excluded_molecules,
        "forbidden_selection_fields_read": [],
    }
    return receipt


def _cluster_member_map(result: Any) -> dict[str, str]:
    return {member: cluster.cluster_id for cluster in result.clusters for member in cluster.member_ids}


def _as_cluster_edge(evidence: Any, formal_membership: Mapping[str, str],
                     development_nodes: Mapping[str, str]) -> dict[str, Any] | None:
    left_formal, right_formal = formal_membership.get(evidence.entity_a), formal_membership.get(evidence.entity_b)
    left_dev, right_dev = development_nodes.get(evidence.entity_a), development_nodes.get(evidence.entity_b)
    if left_formal and right_dev:
        left, right = left_formal, right_dev
    elif right_formal and left_dev:
        left, right = right_formal, left_dev
    elif left_dev and right_dev:
        left, right = left_dev, right_dev
    else:
        return None
    return {"entity_a": left, "entity_b": right, "reason": evidence.reason,
            "score": evidence.score, "details": dict(evidence.details)}


def _development_touched_cluster_ids(formal_cluster_ids: Sequence[str], development_nodes: Sequence[str],
                                     edges: Sequence[Mapping[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    identifiers = sorted(set(formal_cluster_ids) | set(development_nodes))
    components = dataset_builder.connected_clusters(identifiers, edges, prefix="formal-development-union-v1")
    touched: set[str] = set()
    receipts = []
    dev_set = set(development_nodes)
    for cluster in components:
        members = set(cluster.member_ids)
        dev_members = sorted(members & dev_set)
        formal_members = sorted(members - dev_set)
        if dev_members:
            touched.update(formal_members)
            receipts.append({"component_id": cluster.cluster_id, "development_members": dev_members,
                             "formal_cluster_ids": formal_members})
    return touched, receipts


def _runtime_split_policy(root: Path, config: Mapping[str, Any], decision: Mapping[str, Any]) -> Path:
    source_path = APL_ROOT / config["split_policy_path"]
    source, source_bytes = _read_object(source_path)
    if _sha(source_bytes) != config["split_policy_sha256"]:
        raise FormalCorpusError("split policy changed after formal protocol registration")
    policy = json.loads(json.dumps(source))
    caps = decision["caps"]
    policy["budgets"]["molecule_pairs"] = caps["builder_molecule_pairs"]
    policy["budgets"]["protein_pairs"] = caps["builder_protein_pairs"]
    policy["budgets"]["alignment_cells"] = caps["builder_alignment_cells"]
    for key, value in source.items():
        if key != "budgets" and policy[key] != value:
            raise FormalCorpusError("runtime split policy changed a scientific rule")
    directory = root / "policies"
    directory.mkdir(exist_ok=False)
    target = directory / "split_policy_formal_v1.json"
    target.write_bytes(_json_bytes(policy))
    support_path = APL_ROOT / config["support_policy_path"]
    support_bytes = support_path.read_bytes()
    if _sha(support_bytes) != config["support_policy_sha256"]:
        raise FormalCorpusError("support policy changed after formal protocol registration")
    (directory / "support_v1.json").write_bytes(support_bytes)
    return target


def _write_json_new(path: Path, value: Mapping[str, Any]) -> bytes:
    if path.exists() or path.is_symlink():
        raise FormalCorpusError(f"refusing to overwrite formal run artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(dict(value))
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _selection_raw_receipts(ledger: RequestLedger) -> list[dict[str, Any]]:
    result = []
    for item in ledger.attempts:
        if (item.get("phase") not in {"discovery"} or item.get("service") != "chembl"
                or item.get("status") not in {"succeeded", "recovered_success"}
                or not isinstance(item.get("raw_path"), str)):
            continue
        root = Path(item.get("prior_batch_root", ledger.root)).resolve()
        raw_path = (root / item["raw_path"]).resolve(strict=True)
        if root not in raw_path.parents or _sha(raw_path.read_bytes()) != item.get("raw_sha256"):
            raise FormalCorpusError("formal selection raw receipt failed batch-root/SHA verification")
        result.append({"attempt_id": item["attempt_id"], "service": "chembl", "url": item["url"],
                       "batch_root": str(root), "raw_path": item["raw_path"],
                       "raw_sha256": item["raw_sha256"], "source_kind": "online_snapshot"})
    return result


def _seal_selection(root: Path, ledger: RequestLedger, config: Mapping[str, Any], config_sha256: str,
                    selection: Mapping[str, Any], window: Mapping[str, Any], luteolin: Mapping[str, Any],
                    registered_utc: str) -> tuple[dict[str, Any], str]:
    ledger.reconcile_activity_pages()
    ledger_snapshot_path = root / "selection_request_ledger.json"
    ledger_snapshot = ledger.path.read_bytes()
    if ledger_snapshot_path.exists():
        raise FormalCorpusError("selection request-ledger snapshot already exists")
    ledger_snapshot_path.write_bytes(ledger_snapshot)
    receipt = dict(selection)
    receipt.update({
        "config_sha256": config_sha256,
        "source_snapshot": window["source_snapshot"],
        "selection_window": window["query_window"],
        "selection_status": selection["selection_status"],
        "luteolin_case": dict(luteolin),
        "registered_before_source_queries_utc": registered_utc,
        "raw_source_receipts": _selection_raw_receipts(ledger),
        "ledger_snapshot": {"batch_root": str(root.resolve()),
                             "path": ledger_snapshot_path.name,
                             "sha256": _sha(ledger_snapshot)},
        "selection_budget_usage": ledger.summary(),
    })
    raw = _write_json_new(root / "selection_receipt.json", receipt)
    checked, digest = _verify_selection_receipt(root / "selection_receipt.json", config, config_sha256,
                                                expected_sha256=_sha(raw))
    return checked, digest


def _copy_sealed_selection(prior_root: Path, root: Path, config: Mapping[str, Any], config_sha256: str,
                           expected_sha256: str) -> tuple[dict[str, Any], str]:
    prior_path = prior_root / "selection_receipt.json"
    receipt, digest = _verify_selection_receipt(prior_path, config, config_sha256,
                                                expected_sha256=expected_sha256)
    raw = prior_path.read_bytes()
    target = root / "selection_receipt.json"
    if target.exists():
        raise FormalCorpusError("resume target selection receipt already exists")
    target.write_bytes(raw)
    return receipt, digest


def _target_component_identity(target: Mapping[str, Any], accession: str) -> tuple[Any, ...]:
    resolutions = evidence_builder.resolve_target_components(target, source_kind="online_snapshot")
    matches = [row for row in resolutions if row.get("accession") == accession]
    if len(matches) != 1 or matches[0].get("resolution_status") != "verified":
        raise FormalCorpusError("native ChEMBL target does not resolve to exactly one requested component accession")
    row = matches[0]
    return (row.get("target_chembl_id"), row.get("target_type"), row.get("normalized_organism"),
            row.get("target_taxonomy_id"), row.get("component_index"), row.get("component"),
            row.get("accession"), row.get("taxonomy_id"), row.get("identity_key"),
            row.get("resolution_status"))


def _validate_complete_capture(bundle: Mapping[str, Any], result: Mapping[str, Any],
                               selection: Mapping[str, Any], bundle_root: Path | None = None) -> dict[str, Any]:
    _require_formal_online_snapshot(bundle)
    expected_molecules = set(selection["selected_molecule_ids"])
    expected_targets = set(selection["selected_target_chembl_ids"])
    molecule_ids = {row.get("molecule_chembl_id") for row in bundle["molecules"]}
    if molecule_ids != expected_molecules:
        raise FormalCorpusError("captured source molecule IDs differ from the sealed selection receipt")
    for row in bundle["activities"]:
        if (row.get("molecule_chembl_id") not in expected_molecules
                or row.get("target_chembl_id") not in expected_targets
                or row.get("standard_type") != "Ki"
                or row.get("target_organism") != "Homo sapiens"):
            raise FormalCorpusError("captured activity row escaped the sealed molecule/target/Ki/human scope")
    sealed_rows = selection.get("selected_relation_evidence")
    if not isinstance(sealed_rows, list) or not sealed_rows:
        raise FormalCorpusError("selection receipt omitted the exact eligible Ki activity rows")
    row_fields = ("activity_id", "molecule_chembl_id", "target_chembl_id", "assay_chembl_id",
                  "document_chembl_id", "standard_type", "standard_relation")
    def activity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field)) for field in row_fields)
    sealed_counter = Counter(activity_key(row) for row in sealed_rows if isinstance(row, Mapping))
    captured_counter = Counter(activity_key(row) for row in bundle["activities"] if isinstance(row, Mapping))
    if len(sealed_counter) == 0 or any(captured_counter[key] < count for key, count in sealed_counter.items()):
        raise FormalCorpusError("full capture omitted or replaced one or more sealed eligible Activity ID/assay/document/relation rows")
    expected_accessions = {row["target_chembl_id"]: row["accession"]
                           for row in selection.get("target_quality_evidence", [])
                           if row.get("target_chembl_id") in expected_targets}
    if set(expected_accessions) != expected_targets:
        raise FormalCorpusError("selection receipt lacks source-verified target-component identity")
    proteins_by_accession = {}
    for row in bundle["proteins"]:
        if row.get("accession") in expected_accessions.values():
            proteins_by_accession[row["accession"]] = row
    missing = []
    verified_target_bindings = 0
    for target_id, accession in expected_accessions.items():
        protein = proteins_by_accession.get(accession)
        sequence = protein.get("full_sequence") if isinstance(protein, Mapping) else None
        if (not isinstance(protein, Mapping) or protein.get("sequence_status") != "verified"
                or protein.get("identity_status") != "verified" or not isinstance(sequence, str) or not sequence
                or type(protein.get("sequence_length")) is not int or protein["sequence_length"] != len(sequence)
                or protein.get("sequence_sha256") != _sha(sequence.encode("utf-8"))):
            missing.append({"target_chembl_id": target_id, "accession": accession,
                            "reason": protein.get("sequence_status_reason", "complete_source_sequence_missing") if isinstance(protein, Mapping) else "protein_record_missing"})
            continue
        quality = next(row for row in selection["target_quality_evidence"] if row.get("target_chembl_id") == target_id)
        discovery_receipt = quality.get("target_receipt")
        raw_refs = selection.get("raw_source_receipts", [])
        if not isinstance(discovery_receipt, Mapping) or not isinstance(bundle_root, Path):
            raise FormalCorpusError(f"selected target {target_id} has no verifiable target raw-receipt root")
        discovery_source = next((row for row in raw_refs if isinstance(row, Mapping)
                                 and row.get("url") == discovery_receipt.get("url")
                                 and row.get("raw_path") == discovery_receipt.get("raw_path")
                                 and row.get("raw_sha256") == discovery_receipt.get("raw_sha256")), None)
        if discovery_source is None or discovery_source.get("url") != f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json":
            raise FormalCorpusError(f"selected target {target_id} discovery raw receipt is missing or endpoint-mismatched")
        discovery_root = Path(discovery_source.get("batch_root", "")).resolve(strict=True)
        discovery_path = (discovery_root / str(discovery_source.get("raw_path", ""))).resolve(strict=True)
        if (discovery_root not in discovery_path.parents
                or _sha(discovery_path.read_bytes()) != discovery_source.get("raw_sha256")):
            raise FormalCorpusError(f"selected target {target_id} discovery target raw bytes/hash mismatch")
        discovery_target = json.loads(discovery_path.read_text(encoding="utf-8"))
        if discovery_target.get("target_chembl_id") != target_id:
            raise FormalCorpusError(f"selected target {target_id} discovery raw native ID mismatch")

        component_evidence = protein.get("component_evidence") if isinstance(protein, Mapping) else None
        references = component_evidence.get("references") if isinstance(component_evidence, Mapping) else None
        matching_refs = [ref for ref in references if isinstance(ref, Mapping) and ref.get("target_chembl_id") == target_id] if isinstance(references, list) else []
        if len(matching_refs) != 1:
            raise FormalCorpusError(f"selected target {target_id} must have exactly one captured protein component reference")
        reference = matching_refs[0]
        if (reference.get("accession") != accession or reference.get("taxonomy_id") != quality.get("taxonomy_id")
                or quality.get("taxonomy_id") != 9606):
            raise FormalCorpusError(f"selected target {target_id} captured component identity differs from its sealed target/accession/taxonomy")
        target_source = reference.get("target_record_source")
        if (not isinstance(target_source, Mapping)
                or target_source.get("source_id") != f"chembl_target:{target_id}"
                or target_source.get("source_url") != f"{evidence_builder.CHEMBL_API_ROOT}/target/{target_id}.json"):
            raise FormalCorpusError(f"selected target {target_id} captured component lacks its native target source identity")
        provenance = [row for row in bundle.get("source_provenance", []) if isinstance(row, Mapping)
                      and row.get("source_id") == target_source.get("source_id")
                      and row.get("source_path") == target_source.get("source_path")
                      and row.get("raw_sha256") == target_source.get("raw_sha256")]
        if len(provenance) != 1:
            raise FormalCorpusError(f"selected target {target_id} captured target raw receipt is absent or ambiguous")
        relative = Path(str(target_source.get("source_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FormalCorpusError(f"selected target {target_id} captured target raw path is unsafe")
        captured_path = (bundle_root / relative).resolve(strict=True)
        if bundle_root.resolve() not in captured_path.parents or _sha(captured_path.read_bytes()) != target_source.get("raw_sha256"):
            raise FormalCorpusError(f"selected target {target_id} captured target raw bytes/hash mismatch")
        captured_target = json.loads(captured_path.read_text(encoding="utf-8"))
        if (captured_target.get("target_chembl_id") != target_id
                or reference.get("target_record_sha256") != target_source.get("raw_sha256")):
            raise FormalCorpusError(f"selected target {target_id} captured raw/canonical target SHA domains or native ID mismatch")
        discovery_identity = _target_component_identity(discovery_target, accession)
        captured_identity = _target_component_identity(captured_target, accession)
        if discovery_identity != captured_identity:
            raise FormalCorpusError(f"selected target {target_id} captured native target/component identity changed after selection")
        resolutions = evidence_builder.resolve_target_components(
            captured_target, source_kind="online_snapshot", source_path=str(relative),
            source_sha256=target_source.get("raw_sha256"), source_url=target_source.get("source_url"),
        )
        component_index = reference.get("component_index")
        resolved = next((row for row in resolutions if row.get("component_index") == component_index), None)
        if (not isinstance(resolved, Mapping) or resolved.get("resolution_status") != "verified"
                or resolved.get("target_chembl_id") != target_id or resolved.get("accession") != accession
                or resolved.get("taxonomy_id") != quality.get("taxonomy_id")
                or resolved.get("component") != reference.get("component")):
            raise FormalCorpusError(f"selected target {target_id} captured component did not resolve from its own native source record")
        verified_target_bindings += 1
    if missing:
        raise FormalCorpusError(f"complete UniProt full-sequence hydration failed for selected targets: {missing}")
    if result.get("pagination_status") != "complete" or result.get("metadata_status") != "complete":
        raise FormalCorpusError("full formal capture pagination or linked metadata is incomplete")
    return {"selected_molecule_count": len(expected_molecules), "selected_target_count": len(expected_targets),
            "sealed_activity_rows_verified": sum(sealed_counter.values()),
            "selected_target_component_bindings_verified": verified_target_bindings,
            "captured_activity_rows": len(bundle["activities"]), "verified_full_sequence_targets": len(expected_accessions),
            "sequence_lengths": sorted(len(proteins_by_accession[accession]["full_sequence"]) for accession in expected_accessions.values())}


def _luteolin_neighbor_entity(luteolin: Mapping[str, Any]) -> dict[str, Any]:
    molecule_id = luteolin["molecule_chembl_id"]
    smiles = luteolin["molecule_structures"]["canonical_smiles"]
    return {"entity_id": molecule_id, "molecule_chembl_id": molecule_id,
            "canonical_smiles": smiles, "structure_sha256": _sha(smiles.encode("utf-8")),
            "source_kind": "online_snapshot", "source_provenance": []}


def _lookup_luteolin_from_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    identity = selection.get("luteolin_case")
    if not isinstance(identity, Mapping) or identity.get("lookup_complete") is not True:
        raise FormalCorpusError("sealed selection lacks a complete exact-name luteolin source lookup")
    selected_id = identity.get("selected_chembl_id")
    pages = identity.get("pages")
    detail_receipt = identity.get("detail_receipt")
    lookup = identity.get("selection_rule")
    raw_receipts = selection.get("raw_source_receipts", [])
    if (not isinstance(pages, list) or not pages or not isinstance(detail_receipt, Mapping)
            or not isinstance(lookup, Mapping) or not isinstance(raw_receipts, list)):
        raise FormalCorpusError("luteolin case receipt lacks complete search pages or native-ID detail receipt")
    rows: list[dict[str, Any]] = []
    expected_offset = 0
    expected_total = identity.get("total_count")
    for page_index, page in enumerate(pages):
        if (not isinstance(page, Mapping) or type(page.get("offset")) is not int
                or page["offset"] != expected_offset or type(page.get("row_count")) is not int
                or page["row_count"] < 0 or page.get("total_count") != expected_total):
            raise FormalCorpusError("luteolin exact-name page receipt chain is invalid")
        source_url = page.get("source_url")
        _validate_molecule_lookup_url(source_url, expected_offset=expected_offset, lookup=lookup)
        if not any(isinstance(source, Mapping) and source.get("url") == source_url
                   and source.get("raw_path") == page.get("raw_path")
                   and source.get("raw_sha256") == page.get("raw_sha256") for source in raw_receipts):
            raise FormalCorpusError("luteolin search page is not bound to a persisted request-ledger receipt")
        root_text = page.get("batch_root")
        raw_path = page.get("raw_path")
        if not isinstance(root_text, str) or not isinstance(raw_path, str):
            raise FormalCorpusError("luteolin exact-name page receipt lacks batch root/path")
        batch_root = Path(root_text).resolve(strict=True)
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FormalCorpusError("luteolin exact-name page path is unsafe")
        path = (batch_root / relative).resolve(strict=True)
        if batch_root not in path.parents or _sha(path.read_bytes()) != page.get("raw_sha256"):
            raise FormalCorpusError("luteolin exact-name raw page receipt failed integrity verification")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("molecules")
        meta = payload.get("page_meta")
        if (not isinstance(entries, list) or len(entries) != page["row_count"]
                or not isinstance(meta, Mapping) or type(meta.get("offset")) is not int
                or meta.get("offset") != expected_offset or type(meta.get("limit")) is not int
                or meta.get("limit") != lookup.get("page_size")
                or type(meta.get("total_count")) is not int or meta.get("total_count") != expected_total):
            raise FormalCorpusError("luteolin raw page rows disagree with the sealed page receipt")
        for row in entries:
            if not isinstance(row, Mapping) or str(row.get("pref_name", "")).strip().casefold() != "luteolin":
                raise FormalCorpusError("luteolin raw page contains an out-of-scope preferred name")
            rows.append(dict(row))
        expected_offset += len(entries)
        advertised = meta.get("next")
        if type(advertised) not in (str, type(None)):
            raise FormalCorpusError("luteolin raw page next link is malformed")
        if expected_offset < expected_total:
            expected_next = _validate_molecule_lookup_next(advertised, expected_offset=expected_offset, lookup=lookup)
            next_page = pages[page_index + 1] if page_index + 1 < len(pages) else None
            if not isinstance(next_page, Mapping) or next_page.get("source_url") != expected_next:
                raise FormalCorpusError("luteolin raw next link does not match the sealed contiguous page chain")
        elif advertised is not None:
            raise FormalCorpusError("luteolin final raw page still advertises an unsealed next page")
    candidate_ids = sorted(row.get("molecule_chembl_id") for row in rows)
    if (expected_offset != expected_total or candidate_ids != identity.get("exact_name_candidate_ids")
            or not candidate_ids or candidate_ids[0] != selected_id):
        raise FormalCorpusError("luteolin complete exact-name candidate set does not match its frozen lowest-ID rule")
    detail_url = f"{evidence_builder.CHEMBL_API_ROOT}/molecule/{selected_id}.json"
    if (detail_receipt.get("source_url") != detail_url
            or not any(isinstance(source, Mapping) and source.get("url") == detail_url
                       and source.get("raw_path") == detail_receipt.get("raw_path")
                       and source.get("raw_sha256") == detail_receipt.get("raw_sha256") for source in raw_receipts)):
        raise FormalCorpusError("luteolin selected detail is not bound to its exact native-ID request receipt")
    batch_root = Path(detail_receipt.get("batch_root", "")).resolve(strict=True)
    detail_path = batch_root / str(detail_receipt.get("raw_path", ""))
    detail_path = detail_path.resolve(strict=True)
    if (batch_root not in detail_path.parents or _sha(detail_path.read_bytes()) != detail_receipt.get("raw_sha256")):
        raise FormalCorpusError("luteolin selected detail raw receipt failed integrity verification")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (detail.get("molecule_chembl_id") != selected_id
            or detail.get("molecule_chembl_id") != detail_receipt.get("native_molecule_chembl_id")
            or str(detail.get("pref_name", "")).strip().casefold() != "luteolin"):
        raise FormalCorpusError("luteolin Molecule detail native ID/name disagrees with the sealed case identity")
    structures = detail.get("molecule_structures")
    if not isinstance(structures, Mapping) or not structures.get("canonical_smiles"):
        raise FormalCorpusError("luteolin native-ID detail lacks canonical structure")
    search_selected = next((row for row in rows if row.get("molecule_chembl_id") == selected_id), None)
    search_structures = search_selected.get("molecule_structures") if isinstance(search_selected, Mapping) else None
    if (not isinstance(search_structures, Mapping)
            or search_structures.get("canonical_smiles") != structures.get("canonical_smiles")):
        raise FormalCorpusError("luteolin native-ID detail structure disagrees with its exact-name search row")
    return detail


def _derive_development_excluded_bundle(capture_root: Path, output_root: Path, bundle: Mapping[str, Any],
                                        development_bundle: Mapping[str, Any], luteolin: Mapping[str, Any],
                                        policy_path: Path, workload: Mapping[str, Any], caps: Mapping[str, Any]
                                        ) -> dict[str, Any]:
    """Cluster the complete selected scope, attach development edges, then exclude whole connected components."""
    eligibility_started = time.perf_counter()
    molecule_clusters = dataset_builder.build_molecule_clusters(capture_root / "entities.json", policy_path=policy_path)
    protein_clusters = dataset_builder.build_protein_clusters(capture_root / "entities.json", policy_path=policy_path)
    eligibility_elapsed = time.perf_counter() - eligibility_started
    if molecule_clusters.status not in {"complete", "partial"} or protein_clusters.status not in {"complete", "partial"}:
        raise FormalCorpusError(f"eligibility cluster pass failed: molecule={molecule_clusters.status}, protein={protein_clusters.status}")
    if eligibility_elapsed > caps["eligibility_cluster_wall_seconds"]:
        raise FormalSizeLimit("actual_eligibility_cluster_wall_limit_exceeded")
    molecule_membership = _cluster_member_map(molecule_clusters)
    protein_membership = _cluster_member_map(protein_clusters)
    dev_molecules, _ = dataset_builder._molecule_entities(development_bundle)
    luteolin_entity = _luteolin_neighbor_entity(luteolin)
    if luteolin_entity["molecule_chembl_id"] not in {row["molecule_chembl_id"] for row in dev_molecules}:
        dev_molecules.append(luteolin_entity)
    dev_proteins = sorted(development_bundle["proteins"], key=lambda row: (row["taxonomy_id"], row["accession"]))
    formal_protein_records = {f"{row['taxonomy_id']}|{row['accession']}": row for row in bundle["proteins"]}
    development_molecule_nodes = {row["molecule_chembl_id"]: f"development:molecule:{row['molecule_chembl_id']}" for row in dev_molecules}
    development_protein_nodes = {f"{row['taxonomy_id']}|{row['accession']}": f"development:protein:{row['taxonomy_id']}|{row['accession']}" for row in dev_proteins}
    molecule_edges: list[dict[str, Any]] = []
    protein_edges: list[dict[str, Any]] = []
    molecule_pairs_used = 0
    molecule_started = time.perf_counter()
    formal_molecules = list(molecule_clusters.entities)
    for formal in formal_molecules:
        for development in dev_molecules:
            molecule_pairs_used += 1
            if molecule_pairs_used > caps["development_molecule_neighbor_pairs"]:
                raise FormalSizeLimit("actual_development_molecule_neighbor_pair_cap_exceeded")
            if formal["molecule_chembl_id"] == development["molecule_chembl_id"]:
                edges = (dataset_builder.NeighborEvidence(formal["entity_id"], development["entity_id"],
                                                          "exact_development_identity", 1.0, {}),)
            else:
                edges = dataset_builder.molecule_neighbor(formal, development, policy_path=policy_path)
            for evidence in edges:
                edge = _as_cluster_edge(evidence, molecule_membership, development_molecule_nodes)
                if edge is not None:
                    molecule_edges.append(edge)
    for index, left in enumerate(dev_molecules):
        for right in dev_molecules[index + 1:]:
            molecule_pairs_used += 1
            edges = dataset_builder.molecule_neighbor(left, right, policy_path=policy_path)
            for evidence in edges:
                edge = _as_cluster_edge(evidence, molecule_membership, development_molecule_nodes)
                if edge is not None:
                    molecule_edges.append(edge)
    molecule_elapsed = time.perf_counter() - molecule_started
    if molecule_pairs_used != workload["development_molecule_neighbor_pairs"] or molecule_elapsed > caps["development_molecule_neighbor_wall_seconds"]:
        raise FormalSizeLimit("actual_development_molecule_neighbor_wall_limit_or_pair_mismatch")

    protein_pairs_used = 0
    protein_cells_used = 0
    protein_started = time.perf_counter()
    formal_proteins = list(protein_clusters.entities)
    for formal_entity in formal_proteins:
        formal = formal_protein_records[formal_entity["entity_id"]]
        for development in dev_proteins:
            protein_pairs_used += 1
            cells = len(formal["full_sequence"]) * len(development["full_sequence"])
            protein_cells_used += cells
            if (protein_pairs_used > caps["development_neighbor_pairs"]
                    or protein_cells_used > caps["development_neighbor_alignment_cells"]):
                raise FormalSizeLimit("actual_development_protein_neighbor_pair_or_cell_cap_exceeded_before_comparison")
            formal_id = f"{formal['taxonomy_id']}|{formal['accession']}"
            development_id = f"{development['taxonomy_id']}|{development['accession']}"
            if formal_id == development_id:
                edges = (dataset_builder.NeighborEvidence(formal_id, development_id,
                                                          "exact_development_identity", 1.0, {}),)
            else:
                edges = dataset_builder.protein_neighbor(formal, development, policy_path=policy_path)
            for evidence in edges:
                edge = _as_cluster_edge(evidence, protein_membership, development_protein_nodes)
                if edge is not None:
                    protein_edges.append(edge)
    for index, left in enumerate(dev_proteins):
        for right in dev_proteins[index + 1:]:
            protein_pairs_used += 1
            cells = len(left["full_sequence"]) * len(right["full_sequence"])
            protein_cells_used += cells
            if (protein_pairs_used > caps["development_neighbor_pairs"]
                    or protein_cells_used > caps["development_neighbor_alignment_cells"]):
                raise FormalSizeLimit("actual_development_protein_seed_pair_or_cell_cap_exceeded_before_comparison")
            edges = dataset_builder.protein_neighbor(left, right, policy_path=policy_path)
            for evidence in edges:
                edge = _as_cluster_edge(evidence, protein_membership, development_protein_nodes)
                if edge is not None:
                    protein_edges.append(edge)
    protein_elapsed = time.perf_counter() - protein_started
    if (protein_pairs_used != workload["development_neighbor_pairs"]
            or protein_cells_used != workload["development_neighbor_alignment_cells"]
            or protein_elapsed > caps["development_neighbor_wall_seconds"]):
        raise FormalSizeLimit("actual_development_protein_neighbor_wall_limit_or_pair_mismatch")

    molecule_touched, molecule_components = _development_touched_cluster_ids(
        [cluster.cluster_id for cluster in molecule_clusters.clusters], list(development_molecule_nodes.values()), molecule_edges)
    protein_touched, protein_components = _development_touched_cluster_ids(
        [cluster.cluster_id for cluster in protein_clusters.clusters], list(development_protein_nodes.values()), protein_edges)
    excluded_molecule_ids = {member for member, cluster_id in molecule_membership.items() if cluster_id in molecule_touched}
    excluded_protein_ids = {member for member, cluster_id in protein_membership.items() if cluster_id in protein_touched}
    retained_molecules = [row for row in bundle["molecules"] if row.get("molecule_chembl_id") not in excluded_molecule_ids]
    retained_proteins = [row for row in bundle["proteins"]
                         if f"{row.get('taxonomy_id')}|{row.get('accession')}" not in excluded_protein_ids]
    exclusions = list(bundle.get("exclusions", []))
    for row in bundle["molecules"]:
        molecule_id = row.get("molecule_chembl_id")
        if molecule_id in excluded_molecule_ids:
            exclusions.append({"entity_id": molecule_id, "reason": "development_neighbor_connected_component",
                               "cluster_id": molecule_membership[molecule_id],
                               "source_provenance": row.get("source_provenance", [])})
    for row in bundle["proteins"]:
        entity_id = f"{row.get('taxonomy_id')}|{row.get('accession')}"
        if entity_id in excluded_protein_ids:
            exclusions.append({"entity_id": entity_id, "reason": "development_neighbor_connected_component",
                               "cluster_id": protein_membership[entity_id],
                               "source_provenance": row.get("source_provenance", [])})
    if not retained_molecules or not retained_proteins:
        raise FormalCorpusError("development neighbor component exclusions exhausted a formal entity cohort; no reselection is permitted")
    destination = output_root / "formal_bundle"
    destination.mkdir(parents=True, exist_ok=False)
    copied_sources: dict[str, str] = {}
    for receipt in bundle["source_provenance"]:
        relative = Path(receipt["source_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FormalCorpusError("capture bundle raw source path escaped before derived bundle copy")
        source = capture_root / relative
        artifact_store._assert_no_reparse_components(source)
        raw = source.read_bytes()
        if _sha(raw) != receipt.get("raw_sha256"):
            raise FormalCorpusError("capture bundle raw source bytes changed before derived bundle copy")
        if str(relative) in copied_sources:
            if copied_sources[str(relative)] != receipt["raw_sha256"]:
                raise FormalCorpusError("duplicate capture raw path binds conflicting hashes")
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FormalCorpusError("derived raw source path already exists")
        target.write_bytes(raw)
        copied_sources[str(relative)] = receipt["raw_sha256"]
    derived = evidence_builder.write_entity_bundle(
        destination, activities=bundle["activities"], source_provenance=bundle["source_provenance"],
        molecules=retained_molecules, proteins=retained_proteins, exclusions=exclusions,
        source_kind="online_snapshot",
    )
    checked = evidence_builder.load_entity_bundle(destination / "entities.json", verify_sources=True)
    if checked.get("content_sha256") != derived.get("content_sha256"):
        raise FormalCorpusError("derived entity bundle did not reverify after exclusions")
    return {"status": "complete", "source_bundle_sha256": bundle["content_sha256"],
            "derived_bundle_sha256": checked["content_sha256"], "bundle_path": str(destination / "entities.json"),
            "original_molecule_count": len(bundle["molecules"]), "retained_molecule_count": len(retained_molecules),
            "excluded_molecule_count": len(excluded_molecule_ids),
            "original_protein_count": len(bundle["proteins"]), "retained_protein_count": len(retained_proteins),
            "excluded_protein_count": len(excluded_protein_ids),
            "molecule_development_neighbor_pairs": molecule_pairs_used,
            "molecule_development_neighbor_seconds": molecule_elapsed,
            "protein_development_neighbor_pairs": protein_pairs_used,
            "protein_development_neighbor_cells": protein_cells_used,
            "protein_development_neighbor_seconds": protein_elapsed,
            "molecule_components_touching_development": molecule_components,
            "protein_components_touching_development": protein_components,
            "molecule_edges": molecule_edges, "protein_edges": protein_edges}


def _support_diagnostic(manifest_path: Path, audit_report: Any, support_policy_path: Path) -> dict[str, Any]:
    support, _ = _read_object(support_policy_path)
    manifest, _ = _read_object(manifest_path)
    assignments = manifest.get("split", {}).get("pair_assignments", [])
    if not isinstance(assignments, list):
        raise FormalCorpusError("frozen split manifest pair_assignments is not a list")
    distinct = {name: {"molecule_clusters": set(), "protein_clusters": set()}
                for name in ("train", "val", "test")}
    for row in assignments:
        if not isinstance(row, Mapping) or row.get("split") not in distinct:
            continue
        if isinstance(row.get("molecule_cluster_id"), str):
            distinct[row["split"]]["molecule_clusters"].add(row["molecule_cluster_id"])
        if isinstance(row.get("protein_cluster_id"), str):
            distinct[row["split"]]["protein_clusters"].add(row["protein_cluster_id"])
    component_counts = {name: {key: len(value[key]) for key in value} for name, value in distinct.items()}
    class_support = {name: dict(value) for name, value in audit_report.class_support.items()}
    minimum_classes = support["minimum_labeled_examples_per_class"]
    minimum_molecules = support["minimum_distinct_connected_molecule_clusters"]
    minimum_proteins = support["minimum_distinct_connected_protein_clusters"]
    shortfalls = []
    for split in ("train", "val", "test"):
        for label in ("active", "inactive"):
            required = minimum_classes[split]
            actual = class_support.get(split, {}).get(label, 0)
            if actual < required:
                shortfalls.append({"split": split, "metric": f"{label}_examples", "actual": actual, "minimum": required})
        for field, minimum in (("molecule_clusters", minimum_molecules[split]),
                               ("protein_clusters", minimum_proteins[split])):
            if minimum is not None and component_counts[split][field] < minimum:
                shortfalls.append({"split": split, "metric": field,
                                   "actual": component_counts[split][field], "minimum": minimum})
    return {"status": "sufficient_for_minimal_baseline_attempt" if not shortfalls else "blocked_support_shortfall",
            "support_policy_sha256": _sha(support_policy_path.read_bytes()),
            "class_support": class_support, "distinct_cluster_counts_by_split": component_counts,
            "minimums": {"labeled_examples_per_class": minimum_classes,
                         "molecule_clusters": minimum_molecules, "protein_clusters": minimum_proteins},
            "shortfalls": shortfalls, "training_ready": False,
            "interpretation": "engineering admission floor only; no statistical-power or model-performance claim"}


def _capture_summary(result: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "status": result.get("status"),
            "pagination_status": result.get("pagination_status"),
            "metadata_status": result.get("metadata_status"),
            "molecule_ids": result.get("molecule_ids"),
            "activity_filters": result.get("activity_filters"),
            "request_count": result.get("request_count"), "page_count": result.get("page_count"),
            "activity_count": result.get("activity_count"),
            "entity_bundle_content_sha256": bundle.get("content_sha256"),
            "entity_version": bundle.get("entity_version"),
            "complete_sequence_accessions": sorted(row.get("accession") for row in bundle.get("proteins", [])
                                                     if isinstance(row, Mapping) and row.get("sequence_status") == "verified"
                                                     and row.get("identity_status") == "verified")}


def _validate_capture_summary(path: Path, selection: Mapping[str, Any], config: Mapping[str, Any],
                              capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, raw = _read_object(path.resolve(strict=True))
    if summary.get("status") != "complete" or summary.get("pagination_status") != "complete" or summary.get("metadata_status") != "complete":
        raise FormalCorpusError("prior capture summary is not a complete source scope")
    bundle_path = capture_root / "entities.json"
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    if bundle.get("content_sha256") != summary.get("entity_bundle_content_sha256"):
        raise FormalCorpusError("prior complete capture bundle content hash changed")
    if summary.get("molecule_ids") != selection["selected_molecule_ids"]:
        raise FormalCorpusError("prior capture summary molecule IDs differ from sealed selection")
    expected_filters = {"standard_type": "Ki", "target_organism": "Homo sapiens",
                        "target_chembl_id__in": selection["selected_target_chembl_ids"]}
    if summary.get("activity_filters") != expected_filters:
        raise FormalCorpusError("prior capture summary source filter differs from sealed selection")
    _validate_complete_capture(bundle, summary, selection, capture_root)
    return summary, bundle


def _support_rows(manifest_path: Path) -> dict[str, Any]:
    manifest, raw = _read_object(manifest_path)
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise FormalCorpusError("frozen manifest omitted split evidence")
    assignments = split.get("pair_assignments")
    if not isinstance(assignments, list):
        raise FormalCorpusError("frozen manifest pair assignments are invalid")
    retained = [row for row in assignments if isinstance(row, Mapping) and row.get("split") in {"train", "val", "test"}]
    return {"manifest_sha256": _sha(raw),
            "assignment_rows": len(assignments), "retained_assignment_rows": len(retained),
            "unassigned_pairs": len(split.get("unassigned_pairs", [])),
            "crosspair_quarantine": len(split.get("crosspair_quarantine", [])),
            "ambiguous_label_quarantine": len(split.get("ambiguous_label_quarantine", [])),
            "assigned_pair_signatures": [{"pair_key": row.get("pair_key"), "split": row.get("split"),
                                           "molecule_cluster_id": row.get("molecule_cluster_id"),
                                           "protein_cluster_id": row.get("protein_cluster_id")}
                                          for row in retained]}


def expand(config_path: Any, output_dir: Any, *, allow_online: bool = False,
           resume_from: Any = None) -> dict[str, Any]:
    if allow_online is not True:
        raise FormalCorpusError("formal expansion is offline by default; pass --allow-online after reviewing the frozen protocol")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    config, decision = _verify_config(config_file)
    config_sha256 = _sha(config_file.read_bytes())
    resume = Path(resume_from).expanduser().resolve(strict=True) if resume_from is not None else None
    # Validate the prior root/ledger/receipt before creating a new output directory.
    prior_root = BatchRegistry.validate_start(config, config_sha256, resume)
    root = _safe_new_root(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    registered_utc = datetime.now(timezone.utc).isoformat()
    protocol = {"schema_version": 1, "protocol_version": config["protocol_version"],
                "config_sha256": config_sha256, "decision_sha256": config["budget"]["decision_sha256"],
                "base_corpus_sha256": config["base_corpus_sha256"],
                "scope": config["scope"], "selection_window": config["selection_window"],
                "caps": decision["caps"], "registered_utc_before_source_requests": registered_utc,
                "prior_batch_root": str(prior_root) if prior_root else None,
                "global_population_claim": False, "selection_values_or_classes_used": False}
    _write_json_new(root / "formal_batch_protocol.json", protocol)
    registry = BatchRegistry(config, config_sha256, root, resume)
    ledger = RequestLedger(root, config, decision, prior_root=registry.prior_root,
                           registry=registry, config_sha256=config_sha256)
    capture_root: Path
    selection: dict[str, Any]
    selection_sha256: str
    runtime_budget: RuntimeComputeBudget | None = None
    try:
        previous_selection = None
        if registry.prior_root is not None:
            prior_registry, _ = _read_object(registry.path)
            if prior_registry.get("selection_receipt_sha256"):
                previous_selection = registry.prior_root / "selection_receipt.json"
        if previous_selection is not None:
            selection, selection_sha256 = _copy_sealed_selection(
                registry.prior_root, root, config, config_sha256,
                _read_object(registry.path)[0]["selection_receipt_sha256"],
            )
            registry.set_selection(root, selection_sha256)
        else:
            ledger.phase = "discovery"
            luteolin, luteolin_receipt = _lookup_luteolin(ledger, config)
            rows, window = _discover_window(ledger, config)
            selection = _select_from_window(ledger, config, rows, *_development_bundle(config)[:1], luteolin, luteolin_receipt)
            selection["source_snapshot"] = window["source_snapshot"]
            selection["selection_window"] = window["query_window"]
            selection["selection_window_status"] = window["query_window"]["coverage"]
            if selection["selection_status"] != "sealed":
                ledger.reconcile_activity_pages()
                exhausted = dict(selection)
                exhausted.update({"source_snapshot": window["source_snapshot"],
                                  "selection_window": window["query_window"],
                                  "raw_source_receipts": _selection_raw_receipts(ledger),
                                  "registered_utc_before_source_queries": registered_utc,
                                  "selection_budget_usage": ledger.summary(),
                                  "global_population_claim": False})
                _write_json_new(root / "selection_window_exhausted.json", exhausted)
                registry.finish("selection_window_exhausted", _sha((root / "selection_window_exhausted.json").read_bytes()))
                return {"status": "selection_window_exhausted", "reason": "fixed bounded window produced no eligible molecule/target scope",
                        "output_root": str(root), "selection_window": window["query_window"],
                        "budget_usage": ledger.summary()}
            selection["global_population_claim"] = False
            selection["source_snapshot"] = window["source_snapshot"]
            selection, selection_sha256 = _seal_selection(root, ledger, config, config_sha256,
                                                          selection, window, luteolin_receipt, registered_utc)
            registry.set_selection(root, selection_sha256)

        # A second release check binds capture pages to the selection release.
        ledger.phase = "capture"
        capture_status_url = f"{evidence_builder.CHEMBL_API_ROOT}/status.json"
        capture_status, _, capture_status_sha = _fetch_json(ledger, capture_status_url)
        capture_snapshot = capture_status.get("chembl_db_version") or capture_status.get("database_version")
        if capture_snapshot != selection.get("source_snapshot"):
            raise FormalCorpusError("ChEMBL release changed between selection and full source capture")

        if registry.prior_root is not None and (registry.prior_root / "capture" / "capture_result.json").is_file():
            capture_root = registry.prior_root / "capture"
            capture_result, bundle = _validate_capture_summary(capture_root / "capture_result.json", selection, config, capture_root)
            reused_capture = True
        else:
            usage = ledger._used()
            chembl_used = ledger._used("chembl")
            uniprot_used = ledger._used("uniprot")
            capture_request_cap = min(1536, int(decision["caps"]["chembl_requests"] - chembl_used["requests"]))
            capture_page_cap = min(246, int(decision["caps"]["chembl_activity_pages"] - usage["pages"]))
            capture_row_cap = min(19000, int(decision["caps"]["chembl_activity_rows"] - usage["rows"]))
            uniprot_request_cap = int(decision["caps"]["uniprot_requests"] - uniprot_used["requests"])
            uniprot_byte_cap = int(decision["caps"]["uniprot_response_bytes_total"] - uniprot_used["bytes"])
            remaining_wall = decision["caps"]["chembl_wall_seconds"] - usage["wall"]
            if min(capture_request_cap, capture_page_cap, capture_row_cap, uniprot_request_cap) <= 0 or remaining_wall <= 0:
                raise FormalCorpusError("cumulative whole-batch reserve is exhausted before formal capture")
            capture_root = root / "capture"
            filters = {"standard_type": "Ki", "target_organism": "Homo sapiens",
                       "target_chembl_id__in": selection["selected_target_chembl_ids"]}
            capture_result = evidence_builder.fetch_chembl_bounded(
                selection["selected_molecule_ids"], capture_root,
                max_pages=capture_page_cap, max_records=capture_row_cap, timeout=30,
                max_requests=capture_request_cap, max_response_bytes=8_388_608,
                http_get=ledger.http_get, per_request_timeout=True,
                total_timeout_seconds=remaining_wall, activity_filters=filters,
                hydrate_uniprot=True, max_uniprot_requests=uniprot_request_cap,
                uniprot_max_response_bytes=8_388_608,
                uniprot_total_response_bytes=uniprot_byte_cap,
            )
            ledger.reconcile_activity_pages()
            if capture_result.get("status") != "complete" or not (capture_root / "entities.json").is_file():
                summary = {"status": "incomplete", "reason": capture_result.get("reason") or capture_result.get("incomplete_reasons"),
                           "selection_receipt_sha256": selection_sha256,
                           "capture_result": {key: capture_result.get(key) for key in (
                               "status", "pagination_status", "metadata_status", "request_count", "page_count",
                               "activity_count", "incomplete_reasons", "budgets", "uniprot_budget")},
                           "budget_usage": ledger.summary(), "output_root": str(root),
                           "source_snapshot": selection.get("source_snapshot"),
                           "complete_global_source": False}
                receipt_raw = _write_json_new(root / "formal_expansion_receipt.json", summary)
                return {**summary, "receipt_sha256": _sha(receipt_raw)}
            bundle = evidence_builder.load_entity_bundle(capture_root / "entities.json", verify_sources=True)
            capture_integrity = _validate_complete_capture(bundle, capture_result, selection, capture_root)
            capture_summary = _capture_summary(capture_result, bundle)
            capture_summary["capture_source_snapshot"] = capture_snapshot
            capture_summary["capture_status_raw_sha256"] = capture_status_sha
            capture_summary["capture_integrity"] = capture_integrity
            _write_json_new(capture_root / "capture_result.json", capture_summary)
            reused_capture = False

        _require_formal_online_snapshot(bundle)
        initial_preflight_root = root / __import__("uuid").uuid4().hex
        initial_preflight = preflight(config_file, capture_root / "entities.json", initial_preflight_root)
        if initial_preflight["status"] != "within_budget":
            result = {"status": "size_limit", "reason": initial_preflight.get("reason"),
                      "selection_receipt_sha256": selection_sha256,
                      "capture_bundle_sha256": bundle["content_sha256"],
                      "preflight_path": str(initial_preflight_root / "preflight.json"),
                      "budget_usage": ledger.summary(), "output_root": str(root), "training_ready": False}
            raw = _write_json_new(root / "formal_expansion_receipt.json", result)
            registry.finish("size_limit", _sha(raw))
            return {**result, "receipt_sha256": _sha(raw)}

        policy_path = _runtime_split_policy(root, config, decision)
        development_bundle, _ = _development_bundle(config)
        runtime_budget = RuntimeComputeBudget(decision["caps"])

        def record_runtime_size_limit(reason: str, **details: Any) -> dict[str, Any]:
            if runtime_budget.exceeded_reason is None:
                runtime_budget.exceeded_reason = reason
            result = {"status": "size_limit", "reason": reason,
                      "selection_receipt_sha256": selection_sha256,
                      "capture_bundle_sha256": bundle["content_sha256"],
                      "budget_usage": ledger.summary(), "runtime_compute_budget": runtime_budget.to_dict(),
                      "output_root": str(root), "training_ready": False, **details}
            raw = _write_json_new(root / "formal_expansion_receipt.json", result)
            registry.finish("size_limit", _sha(raw))
            return {**result, "receipt_sha256": _sha(raw)}

        try:
            exclusions = runtime_budget.run_pass(
                "eligibility_cluster_and_development_neighbors", _derive_development_excluded_bundle,
                decision["caps"]["eligibility_cluster_wall_seconds"],
                capture_root, root, bundle, development_bundle,
                _lookup_luteolin_from_selection(selection), policy_path,
                initial_preflight["workload"], decision["caps"],
            )
        except FormalSizeLimit as exc:
            return record_runtime_size_limit(str(exc))
        exclusion_raw = _write_json_new(root / "development_exclusion_receipt.json", exclusions)
        final_bundle_path = Path(exclusions["bundle_path"])
        final_preflight_root = root / __import__("uuid").uuid4().hex
        final_preflight = preflight(config_file, final_bundle_path, final_preflight_root)
        if final_preflight["status"] != "within_budget":
            result = {"status": "size_limit", "reason": final_preflight.get("reason"),
                      "selection_receipt_sha256": selection_sha256,
                      "exclusion_receipt_sha256": _sha(exclusion_raw),
                      "preflight_path": str(final_preflight_root / "preflight.json"),
                      "budget_usage": ledger.summary(), "output_root": str(root), "training_ready": False}
            raw = _write_json_new(root / "formal_expansion_receipt.json", result)
            registry.finish("size_limit", _sha(raw))
            return {**result, "receipt_sha256": _sha(raw)}

        frozen_root = root / "frozen"
        original_strict = dataset_builder.strict_double_cold_split
        original_candidate_audit = artifact_store._audit_candidate_split
        original_manifest_audit = artifact_store.audit_split
        candidate_audit_count = 0

        def timed_strict(*args: Any, **kwargs: Any) -> Any:
            return runtime_budget.run_pass("strict_builder", original_strict,
                                           decision["caps"]["builder_wall_seconds"], *args, **kwargs)

        def timed_candidate_audit(*args: Any, **kwargs: Any) -> Any:
            nonlocal candidate_audit_count
            candidate_audit_count += 1
            return runtime_budget.run_pass(f"publisher_candidate_audit_{candidate_audit_count}",
                                           original_candidate_audit,
                                           decision["caps"]["publisher_audit_wall_seconds"], *args, **kwargs)

        def timed_manifest_audit(*args: Any, **kwargs: Any) -> Any:
            return runtime_budget.run_pass("publisher_manifest_audit", original_manifest_audit,
                                           decision["caps"]["publisher_audit_wall_seconds"], *args, **kwargs)

        try:
            dataset_builder.strict_double_cold_split = timed_strict
            artifact_store._audit_candidate_split = timed_candidate_audit
            artifact_store.audit_split = timed_manifest_audit
            runtime_budget.check_total()
            frozen = dataset_builder.publish_frozen_dataset(
                final_bundle_path, frozen_root,
                version_id=f"corpus-formal-v1-{exclusions['derived_bundle_sha256'][:12]}",
                policy_path=policy_path, label_policy_path=APL_ROOT / config["label_policy_path"],
            )
        except FormalSizeLimit as exc:
            return record_runtime_size_limit(str(exc))
        finally:
            dataset_builder.strict_double_cold_split = original_strict
            artifact_store._audit_candidate_split = original_candidate_audit
            artifact_store.audit_split = original_manifest_audit
        if runtime_budget.exceeded_reason:
            return record_runtime_size_limit(runtime_budget.exceeded_reason,
                                             frozen_status=frozen.status, frozen_error=frozen.error)
        runtime_budget.check_total()
        frozen_dict = frozen.to_dict()
        if frozen.status not in {"complete", "insufficient"} or not frozen.manifest_path:
            raise FormalCorpusError(f"candidate freeze failed: {frozen.error or frozen.status}")
        try:
            audit = runtime_budget.run_pass(
                "independent_audit", leak_audit.audit_split, decision["caps"]["auditor_wall_seconds"],
                frozen.manifest_path, policy_path=policy_path,
                label_policy_path=APL_ROOT / config["label_policy_path"],
                budget_overrides={"molecule_pairs": decision["caps"]["auditor_molecule_pairs"],
                                  "protein_pairs": decision["caps"]["auditor_protein_pairs"],
                                  "alignment_cells": decision["caps"]["auditor_alignment_cells"]},
            )
        except FormalSizeLimit as exc:
            return record_runtime_size_limit(str(exc))
        audit_dict = audit.to_dict()
        _write_json_new(root / "independent_audit.json", audit_dict)
        if audit.status != "clean" or audit.exit_code != 0:
            raise FormalCorpusError(f"independent candidate audit not clean: {audit.status} exit={audit.exit_code}")
        support = _support_diagnostic(Path(frozen.manifest_path), audit, policy_path.parent / "support_v1.json")
        split_metrics = _support_rows(Path(frozen.manifest_path))
        runtime_budget.check_total()
        result = {
            "schema_version": 1, "status": "candidate_audited_not_ready",
            "protocol_version": config["protocol_version"], "config_sha256": config_sha256,
            "decision_sha256": config["budget"]["decision_sha256"],
            "base_corpus_sha256": config["base_corpus_sha256"],
            "label_policy_sha256": config["label_policy_sha256"],
            "split_policy_sha256": config["split_policy_sha256"],
            "support_policy_sha256": config["support_policy_sha256"],
            "selection_receipt_sha256": selection_sha256,
            "source_snapshot": selection["source_snapshot"],
            "source_window_complete_global": False,
            "selected_molecule_ids": selection["selected_molecule_ids"],
            "selected_target_chembl_ids": selection["selected_target_chembl_ids"],
            "capture_bundle_sha256": bundle["content_sha256"],
            "derived_bundle_sha256": exclusions["derived_bundle_sha256"],
            "frozen_manifest_path": frozen.manifest_path,
            "frozen_manifest_sha256": _sha(Path(frozen.manifest_path).read_bytes()),
            "frozen_manifest_semantic_sha256": frozen.semantic_sha256,
            "independent_audit": audit_dict,
            "support": support,
            "split_metrics": split_metrics,
            "development_exclusion_receipt_sha256": _sha(exclusion_raw),
            "development_exclusion": {key: exclusions[key] for key in (
                "original_molecule_count", "retained_molecule_count", "excluded_molecule_count",
                "original_protein_count", "retained_protein_count", "excluded_protein_count",
                "molecule_development_neighbor_pairs", "molecule_development_neighbor_seconds",
                "protein_development_neighbor_pairs", "protein_development_neighbor_cells",
                "protein_development_neighbor_seconds")},
            "budget_usage": ledger.summary(),
            "runtime_compute_budget": runtime_budget.to_dict(),
            "support_sufficient_for_minimal_baseline_attempt": support["status"] == "sufficient_for_minimal_baseline_attempt",
            "training_ready": False,
            "training_blockers": ["real local ESM inference readiness is a separate Phase 02-04 acceptance",
                                  "candidate corpus is not admitted to training by this plan"],
            "reused_prior_complete_capture": reused_capture,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt_raw = _write_json_new(root / "formal_expansion_receipt.json", result)
        ledger.reconcile_activity_pages()
        registry.finish("candidate_audited_not_ready", _sha(receipt_raw))
        return {**result, "receipt_sha256": _sha(receipt_raw), "output_root": str(root)}
    except BaseException as exc:
        ledger.settle_pending(f"{type(exc).__name__}: {exc}")
        try:
            _write_json_new(root / "attempt_error.json", {"status": "incomplete", "error_type": type(exc).__name__,
                                                           "error": str(exc), "budget_usage": ledger.summary(),
                                                           "runtime_compute_budget": runtime_budget.to_dict() if runtime_budget is not None else None,
                                                           "selection_receipt_sha256": _read_object(root / "selection_receipt.json")[1]
                                                           if (root / "selection_receipt.json").is_file() else None})
        except Exception:
            pass
        raise


def _workload(molecule_count: int, protein_lengths: Sequence[int], caps: Mapping[str, Any],
              cells_per_second: float, development_lengths: Sequence[int] = (),
              development_molecule_count: int = 3, molecule_warm_seconds: float = 0.0,
              molecule_cold_seconds: float = 0.0) -> dict[str, Any]:
    molecule_pairs = molecule_count * (molecule_count - 1) // 2
    protein_count = len(protein_lengths)
    protein_pairs = protein_count * (protein_count - 1) // 2
    development_pairs = protein_count * len(development_lengths) + len(development_lengths) * (len(development_lengths) - 1) // 2
    development_molecule_pairs = molecule_count * development_molecule_count + development_molecule_count * (development_molecule_count - 1) // 2
    development_molecule_seconds = (molecule_cold_seconds + development_molecule_pairs * molecule_warm_seconds
                                    if development_molecule_pairs else 0.0)
    if molecule_pairs > caps["builder_molecule_pairs"] or protein_pairs > caps["builder_protein_pairs"]:
        return {"status": "size_limit", "molecule_pairs": molecule_pairs, "protein_pairs": protein_pairs,
                "alignment_cells": None, "max_single_alignment_cells": None,
                "development_neighbor_pairs": development_pairs,
                "development_neighbor_alignment_cells": None,
                "development_molecule_neighbor_pairs": development_molecule_pairs,
                "development_molecule_neighbor_estimated_seconds": None,
                "eligibility_cluster_molecule_pairs": molecule_pairs,
                "eligibility_cluster_protein_pairs": protein_pairs,
                "eligibility_cluster_alignment_cells": None,
                "publisher_audit_passes": caps["publisher_audit_passes"],
                "publisher_audit_each_estimated_seconds": None,
                "full_neighbor_pass_count": 1 + 1 + caps["publisher_audit_passes"] + 1,
                "pipeline_total_alignment_cells": None,
                "pipeline_total_estimated_seconds": None,
                "builder_estimated_seconds": None, "auditor_estimated_seconds": None,
                "reason": "pair_count_exceeds_registered_budget_before_alignment"}
    development_cells = (sum(length * dev_length for length in protein_lengths for dev_length in development_lengths)
                         + sum(left * right for i, left in enumerate(development_lengths) for right in development_lengths[i + 1:]))
    cells = sum(left * right for i, left in enumerate(protein_lengths) for right in protein_lengths[i + 1:])
    maximum_pair = max((left * right for i, left in enumerate(protein_lengths) for right in protein_lengths[i + 1:]), default=0)
    builder_seconds = cells / cells_per_second
    auditor_seconds = cells / cells_per_second
    eligibility_seconds = cells / cells_per_second
    formal_molecule_seconds = molecule_cold_seconds + molecule_pairs * molecule_warm_seconds
    builder_seconds += formal_molecule_seconds
    auditor_seconds += formal_molecule_seconds
    eligibility_seconds += formal_molecule_seconds
    publisher_seconds = caps["publisher_audit_passes"] * (cells / cells_per_second + formal_molecule_seconds)
    total_passes = 1 + 1 + caps["publisher_audit_passes"] + 1
    total_cells = total_passes * cells + development_cells
    total_seconds = (total_cells / cells_per_second + total_passes * formal_molecule_seconds
                     + development_molecule_seconds)
    over = (cells > caps["builder_alignment_cells"] or cells > caps["auditor_alignment_cells"]
            or builder_seconds > caps["builder_wall_seconds"] or auditor_seconds > caps["auditor_wall_seconds"]
            or development_pairs > caps["development_neighbor_pairs"]
            or development_cells > caps["development_neighbor_alignment_cells"]
            or development_cells / cells_per_second > caps["development_neighbor_wall_seconds"]
            or development_molecule_pairs > caps["development_molecule_neighbor_pairs"]
            or development_molecule_seconds > caps["development_molecule_neighbor_wall_seconds"]
            or molecule_pairs > caps["eligibility_cluster_molecule_pairs"]
            or protein_pairs > caps["eligibility_cluster_protein_pairs"]
            or cells > caps["eligibility_cluster_alignment_cells"]
            or eligibility_seconds > caps["eligibility_cluster_wall_seconds"]
            or cells > caps["publisher_audit_alignment_cells"]
            or cells / cells_per_second + formal_molecule_seconds > caps["publisher_audit_wall_seconds"]
            or total_cells > caps["pipeline_total_alignment_cells"]
            or total_seconds > caps["pipeline_total_wall_seconds"])
    return {"status": "size_limit" if over else "within_budget", "molecule_pairs": molecule_pairs,
            "protein_pairs": protein_pairs, "alignment_cells": cells,
            "development_neighbor_pairs": development_pairs,
            "development_neighbor_alignment_cells": development_cells,
            "development_neighbor_estimated_seconds": development_cells / cells_per_second,
            "development_molecule_neighbor_pairs": development_molecule_pairs,
            "development_molecule_neighbor_estimated_seconds": development_molecule_seconds,
            "eligibility_cluster_molecule_pairs": molecule_pairs,
            "eligibility_cluster_protein_pairs": protein_pairs,
            "eligibility_cluster_alignment_cells": cells,
            "eligibility_cluster_estimated_seconds": eligibility_seconds,
            "publisher_audit_passes": caps["publisher_audit_passes"],
            "publisher_audit_each_estimated_seconds": cells / cells_per_second + formal_molecule_seconds,
            "full_neighbor_pass_count": total_passes,
            "pipeline_total_alignment_cells": total_cells,
            "pipeline_total_estimated_seconds": total_seconds,
            "max_single_alignment_cells": maximum_pair, "builder_estimated_seconds": builder_seconds,
            "auditor_estimated_seconds": auditor_seconds,
            "reason": "alignment_or_time_budget_exceeded" if over else None}


def preflight(config_path: Any, entities_path: Any, output_dir: Any) -> dict[str, Any]:
    """Verify a source bundle and calculate pair/cell/time costs before alignment."""
    started = time.perf_counter()
    config, decision = _verify_config(Path(config_path))
    bundle_path = Path(entities_path).expanduser().resolve(strict=True)
    bundle = evidence_builder.load_entity_bundle(bundle_path, verify_sources=True)
    _require_formal_online_snapshot(bundle)
    root = _safe_new_root(output_dir)
    development_path = (APL_ROOT / config["development_exclusion"]["source_bundle_path"]).resolve(strict=True)
    development_bundle = evidence_builder.load_entity_bundle(development_path, verify_sources=True)
    if development_bundle.get("content_sha256") != config["development_exclusion"]["source_bundle_content_sha256"]:
        raise FormalCorpusError("development exclusion bundle changed from its preregistered content hash")
    molecules, excluded_molecules = dataset_builder._molecule_entities(bundle)
    usable_proteins, excluded_verified_proteins = dataset_builder._protein_entities(bundle)
    supported_proteins, unsupported_by_length = _sequence_applicability(bundle)
    development_molecules, _ = dataset_builder._molecule_entities(development_bundle)
    development_proteins, _ = dataset_builder._protein_entities(development_bundle)
    excluded_proteins = excluded_verified_proteins + [item for item in unsupported_by_length
                                                       if item.get("reason") == "unsupported_length_full_sequence_retained"]
    caps = decision["caps"]
    # The full source bundle is exactly what the strict split builder/auditor would consume.
    # ESM length applicability never removes a sequence from duplicate/neighbor auditing.
    lengths = [len(item["full_sequence"]) for item in usable_proteins]
    development_lengths = [len(item["full_sequence"]) for item in development_proteins]
    molecule_measurement = decision["measurement"]["molecule_neighbor"]
    workload = _workload(len(molecules), lengths, caps, decision["measurement"]["alignment_cells_per_second"],
                         development_lengths, len(development_molecules) + 1,
                         molecule_measurement["warm_median_seconds"], molecule_measurement["cold_first_seconds"])
    status = "size_limit" if (len(molecules) > caps["molecules"] or len(usable_proteins) > caps["targets"]
                              or workload["status"] == "size_limit") else "within_budget"
    report = {
        "schema_version": 1, "protocol_version": config["protocol_version"],
        "status": status, "reason": None if status == "within_budget" else "declared_scope_exceeds_registered_entity_or_alignment_budget",
        "training_ready": False,
        "support_assessment": "not inferred from development-only pilot; formal split support remains unassessed",
        "base_corpus_sha256": config["base_corpus_sha256"], "budget_decision_sha256": config["budget"]["decision_sha256"],
        "entity_bundle": {"path": str(bundle_path), "content_sha256": bundle["content_sha256"],
                          "source_kind": bundle.get("source_kind")},
        "counts": {"source_molecules": len(bundle.get("molecules", [])), "usable_molecules": len(molecules),
                   "excluded_molecules": len(excluded_molecules), "source_proteins": len(bundle.get("proteins", [])),
                   "split_usable_full_sequence_proteins": len(usable_proteins),
                   "length_supported_proteins": len(supported_proteins), "unsupported_or_unverified_proteins": len(excluded_proteins)},
        "sequence_applicability": {"max_full_sequence_residues": MAX_SUPPORTED_RESIDUES,
                                   "supported": supported_proteins, "excluded": excluded_proteins},
        "workload": {"molecule_pairs": workload["molecule_pairs"], "protein_pairs": workload["protein_pairs"],
                     "alignment_cells": workload["alignment_cells"], "max_single_alignment_cells": workload["max_single_alignment_cells"],
                     "development_neighbor_pairs": workload["development_neighbor_pairs"],
                     "development_neighbor_alignment_cells": workload["development_neighbor_alignment_cells"],
                     "development_neighbor_estimated_seconds": workload.get("development_neighbor_estimated_seconds"),
                     "development_molecule_neighbor_pairs": workload["development_molecule_neighbor_pairs"],
                     "development_molecule_neighbor_estimated_seconds": workload.get("development_molecule_neighbor_estimated_seconds"),
                     "eligibility_cluster_molecule_pairs": workload["eligibility_cluster_molecule_pairs"],
                     "eligibility_cluster_protein_pairs": workload["eligibility_cluster_protein_pairs"],
                     "eligibility_cluster_alignment_cells": workload["eligibility_cluster_alignment_cells"],
                     "eligibility_cluster_estimated_seconds": workload.get("eligibility_cluster_estimated_seconds"),
                     "publisher_audit_passes": workload["publisher_audit_passes"],
                     "publisher_audit_each_estimated_seconds": workload["publisher_audit_each_estimated_seconds"],
                     "full_neighbor_pass_count": workload["full_neighbor_pass_count"],
                     "pipeline_total_alignment_cells": workload["pipeline_total_alignment_cells"],
                     "pipeline_total_estimated_seconds": workload["pipeline_total_estimated_seconds"],
                     "alignment_cells_measure": "sum(len(sequence_i)*len(sequence_j)) over each unique full-sequence pair",
                     "builder_estimated_seconds": workload["builder_estimated_seconds"], "auditor_estimated_seconds": workload["auditor_estimated_seconds"],
                     "workload_status": workload["status"], "workload_reason": workload["reason"],
                     "dependencies": {"python": sys.version.split()[0],
                                      "rdkit": __import__("rdkit").__version__,
                                      "biopython": __import__("Bio").__version__},
                     "sequence_length_distribution": {
                         "count": len(lengths), "minimum": min(lengths) if lengths else None,
                         "maximum": max(lengths) if lengths else None,
                         "bins": {"1-256": sum(1 <= value <= 256 for value in lengths),
                                  "257-512": sum(257 <= value <= 512 for value in lengths),
                                  "513-768": sum(513 <= value <= 768 for value in lengths),
                                  "769-1022": sum(769 <= value <= 1022 for value in lengths),
                                  "over-1022": sum(value > 1022 for value in lengths)}},
                     "memory_estimate": {"single_active_pair_upper_bound_bytes": 16 * workload["max_single_alignment_cells"],
                                          "method": "2 float64 dynamic-programming planes times the largest single full-sequence pair; estimate only, not measured peak RSS"},
                     "measurement_source": decision["measurement"]["source"]},
        "caps": dict(caps),
        "measurement": {"preflight_seconds": time.perf_counter() - started,
                        "generated_utc": datetime.now(timezone.utc).isoformat()},
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / "preflight.json").write_bytes(_json_bytes(report))
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight a new formal source-verified corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="calculate all-pairs resource cost before any builder/auditor run")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--entities", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True, help="new absolute 32-character GUID directory")
    expand = commands.add_parser("expand", help="capture the pre-registered bounded formal candidate cohort")
    expand.add_argument("--config", type=Path, required=True)
    expand.add_argument("--out", type=Path, required=True)
    expand.add_argument("--allow-online", action="store_true")
    expand.add_argument("--resume-from", type=Path,
                        help="resume the registered formal-v1 batch cumulatively from its latest GUID ledger root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.config, args.entities, args.out)
            code = 0 if result["status"] == "within_budget" else 1
        else:
            result = expand(args.config, args.out, allow_online=args.allow_online,
                            resume_from=args.resume_from)
            code = 0 if result.get("status") in {"candidate_audited_not_ready", "selection_window_exhausted", "size_limit"} else 1
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
