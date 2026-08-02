from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from memory_metadata import CANONICAL_DOMAINS


DOMAIN_POLICY_SCHEMA_VERSION = 1
DOMAIN_POLICY_PUBLISH_CONFIRMATION = "PUBLISH_DOMAIN_RECALL_POLICIES"
DOMAIN_RECALL_POLICIES = frozenset({"normal", "explicit_only", "excluded"})
DEFAULT_DOMAIN_POLICIES = {
    domain: "explicit_only" if domain == "tech" else "normal"
    for domain in sorted(CANONICAL_DOMAINS)
}


class DomainRecallPolicy:
    """Versioned, atomically activated recall policy for canonical Scene domains."""

    def __init__(self, config: dict[str, Any]):
        project_dir = Path(__file__).resolve().parent.parent
        state_dir = Path(str(config.get("buckets_dir") or project_dir)).resolve()
        gateway_cfg = config.get("gateway", {})
        gateway_cfg = gateway_cfg if isinstance(gateway_cfg, dict) else {}
        cfg = gateway_cfg.get("domain_recall_policy", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        configured_dir = str(cfg.get("publish_dir") or "").strip()
        self.publish_dir = (
            Path(configured_dir).expanduser().resolve()
            if configured_dir
            else state_dir / "domain_recall_policies"
        )
        self.active_manifest_path = self.publish_dir / "active.json"
        self._publish_lock = asyncio.Lock()
        self._loaded_payload: dict[str, Any] | None = None
        self._loaded_signature: tuple[str, int] | None = None

    @staticmethod
    def _seed_payload() -> dict[str, Any]:
        return {
            "schema_version": DOMAIN_POLICY_SCHEMA_VERSION,
            "dataset_version": 1,
            "policies": [
                {"key": key, "policy": policy}
                for key, policy in DEFAULT_DOMAIN_POLICIES.items()
            ],
        }

    @staticmethod
    def _normalize_policies(policies: Any) -> list[dict[str, str]]:
        if not isinstance(policies, list):
            raise ValueError("domain_policy_list_missing")
        normalized: dict[str, str] = {}
        for item in policies:
            if not isinstance(item, dict):
                raise ValueError("domain_policy_item_invalid")
            key = str(item.get("key") or "").strip().lower()
            policy = str(item.get("policy") or "").strip().lower()
            if key not in CANONICAL_DOMAINS:
                raise ValueError(f"domain_policy_unknown_domain:{key}")
            if key in normalized:
                raise ValueError(f"domain_policy_duplicate_domain:{key}")
            if policy not in DOMAIN_RECALL_POLICIES:
                raise ValueError(f"domain_policy_invalid_policy:{key}")
            normalized[key] = policy
        missing = sorted(CANONICAL_DOMAINS - set(normalized))
        if missing:
            raise ValueError(f"domain_policy_missing_domains:{','.join(missing)}")
        return [
            {"key": key, "policy": normalized[key]}
            for key in sorted(CANONICAL_DOMAINS)
        ]

    @classmethod
    def _validate_payload(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("domain_policy_payload_invalid")
        if int(payload.get("schema_version") or 0) != DOMAIN_POLICY_SCHEMA_VERSION:
            raise RuntimeError("domain_policy_schema_mismatch")
        version = int(payload.get("dataset_version") or 0)
        if version < 1:
            raise RuntimeError("domain_policy_version_invalid")
        return {
            "schema_version": DOMAIN_POLICY_SCHEMA_VERSION,
            "dataset_version": version,
            "policies": cls._normalize_policies(payload.get("policies")),
        }

    def _active_source_path(self) -> tuple[Path | None, dict[str, Any] | None]:
        if not self.active_manifest_path.exists():
            return None, None
        try:
            manifest = json.loads(self.active_manifest_path.read_text(encoding="utf-8"))
            generation = str(manifest.get("generation") or "").strip()
            if not generation or Path(generation).name != generation:
                raise RuntimeError("domain_policy_manifest_invalid")
            generation_dir = (self.publish_dir / generation).resolve()
            if generation_dir.parent != self.publish_dir.resolve():
                raise RuntimeError("domain_policy_manifest_invalid")
            return generation_dir / "source.json", manifest
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("domain_policy_manifest_invalid") from exc

    def _load_payload(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        source_path, manifest = self._active_source_path()
        if source_path is None:
            return self._seed_payload(), None
        if not source_path.exists():
            raise RuntimeError("domain_policy_source_missing")
        signature = (str(source_path), source_path.stat().st_mtime_ns)
        if self._loaded_payload is None or self._loaded_signature != signature:
            try:
                raw = json.loads(source_path.read_text(encoding="utf-8"))
                self._loaded_payload = self._validate_payload(raw)
                self._loaded_signature = signature
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(str(exc) or "domain_policy_source_invalid") from exc
        return dict(self._loaded_payload), manifest

    def dataset_payload(self) -> dict[str, Any]:
        payload, manifest = self._load_payload()
        return {
            "ok": True,
            **payload,
            "active": manifest is not None,
            "deployment_state": "production",
            "source_kind": "published" if manifest else "seed",
            "published": manifest,
        }

    @property
    def active(self) -> bool:
        return self.active_manifest_path.exists()

    def policy_for_domain(self, domain: object) -> str:
        key = str(domain or "general").strip().lower()
        if key not in CANONICAL_DOMAINS:
            key = "general"
        payload, _manifest = self._load_payload()
        policies = {
            str(item.get("key") or ""): str(item.get("policy") or "normal")
            for item in payload["policies"]
        }
        return policies.get(key, "normal")

    async def publish_dataset(
        self,
        *,
        policies: Any,
        expected_dataset_version: int,
        confirmation: str,
    ) -> dict[str, Any]:
        if confirmation != DOMAIN_POLICY_PUBLISH_CONFIRMATION:
            raise ValueError("domain_policy_publish_confirmation_required")
        async with self._publish_lock:
            current = self.dataset_payload()
            current_version = int(current["dataset_version"])
            if int(expected_dataset_version) != current_version:
                raise ValueError(f"domain_policy_publish_version_conflict:{current_version}")
            normalized = self._normalize_policies(policies)
            next_version = current_version + 1
            source_payload = {
                "schema_version": DOMAIN_POLICY_SCHEMA_VERSION,
                "dataset_version": next_version,
                "policies": normalized,
            }
            self.publish_dir.mkdir(parents=True, exist_ok=True)
            staging_dir = self.publish_dir / f".staging-{uuid4().hex}"
            staging_dir.mkdir()
            generation_dir: Path | None = None
            activated = False
            try:
                source_path = staging_dir / "source.json"
                source_path.write_text(
                    json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._validate_payload(json.loads(source_path.read_text(encoding="utf-8")))
                generation = f"v{next_version:06d}-{uuid4().hex[:12]}"
                generation_dir = self.publish_dir / generation
                staging_dir.replace(generation_dir)
                manifest = {
                    "schema_version": 1,
                    "dataset_version": next_version,
                    "generation": generation,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "policy_count": len(normalized),
                }
                manifest_tmp = self.publish_dir / f".active-{uuid4().hex}.tmp"
                manifest_tmp.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                manifest_tmp.replace(self.active_manifest_path)
                activated = True
                self._loaded_payload = None
                self._loaded_signature = None
                return self.dataset_payload()
            except Exception:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                if not activated and generation_dir is not None and generation_dir.exists():
                    shutil.rmtree(generation_dir, ignore_errors=True)
                raise
