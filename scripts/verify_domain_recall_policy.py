from __future__ import annotations

import asyncio
import json
from datetime import timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_relevance import memory_relevance_options_from_config
from memory_recall.domain_policy import (
    DOMAIN_POLICY_PUBLISH_CONFIRMATION,
    DomainRecallPolicy,
)
from recall_policy import RecallPolicy


class RequestStub:
    headers: dict[str, str] = {}

    def __init__(self, body: dict | None = None):
        self.body = body or {}

    async def json(self) -> dict:
        return self.body


def scene(domain: str) -> dict:
    return {
        "id": f"scene-{domain}",
        "metadata": {
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "canonical_domain": domain,
            "name": f"{domain} scene",
        },
        "content": "canonical Scene evidence",
    }


def gateway_service(store: DomainRecallPolicy) -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.identity = {
        "ai_name": "Haven",
        "user_name": "小雨",
        "user_display_name": "小雨",
        "relationship_terms": [],
        "user_aliases": [],
    }
    service.recall_policy = RecallPolicy()
    service.high_confidence_semantic_score = 0.72
    service.recall_admission_semantic_score = 0.72
    service.first_card_min_score = 0.55
    service.inject_max_cards = 2
    service.gateway_tz = timezone.utc
    service.config = {
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "explicit_vector_min_score": 0.55,
        }
    }
    service.relevance_options = memory_relevance_options_from_config({})
    service.self_anchor_entry_bucket_id = ""
    service.domain_recall_policy = store
    service._authorize = lambda _authorization: None
    return service


async def verify() -> None:
    with TemporaryDirectory() as directory:
        store = DomainRecallPolicy({"buckets_dir": directory})
        initial = store.dataset_payload()
        assert initial["dataset_version"] == 1
        assert initial["source_kind"] == "seed"
        assert initial["active"] is False
        assert store.policy_for_domain("tech") == "explicit_only"
        assert store.policy_for_domain("relationship") == "normal"

        policies = [dict(item) for item in initial["policies"]]
        for item in policies:
            if item["key"] == "relationship":
                item["policy"] = "excluded"
            if item["key"] == "inner":
                item["policy"] = "explicit_only"
        published = await store.publish_dataset(
            policies=policies,
            expected_dataset_version=1,
            confirmation=DOMAIN_POLICY_PUBLISH_CONFIRMATION,
        )
        assert published["dataset_version"] == 2
        assert published["source_kind"] == "published"
        assert published["active"] is True
        assert store.policy_for_domain("relationship") == "excluded"
        assert store.policy_for_domain("inner") == "explicit_only"

        active_before_conflict = store.active_manifest_path.read_text(encoding="utf-8")
        try:
            await store.publish_dataset(
                policies=policies,
                expected_dataset_version=1,
                confirmation=DOMAIN_POLICY_PUBLISH_CONFIRMATION,
            )
        except ValueError as exc:
            assert str(exc) == "domain_policy_publish_version_conflict:2"
        else:
            raise AssertionError("stale domain policy publisher should be rejected")
        assert store.active_manifest_path.read_text(encoding="utf-8") == active_before_conflict

        service = gateway_service(store)

        excluded_item = {"bucket": scene("relationship"), "semantic_score": 0.99}
        assert service._admit_bucket_for_recall("我们的关系", excluded_item) is False
        assert excluded_item["admission_reason"] == "domain_excluded"

        implicit_item = {"bucket": scene("inner"), "semantic_score": 0.99}
        assert service._admit_bucket_for_recall("我的心事是", implicit_item) is False
        assert implicit_item["admission_reason"] == "domain_explicit_only"

        explicit_item = {
            "bucket": scene("inner"),
            "semantic_score": 0.01,
            "authored_cue_match": True,
            "authored_cue_terms": ["我们的心事"],
        }
        assert service._admit_bucket_for_recall("我们的心事", explicit_item) is True
        assert explicit_item["admission_reason"] == "scene_authored_evidence"

        assert service._canonical_scene_domain_policy_rejection(
            scene("relationship"),
            explicit_id=True,
        )["reason"] == "domain_excluded"
        assert service._canonical_scene_domain_policy_rejection(
            scene("inner"),
            explicit_id=True,
        ) is None
        inferred_only = scene("general")
        inferred_only["metadata"].pop("canonical_domain")
        inferred_only["metadata"]["domain"] = ["tech"]
        inferred_only["metadata"]["tags"] = ["tech"]
        assert service._canonical_scene_domain_policy(inferred_only) == ("general", "normal")

        read_response = await service.handle_domain_recall_policies(RequestStub())
        assert read_response.status_code == 200
        assert json.loads(read_response.body)["dataset_version"] == 2
        conflict_response = await service.handle_domain_recall_policy_publish(
            RequestStub(
                {
                    "expected_dataset_version": 1,
                    "confirm": DOMAIN_POLICY_PUBLISH_CONFIRMATION,
                    "policies": policies,
                }
            )
        )
        assert conflict_response.status_code == 409
        assert json.loads(conflict_response.body)["error"] == "domain_policy_publish_version_conflict:2"


asyncio.run(verify())
print("domain recall policy atomic publish verification passed")
