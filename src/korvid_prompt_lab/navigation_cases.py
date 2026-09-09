"""Small UI-assistance tasks, deliberately separate from diagnostic evals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NavigationCase:
    case_id: str
    split: str
    prompt: str
    namespace: str
    pod: str
    release: str
    initial_kind: str
    initial_scope: str
    initial_filter: str
    expected: tuple[tuple[str, str], ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {"case": asdict(self), "objects": fixture_objects(self)},
                sort_keys=True, ensure_ascii=False,
            ).encode()
        ).hexdigest()


def navigation_cases() -> tuple[NavigationCase, ...]:
    cases: list[NavigationCase] = []
    for split, namespace, pod, release in (
        ("train", "shop", "checkout-1", "store"),
        ("validation", "monitoring", "metrics-2", "telemetry"),
        ("holdout", "sandbox", "worker-3", "queue"),
    ):
        prompts = {
            "train": (
                f"{namespace} 네임스페이스의 파드 목록을 보여줘.",
                f"Open the Helm releases screen in namespace {namespace}.",
                "모든 네임스페이스의 파드 목록으로 이동해줘.",
                f"Show only {pod} in the current pod list.",
                "현재 파드 목록의 필터를 해제해줘.",
                f"Open the logs for pod {pod} in namespace {namespace}. Do not diagnose it.",
                f"{namespace}에 있는 {pod} 파드의 describe 화면을 열어줘.",
                f"Show revision history for Helm release {release} in namespace {namespace}.",
            ),
            "validation": (
                f"Show the pods in namespace {namespace}.",
                f"{namespace}의 Helm 릴리스 화면을 열어줘.",
                "Show pods across all namespaces.",
                f"현재 파드 목록에서 {pod}만 보이게 필터링해줘.",
                "Clear the filter on the current pod list.",
                f"{namespace}의 {pod} 파드 로그 화면만 열어줘. 원인 분석은 하지 마.",
                f"Open the describe screen for pod {pod} in namespace {namespace}.",
                f"{namespace}의 Helm 릴리스 {release} 리비전 이력으로 이동해줘.",
            ),
            "holdout": (
                f"Pod 화면을 {namespace} 범위로 전환해줘.",
                f"Take me to the installed Helm releases under {namespace}.",
                "네임스페이스 제한 없이 전체 Pod를 볼 수 있게 바꿔줘.",
                f"Filter this pod table by the name {pod}.",
                "검색 필터를 없애서 현재 파드 목록 전체가 보이게 해줘.",
                f"Let me watch {namespace}/{pod} logs on screen; no troubleshooting.",
                f"{namespace}/{pod} 매니페스트를 상세 화면에서 보고 싶어.",
                f"Navigate into the revisions of {release}, the Helm release in {namespace}.",
            ),
        }[split]
        for task, prompt in zip(
            ("pods", "helm", "all-pods", "filter", "clear", "logs", "describe", "history"),
            prompts,
            strict=True,
        ):
            initial_kind = "deployments" if task in {"pods", "all-pods"} else "pods"
            initial_scope = "default" if task in {"pods", "helm", "history"} else namespace
            expected = {"kind": initial_kind, "scope": initial_scope, "filter": ""}
            if task in {"pods", "helm", "all-pods", "history"}:
                expected.update(
                    kind={"helm": "helmreleases", "history": "helmrevisions"}.get(task, "pods"),
                    scope="*" if task == "all-pods" else namespace,
                )
            if task == "filter":
                expected["filter"] = pod
            if task == "history":
                expected["drill_parent"] = f"helm:{namespace}/{release}"
            if task == "logs":
                expected["logs"] = f"{namespace}/{pod}/main"
            if task == "describe":
                expected["describe"] = f"pods/{namespace}/{pod}"
            cases.append(
                NavigationCase(
                    case_id=f"{split}-{task}",
                    split=split,
                    prompt=prompt,
                    namespace=namespace,
                    pod=pod,
                    release=release,
                    initial_kind=initial_kind,
                    initial_scope=initial_scope,
                    initial_filter="does-not-match" if task == "clear" else "",
                    expected=tuple(sorted(expected.items())),
                )
            )
    return tuple(cases)


def find_navigation_case(case_id: str) -> NavigationCase:
    for case in navigation_cases():
        if case.case_id == case_id:
            return case
    raise ValueError(f"unknown navigation case: {case_id}")


def require_complete_navigation_pack(case_ids: Sequence[str]) -> None:
    expected = {case.case_id for case in navigation_cases()}
    if len(case_ids) != len(expected) or set(case_ids) != expected:
        raise ValueError("navigation campaign requires the complete authored 24-case pack")


def fixture_objects(case: NavigationCase) -> tuple[dict[str, Any], ...]:
    objects: list[dict[str, Any]] = []
    for namespace in (case.namespace, "other"):
        objects.extend(
            [
                {
                    "apiVersion": "v1", "kind": "Pod",
                    "metadata": {
                        "name": case.pod, "namespace": namespace,
                        "uid": f"pod-{namespace}-{case.pod}",
                    },
                    "spec": {"containers": [{"name": "main", "image": "fixture:1"}]},
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [{
                            "name": "main", "ready": True, "restartCount": 0,
                            "state": {"running": {}},
                        }],
                    },
                },
                {
                    "apiVersion": "apps/v1", "kind": "Deployment",
                    "metadata": {"name": "frontend", "namespace": namespace, "uid": f"deploy-{namespace}"},
                    "spec": {"replicas": 1},
                    "status": {"replicas": 1, "readyReplicas": 1},
                },
            ]
        )
        for revision in (1, 2):
            objects.append(
                {
                    "apiVersion": "v1", "kind": "Secret", "type": "helm.sh/release.v1",
                    "metadata": {
                        "name": f"sh.helm.release.v1.{case.release}.v{revision}",
                        "namespace": namespace, "uid": f"helm-{namespace}-{revision}",
                        "labels": {
                            "name": case.release, "version": str(revision),
                            "status": "deployed", "owner": "helm",
                        },
                    },
                }
            )
    return tuple(objects)
