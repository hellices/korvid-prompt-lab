from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest

from korvid_prompt_lab.aks import AKSPortForwardError, AKSPreflightTransientError
from korvid_prompt_lab.contracts import KorvidUpstreamServing
from korvid_prompt_lab.experiment_budget import ExperimentBudget
from korvid_prompt_lab.experiment_config import (
    AKSConnection,
    ExperimentSpec,
    LoopbackConnection,
    ModelProfile,
    ReflectionProfile,
)
from korvid_prompt_lab.model_session import ModelSessionError, model_session
from korvid_prompt_lab.upstream_contract import KORVID_REVISION

TARGET_DIGEST = f"sha256:{'1' * 64}"
REFLECTION_DIGEST = f"sha256:{'2' * 64}"


def _spec(serving: LoopbackConnection | AKSConnection) -> ExperimentSpec:
    model = ModelProfile(
        reference="ollama/qwen3:0.6b",
        digest=TARGET_DIGEST,
        options={"native_thinking": True, "think": False, "num_ctx": 16384},
    )
    return ExperimentSpec(
        campaign_id="unified-native",
        runtime=KorvidUpstreamServing(
            backend="korvid_upstream",
            source_root="/reviewed/korvid",
            base_url="",
            korvid_revision=KORVID_REVISION,
            timeout_seconds=240,
            model_options=model.options,
        ),
        serving=serving,
        model=model,
        reflection=ReflectionProfile(
            reference="ollama_chat/qwen3:14b",
            digest=REFLECTION_DIGEST,
            timeout_seconds=180,
            options={"num_ctx": 4096, "max_tokens": 512},
        ),
        repetitions=5,
        evaluation_seed=0,
        stages=(),
        total_metric_calls=96,
        max_evaluations=256,
        max_proposals=12,
        wall_clock_seconds=3600,
        stagnation_attempt_limit=3,
        case_splits={
            "train": ("scenarios/healthy-deployment",),
            "validation": ("journeys/tui-follow",),
            "holdout": ("journeys/compare-namespaces",),
        },
    )


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHttpClient:
    instances: ClassVar[list[FakeHttpClient]] = []
    payloads: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.urls: list[str] = []
        self._payloads = list(self.payloads)
        self.closed = False
        self.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def get(self, url: str) -> FakeResponse:
        self.urls.append(url)
        if not self._payloads:
            raise AssertionError(f"unexpected GET {url}")
        return FakeResponse(self._payloads.pop(0))


def _model_payload() -> dict[str, Any]:
    return {
        "models": [
            {"name": "qwen3:0.6b", "digest": "1" * 64},
            {"name": "qwen3:14b", "digest": REFLECTION_DIGEST},
        ]
    }


def test_loopback_session_verifies_both_digests_and_persists_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHttpClient.instances.clear()
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    spec = _spec(LoopbackConnection("loopback", "http://127.0.0.1:11434"))

    with model_session(spec, tmp_path) as resolved:
        assert resolved.base_url == "http://127.0.0.1:11434"
        assert resolved.evidence["models"]["target"]["verified"] is True
        assert resolved.evidence["models"]["reflection"]["verified"] is True
        assert resolved.evidence["api_version"] == "0.11.4"
        FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
        assert resolved.verify()["models"]["target"]["verified"] is True

    assert len(FakeHttpClient.instances) == 2
    assert all(client.kwargs["trust_env"] is False for client in FakeHttpClient.instances)
    assert all(client.kwargs["follow_redirects"] is False for client in FakeHttpClient.instances)
    assert all(client.closed for client in FakeHttpClient.instances)
    artifact = json.loads((tmp_path / "model-session.json").read_text())
    assert artifact["status"] == "closed"
    assert artifact["connection"]["owned_process"] is False


def test_digest_mismatch_fails_before_session_is_yielded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHttpClient.instances.clear()
    FakeHttpClient.payloads = [
        {
            "models": [
                {"name": "qwen3:0.6b", "digest": "f" * 64},
                {"name": "qwen3:14b", "digest": "2" * 64},
            ]
        },
        {"version": "0.11.4"},
    ]
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)

    with (
        pytest.raises(ModelSessionError, match="digest mismatch"),
        model_session(
            _spec(LoopbackConnection("loopback", "http://127.0.0.1:11434")),
            tmp_path,
        ),
    ):
        pytest.fail("invalid model digest must fail before inference")


class FakeForward:
    instances: ClassVar[list[FakeForward]] = []
    enter_effects: ClassVar[list[BaseException | None]] = []

    def __init__(self, serving, *, workspace_dir: Path, **kwargs: Any) -> None:
        self.serving = serving
        self.workspace_dir = workspace_dir
        self.kwargs = kwargs
        self.base_url = "http://127.0.0.1:43123"
        self.closed = False
        self.instances.append(self)

    def __enter__(self) -> Self:
        effect = self.enter_effects.pop(0) if self.enter_effects else None
        if effect is not None:
            self.closed = True
            raise effect
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


def test_aks_session_composes_native_runtime_and_reuses_one_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.instances.clear()
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    spec = _spec(
        AKSConnection(
            "aks_port_forward", "rg", "aks", "models", "ollama", node_pool=None
        )
    )

    with model_session(spec, tmp_path) as resolved:
        from korvid_prompt_lab.upstream_contract import SourceCase
        monkeypatch.setattr(
            "korvid_prompt_lab.upstream_contract.load_source_cases",
            lambda root, refs: tuple(SourceCase(
                reference=ref, case_id=ref.split("/")[1],
                kind="scenario" if ref.startswith("scenarios/") else "journey", prompt="Unit metadata fixture",
                source_path=f"src/korvid/evals/{ref}.yaml",
                source_sha256="1" * 64, questions=("Unit metadata fixture",),
            ) for ref in refs),
        )
        campaign = spec.campaign(resolved.base_url)
        assert isinstance(campaign.serving, KorvidUpstreamServing)
        assert campaign.serving.base_url == "http://127.0.0.1:43123"
        assert dict(campaign.serving.model_options)["native_thinking"] is True
        assert len(FakeForward.instances) == 1
        assert FakeForward.instances[0].serving.model == "qwen3:0.6b"
        assert not hasattr(FakeForward.instances[0].serving, "command")
        assert not FakeForward.instances[0].closed

    assert FakeForward.instances[0].closed


def test_aks_transient_preflight_is_retried_and_failed_listener_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = [AKSPreflightTransientError("warming"), None]
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    monkeypatch.setattr("korvid_prompt_lab.model_session.time.sleep", lambda _: None)

    with model_session(
        _spec(AKSConnection("aks_port_forward", "rg", "aks", "models", "ollama")),
        tmp_path,
    ):
        pass

    assert len(FakeForward.instances) == 2
    assert all(forward.closed for forward in FakeForward.instances)


@dataclass
class Completed:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def test_owned_zero_node_pool_is_restored_when_evaluation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    results = iter(
        [
            Completed(stdout="0\n"),
            Completed(),
            Completed(),
            Completed(stdout="0\n"),
        ]
    )
    calls: list[tuple[str, ...]] = []

    def run(args, **kwargs):
        calls.append(tuple(args))
        return next(results)

    monkeypatch.setattr("korvid_prompt_lab.model_session.subprocess.run", run)
    spec = _spec(
        AKSConnection(
            "aks_port_forward", "rg", "aks", "models", "ollama", node_pool="eval"
        )
    )

    with (
        pytest.raises(RuntimeError, match="evaluation failed"),
        model_session(spec, tmp_path, allow_capacity_changes=True),
    ):
        raise RuntimeError("evaluation failed")

    scale_counts = [
        call[call.index("--node-count") + 1] for call in calls if "scale" in call
    ]
    assert scale_counts == ["1", "0"]
    assert FakeForward.instances[0].closed
    evidence = json.loads((tmp_path / "model-session.json").read_text())
    assert evidence["node_pool"]["original_count"] == 0
    assert evidence["node_pool"]["final_count"] == 0
    assert evidence["status"] == "failed"


def test_zero_node_pool_requires_explicit_capacity_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "korvid_prompt_lab.model_session.subprocess.run",
        lambda *args, **kwargs: Completed(stdout="0\n"),
    )
    FakeForward.instances.clear()
    with (
        pytest.raises(PermissionError, match="allow_capacity_changes"),
        model_session(
            _spec(
                AKSConnection(
                    "aks_port_forward",
                    "rg",
                    "aks",
                    "models",
                    "ollama",
                    node_pool="eval",
                )
            ),
            tmp_path,
        ),
    ):
        pytest.fail("capacity permission is required before starting a forward")
    assert FakeForward.instances == []


def test_running_node_pool_is_never_resized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    results = iter([Completed(stdout="2\n"), Completed(stdout="2\n")])
    calls: list[tuple[str, ...]] = []

    def run(args, **kwargs):
        calls.append(tuple(args))
        return next(results)

    monkeypatch.setattr("korvid_prompt_lab.model_session.subprocess.run", run)
    with model_session(
        _spec(
            AKSConnection(
                "aks_port_forward",
                "rg",
                "aks",
                "models",
                "ollama",
                node_pool="eval",
            )
        ),
        tmp_path,
        allow_capacity_changes=True,
    ):
        pass

    assert not any("scale" in call for call in calls)
    evidence = json.loads((tmp_path / "model-session.json").read_text())
    assert evidence["node_pool"]["original_count"] == 2
    assert evidence["node_pool"]["final_count"] == 2
    assert evidence["node_pool"]["capacity_changed"] is False


def test_restore_failure_is_surfaced_without_hiding_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    results = iter(
        [
            Completed(stdout="0\n"),
            Completed(),
            Completed(returncode=1),
            Completed(stdout="1\n"),
        ]
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.model_session.subprocess.run",
        lambda *args, **kwargs: next(results),
    )

    with (
        pytest.raises(BaseExceptionGroup) as raised,
        model_session(
            _spec(
                AKSConnection(
                    "aks_port_forward",
                    "rg",
                    "aks",
                    "models",
                    "ollama",
                    node_pool="eval",
                )
            ),
            tmp_path,
            allow_capacity_changes=True,
        ),
    ):
        raise RuntimeError("primary evaluation failure")

    assert any(
        isinstance(error, RuntimeError)
        and str(error) == "primary evaluation failure"
        for error in raised.value.exceptions
    )
    assert any(
        isinstance(error, RuntimeError)
        and str(error) == "AKS node pool command failed"
        for error in raised.value.exceptions
    )
    assert FakeForward.instances[0].closed


def test_permanent_aks_preflight_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = [AKSPortForwardError("permanent")]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)

    with (
        pytest.raises(AKSPortForwardError, match="permanent"),
        model_session(
            _spec(AKSConnection("aks_port_forward", "rg", "aks", "models", "ollama")),
            tmp_path,
        ),
    ):
        pytest.fail("permanent failure must not be retried")

    assert len(FakeForward.instances) == 1


def test_scale_timeout_still_restores_owned_zero_node_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    show_results = iter(["0\n", "0\n"])

    def run(args, **kwargs):
        call = tuple(args)
        calls.append(call)
        if "show" in call:
            return Completed(stdout=next(show_results))
        if call[call.index("--node-count") + 1] == "1":
            raise subprocess.TimeoutExpired(call, timeout=600)
        return Completed()

    monkeypatch.setattr("korvid_prompt_lab.model_session.subprocess.run", run)

    with (
        pytest.raises(ModelSessionError, match="node pool"),
        model_session(
            _spec(
                AKSConnection(
                    "aks_port_forward",
                    "rg",
                    "aks",
                    "models",
                    "ollama",
                    node_pool="eval",
                )
            ),
            tmp_path,
            allow_capacity_changes=True,
        ),
    ):
        pytest.fail("a timed-out capacity change must not start evaluation")

    scale_counts = [
        call[call.index("--node-count") + 1] for call in calls if "scale" in call
    ]
    assert scale_counts == ["1", "0"]


def test_session_cannot_close_when_owned_node_pool_was_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    results = iter(
        [
            Completed(stdout="0\n"),
            Completed(),
            Completed(),
            Completed(stdout="1\n"),
        ]
    )
    monkeypatch.setattr(
        "korvid_prompt_lab.model_session.subprocess.run",
        lambda *args, **kwargs: next(results),
    )

    with (
        pytest.raises(ModelSessionError, match="restor|original"),
        model_session(
            _spec(
                AKSConnection(
                    "aks_port_forward",
                    "rg",
                    "aks",
                    "models",
                    "ollama",
                    node_pool="eval",
                )
            ),
            tmp_path,
            allow_capacity_changes=True,
        ),
    ):
        pass

    evidence = json.loads((tmp_path / "model-session.json").read_text())
    assert evidence["node_pool"]["final_count"] == 1
    assert evidence["status"] != "closed"


def test_verify_records_rechecks_and_rejects_ollama_version_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHttpClient.instances.clear()
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    spec = _spec(LoopbackConnection("loopback", "http://127.0.0.1:11434"))

    with model_session(spec, tmp_path) as resolved:
        FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
        resolved.verify()
        assert resolved.evidence["rechecks"][-1]["api_version"] == "0.11.4"

        FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.5"}]
        with pytest.raises(ModelSessionError, match="version"):
            resolved.verify()
        assert resolved.evidence["rechecks"][-1]["api_version"] == "0.11.5"


def test_verify_records_digest_mismatch_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    spec = _spec(LoopbackConnection("loopback", "http://127.0.0.1:11434"))

    with model_session(spec, tmp_path) as resolved:
        FakeHttpClient.payloads = [{
            "models": [
                {"name": "qwen3:0.6b", "digest": "f" * 64},
                {"name": "qwen3:14b", "digest": "2" * 64},
            ]
        }]
        with pytest.raises(ModelSessionError, match="digest mismatch"):
            resolved.verify()
        assert resolved.evidence["rechecks"][-1]["status"] == "failed"
        assert "digest mismatch" in resolved.evidence["rechecks"][-1]["error"]


def test_budget_clips_bootstrap_subprocess_http_and_forward_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeForward.instances.clear()
    FakeForward.enter_effects = []
    FakeHttpClient.instances.clear()
    FakeHttpClient.payloads = [_model_payload(), {"version": "0.11.4"}]
    monkeypatch.setattr("korvid_prompt_lab.model_session.AKSPortForward", FakeForward)
    monkeypatch.setattr("korvid_prompt_lab.model_session.httpx.Client", FakeHttpClient)
    subprocess_timeouts: list[float] = []
    show_results = iter(["2\n", "2\n"])

    def run(args, **kwargs):
        subprocess_timeouts.append(kwargs["timeout"])
        return Completed(stdout=next(show_results))

    monkeypatch.setattr("korvid_prompt_lab.model_session.subprocess.run", run)
    budget = ExperimentBudget(10, 10, 0.25, clock=lambda: 100.0)
    spec = _spec(
        AKSConnection(
            "aks_port_forward", "rg", "aks", "models", "ollama", node_pool="eval"
        )
    )

    with model_session(spec, tmp_path, budget=budget):
        pass

    assert subprocess_timeouts[0] <= 0.25
    assert subprocess_timeouts[-1] == 30
    assert FakeForward.instances[0].kwargs["port_forward_ready_timeout_seconds"] <= 0.25
    assert FakeHttpClient.instances[0].kwargs["timeout"].connect <= 0.25
