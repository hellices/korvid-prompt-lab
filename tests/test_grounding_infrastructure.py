"""Offline contract tests for the Prompt Lab ARC runner and grounding-access files."""
from __future__ import annotations

import base64
import json
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / "infra/arc/prompt-lab-runners-values.yaml"
SERVICE_ACCOUNT = ROOT / "infra/arc/prompt-lab-runner-service-account.yaml"
RUNNER_DOCKERFILE = ROOT / "infra/arc/runner/Dockerfile"


def test_prompt_lab_runner_values_are_repo_scoped_and_serial() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    assert values["githubConfigUrl"] == "https://github.com/hellices/korvid-prompt-lab"
    assert values["githubConfigSecret"] == "prompt-lab-runners-github-app"
    assert values["runnerScaleSetName"] == "prompt-lab-runners"
    assert values["minRunners"] == 0
    assert values["maxRunners"] == 1


def test_prompt_lab_runners_cannot_schedule_on_model_compute() -> None:
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    pod = values["template"]["spec"]
    assert pod["serviceAccountName"] == "prompt-lab-runners-no-permission"
    assert pod["automountServiceAccountToken"] is False
    assert pod["nodeSelector"] == {"workload": "gha-runner"}
    assert pod["tolerations"] == [
        {
            "key": "gha-runner",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    assert all(item["key"] != "workload" for item in pod["tolerations"])
    assert pod["containers"][0]["image"] == (
        "acrpensionguard.azurecr.io/runner-base:prompt-lab-v1"
    )


def test_runner_service_account_is_tokenless_and_role_free() -> None:
    docs = list(yaml.safe_load_all(SERVICE_ACCOUNT.read_text(encoding="utf-8")))
    assert [doc["kind"] for doc in docs] == ["Namespace", "ServiceAccount"]
    assert docs[0]["metadata"]["name"] == "arc-runners-prompt-lab"
    assert docs[1]["metadata"]["namespace"] == "arc-runners-prompt-lab"
    assert docs[1]["automountServiceAccountToken"] is False


def test_runner_container_runs_non_root() -> None:
    """Fix 1: container securityContext must enforce non-root execution."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    sc = values["template"]["spec"]["containers"][0]["securityContext"]
    assert sc["runAsNonRoot"] is True
    assert sc["allowPrivilegeEscalation"] is False


def test_controller_service_account_cross_namespace_discovery() -> None:
    """Fix 2: explicit controllerServiceAccount avoids cross-namespace discovery failure."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    csa = values["controllerServiceAccount"]
    assert csa["name"] == "arc-gha-rs-controller"
    assert csa["namespace"] == "arc-systems"


def test_runner_container_security_context_has_numeric_uid_gid() -> None:
    """Task 1 cannot-verify fix: add runAsUser/runAsGroup so Kubernetes can
    enforce non-root numerically even when the image USER is the string 'runner'."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    sc = values["template"]["spec"]["containers"][0]["securityContext"]
    assert sc["runAsUser"] == 1001
    assert sc["runAsGroup"] == 1001


def test_prompt_lab_runner_image_pins_required_tools_and_non_root_user() -> None:
    body = RUNNER_DOCKERFILE.read_text(encoding="utf-8")
    assert body.startswith(
        "FROM ghcr.io/astral-sh/uv:0.10.9 AS uv\n"
        "FROM acrpensionguard.azurecr.io/runner-base:v1"
    )
    assert "--client-version v1.35.6" in body
    assert "--kubectl-version" not in body
    assert "--kubelogin-version v0.2.19" in body
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in body
    assert body.rstrip().endswith("USER runner")


# ---------------------------------------------------------------------------
# Task 2: Grounding-round workflow routes to the Prompt Lab runner scale set
# ---------------------------------------------------------------------------
WORKFLOW = ROOT / ".github/workflows/grounding-round.yml"


def test_grounding_workflow_routes_to_prompt_lab_runners() -> None:
    """The grounding-round job must run on prompt-lab-runners, not korvid-runners."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    runs_on = wf["jobs"]["grounding"]["runs-on"]
    assert runs_on == "prompt-lab-runners", (
        f"Expected 'prompt-lab-runners' but got {runs_on!r}. "
        "Update .github/workflows/grounding-round.yml runs-on."
    )


def test_grounding_workflow_preserves_environment_and_concurrency() -> None:
    """Environment and concurrency settings must be preserved after runner change."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert wf["jobs"]["grounding"]["environment"] == "aks-grounding"
    assert wf["jobs"]["grounding"]["timeout-minutes"] == 180
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_grounding_workflow_preserves_permissions() -> None:
    """Top-level permissions must remain unchanged."""
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    perms = wf["permissions"]
    assert perms["contents"] == "read"
    assert perms["id-token"] == "write"
    assert perms["pull-requests"] == "write"


# ---------------------------------------------------------------------------
# Task 3: least-privilege grounding access bootstrap
#
# The access bootstrap is a deployment tool, so grepping its source proves
# almost nothing: a script can contain every required string and still create
# the wrong scope, re-create an identity on every run, or leak a private key.
# These tests therefore execute the real script against strict fake ``az`` and
# ``gh`` executables that keep state on disk, refuse commands the contract does
# not allow, and record every argv, payload file, and stdin byte they receive.
# ---------------------------------------------------------------------------
ACCESS_SCRIPT = ROOT / "scripts/configure-grounding-access.sh"
RBAC_TEMPLATE = ROOT / "infra/azure/grounding-kubernetes-role.json.tpl"

RETIRED_ACCESS_PATHS = (
    ROOT / "scripts/setup-oidc-federation.sh",
    ROOT / "scripts/setup-azure-roles.sh",
    ROOT / "scripts/setup-github-environment.sh",
    ROOT / "infra/azure/prompt-lab-k8s-data-role.json",
    ROOT / "tests/test_oidc_and_roles.py",
)

SUBSCRIPTION_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "22222222-2222-2222-2222-222222222222"
NEW_APP_ID = "33333333-3333-3333-3333-333333333333"
NEW_APP_OBJECT_ID = "44444444-4444-4444-4444-444444444444"
EXISTING_APP_ID = "55555555-5555-5555-5555-555555555555"
EXISTING_APP_OBJECT_ID = "66666666-6666-6666-6666-666666666666"
SP_OBJECT_ID = "77777777-7777-7777-7777-777777777777"
GH_USER_ID = "4242"
KORVID_APP_ID = "987654"

AKS_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/rg-pension-guard"
    "/providers/Microsoft.ContainerService/managedClusters/aks-shared-runners"
)
NODE_POOL_ID = f"{AKS_ID}/agentPools/modeleval"
NAMESPACE_SCOPE = f"{AKS_ID}/namespaces/ollama"
SUBSCRIPTION_SCOPE = f"/subscriptions/{SUBSCRIPTION_ID}"

KUBERNETES_ROLE_NAME = "Korvid Prompt Lab Grounding Kubernetes Access"
CLUSTER_USER_ROLE = "Azure Kubernetes Service Cluster User Role"

FEDERATED_NAME = "github-aks-grounding"
FEDERATED_ISSUER = "https://token.actions.githubusercontent.com"
FEDERATED_SUBJECT = "repo:hellices/korvid-prompt-lab:environment:aks-grounding"
FEDERATED_AUDIENCE = "api://AzureADTokenExchange"

PRIVATE_KEY_MARKER = "korvid-app-private-key-marker"
PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    f"{PRIVATE_KEY_MARKER}\n"
    "-----END RSA PRIVATE KEY-----\n"
)
REFLECTION_CREDENTIAL_MARKER = "reflection-credential-marker"

REQUIRED_ENVIRONMENT_VARIABLES = {
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "KORVID_APP_ID",
    "KORVID_AKS_NAMESPACE",
    "KORVID_AKS_SERVICE",
}

_JQ = shutil.which("jq")

#: Shared shim prelude.  Every fake records the full argv, the mode and content
#: of any ``@file``/``--input`` payload, and honours ``fail-<key>`` markers so a
#: test can prove a real CLI failure propagates instead of being swallowed.
_FAKE_PRELUDE = r"""#!/usr/bin/env bash
set -Eeuo pipefail

TOOL="$(basename "$0")"
state="${FAKE_STATE_DIR:?FAKE_STATE_DIR is required}"
calls="${state}/calls.jsonl"

die() {
  printf 'fake %s: %s\n' "$TOOL" "$*" >&2
  exit 90
}

next_seq() {
  local seq_file="${state}/seq" n=0
  if [[ -f "$seq_file" ]]; then
    n="$(cat "$seq_file")"
  fi
  n=$((n + 1))
  printf '%s' "$n" >"$seq_file"
  printf '%04d' "$n"
}

file_mode() {
  stat -f '%Lp' "$1" 2>/dev/null || stat -c '%a' "$1"
}

opt_value() {
  local want="$1" prev="" arg
  shift
  for arg in "$@"; do
    if [[ "$prev" == "$want" ]]; then
      printf '%s' "$arg"
      return 0
    fi
    prev="$arg"
  done
  return 0
}

payload_arg() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == @* ]]; then
      printf '%s' "${arg#@}"
      return 0
    fi
  done
  return 0
}

record_call() {
  local payload="$1"
  shift
  local payload_id="" payload_mode=""
  if [[ -n "$payload" ]]; then
    payload_id="$(next_seq)"
    payload_mode="$(file_mode "$payload")"
    mkdir -p "${state}/payloads"
    cp "$payload" "${state}/payloads/${payload_id}.json"
  fi
  jq -cn --arg tool "$TOOL" --arg payload "$payload_id" --arg mode "$payload_mode" --args \
    '{tool: $tool, payload: $payload, payload_mode: $mode, argv: $ARGS.positional}' \
    -- "$@" >>"$calls"
}

fail_if_requested() {
  if [[ -f "${state}/fail-$1" ]]; then
    printf 'fake %s: forced failure (%s)\n' "$TOOL" "$1" >&2
    exit 7
  fi
  if [[ -f "${state}/fail-once-$1" ]]; then
    rm -f "${state}/fail-once-$1"
    printf 'fake %s: transient failure (%s)\n' "$TOOL" "$1" >&2
    exit 8
  fi
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-'
}
"""

_FAKE_AZ = _FAKE_PRELUDE + r"""
sub1="${1:-}"
sub2="${2:-}"
sub3="${3:-}"
sub4="${4:-}"
payload="$(payload_arg "$@")"
record_call "$payload" "$@"

case "$sub1" in
  account)
    [[ "$sub2" == "show" ]] || die "unsupported account command: $sub2"
    fail_if_requested "account-show"
    jq -n --arg id "${FAKE_SUBSCRIPTION_ID}" --arg tenant "${FAKE_TENANT_ID}" \
      '{id: $id, tenantId: $tenant, name: "fake-subscription"}'
    ;;
  ad)
    case "$sub2 $sub3" in
      "app list")
        [[ "$(opt_value --filter "$@")" == *"korvid-prompt-lab-grounding"* ]] \
          || die "app list without the korvid-prompt-lab-grounding filter"
        if [[ -f "${state}/app_id" ]]; then cat "${state}/app_id"; fi
        printf '\n'
        ;;
      "app create")
        fail_if_requested "app-create"
        [[ "$(opt_value --display-name "$@")" == "korvid-prompt-lab-grounding" ]] \
          || die "unexpected application display name"
        printf '%s' "${FAKE_NEW_APP_ID}" >"${state}/app_id"
        printf '%s' "${FAKE_NEW_APP_OBJECT_ID}" >"${state}/app_object_id"
        printf '%s\n' "${FAKE_NEW_APP_ID}"
        ;;
      "app show")
        [[ -f "${state}/app_object_id" ]] || die "app show before the application exists"
        cat "${state}/app_object_id"
        printf '\n'
        ;;
      "app federated-credential")
        case "$sub4" in
          list)
            if [[ -f "${state}/fedcred.json" ]]; then
              cat "${state}/fedcred.json"
            else
              printf '[]\n'
            fi
            ;;
          create)
            fail_if_requested "fedcred-create"
            [[ -n "$payload" ]] || die "federated-credential create without @parameters file"
            if [[ ! -f "${state}/fedcred.json" ]]; then printf '[]' >"${state}/fedcred.json"; fi
            jq --slurpfile new "$payload" \
              '. + [$new[0] + {id: "00000000-fed0-0000-0000-000000000001"}]' \
              "${state}/fedcred.json" >"${state}/fedcred.next"
            mv "${state}/fedcred.next" "${state}/fedcred.json"
            ;;
          delete)
            cred_id="$(opt_value --federated-credential-id "$@")"
            [[ -n "$cred_id" ]] || die "federated-credential delete without --federated-credential-id"
            [[ -f "${state}/fedcred.json" ]] || die "federated-credential delete with no credential"
            jq --arg id "$cred_id" 'map(select(.id != $id))' \
              "${state}/fedcred.json" >"${state}/fedcred.next"
            mv "${state}/fedcred.next" "${state}/fedcred.json"
            ;;
          *) die "unsupported federated-credential command: $sub4" ;;
        esac
        ;;
      "sp list")
        [[ "$(opt_value --filter "$@")" == *"appId eq"* ]] || die "sp list without an appId filter"
        if [[ -f "${state}/sp_id" ]]; then cat "${state}/sp_id"; fi
        printf '\n'
        ;;
      "sp create")
        fail_if_requested "sp-create"
        printf '%s' "${FAKE_SP_OBJECT_ID}" >"${state}/sp_id"
        printf '%s\n' "${FAKE_SP_OBJECT_ID}"
        ;;
      *) die "unsupported ad command: $sub2 $sub3" ;;
    esac
    ;;
  aks)
    case "$sub2" in
      show)
        [[ "$(opt_value --name "$@")" == "aks-shared-runners" ]] || die "unexpected cluster name"
        printf '%s\n' "${FAKE_AKS_ID}"
        ;;
      nodepool)
        [[ "$sub3" == "show" ]] || die "unsupported nodepool command: $sub3"
        [[ "$(opt_value --name "$@")" == "modeleval" ]] || die "unexpected node pool name"
        if [[ -f "${state}/nodepool_id" ]]; then
          cat "${state}/nodepool_id"
          printf '\n'
        else
          printf '%s\n' "${FAKE_NODEPOOL_ID}"
        fi
        ;;
      *) die "unsupported aks command: $sub2" ;;
    esac
    ;;
  role)
    case "$sub2" in
      definition)
        case "$sub3" in
          list)
            name="$(opt_value --name "$@")"
            [[ -n "$name" ]] || die "role definition list without --name"
            if [[ -f "${state}/roles/$(slugify "$name")" ]]; then
              printf '%s\n' "$name"
            else
              printf '\n'
            fi
            ;;
          create|update)
            fail_if_requested "role-definition-${sub3}"
            [[ -n "$payload" ]] || die "role definition ${sub3} without a @role-definition file"
            name="$(jq -r '.Name' "$payload")"
            slug="$(slugify "$name")"
            if [[ "$sub3" == "update" && ! -f "${state}/roles/${slug}" ]]; then
              die "update of an undefined role: $name"
            fi
            if [[ "$sub3" == "create" && -f "${state}/roles/${slug}" ]]; then
              die "create of an already defined role: $name"
            fi
            mkdir -p "${state}/roles"
            printf '%s' "$name" >"${state}/roles/${slug}"
            ;;
          *) die "unsupported role definition command: $sub3" ;;
        esac
        ;;
      assignment)
        role="$(opt_value --role "$@")"
        scope="$(opt_value --scope "$@")"
        [[ -n "$role" && -n "$scope" ]] || die "role assignment without --role/--scope"
        case "$sub3" in
          list)
            if [[ -f "${state}/assignments.txt" ]] \
              && grep -Fxq "${role}|${scope}" "${state}/assignments.txt"; then
              printf '1\n'
            else
              printf '0\n'
            fi
            ;;
          create)
            fail_if_requested "role-assignment-create"
            if [[ "$role" != "Azure Kubernetes Service Cluster User Role" \
              && ! -f "${state}/roles/$(slugify "$role")" ]]; then
              die "role assignment before the custom role is defined: $role"
            fi
            printf '%s\n' "${role}|${scope}" >>"${state}/assignments.txt"
            ;;
          *) die "unsupported role assignment command: $sub3" ;;
        esac
        ;;
      *) die "unsupported role command: $sub2" ;;
    esac
    ;;
  *) die "unsupported command: $*" ;;
esac
"""

_FAKE_GH = _FAKE_PRELUDE + r"""
sub1="${1:-}"
sub2="${2:-}"
sub3="${3:-}"

case "$sub1" in
  api)
    if [[ "$sub2" == "user" ]]; then
      record_call "" "$@"
      fail_if_requested "gh-user"
      printf '%s\n' "${FAKE_GH_USER_ID}"
      exit 0
    fi
    payload="$(opt_value --input "$@")"
    record_call "$payload" "$@"
    fail_if_requested "gh-environment"
    [[ "$(opt_value --method "$@")" == "PUT" ]] || die "environment call without --method PUT"
    [[ -n "$payload" ]] || die "environment call without an --input payload"
    ;;
  variable|secret)
    [[ "$sub2" == "set" ]] || die "unsupported ${sub1} command: $sub2"
    [[ -n "$sub3" ]] || die "${sub1} set without a name"
    record_call "" "$@"
    mkdir -p "${state}/${sub1}s"
    cat >"${state}/${sub1}s/${sub3}"
    fail_if_requested "gh-${sub1}-set"
    ;;
  *) die "unsupported command: $sub1" ;;
esac
"""

#: kubectl and kubelogin exist only so the prerequisite check can find them.
#: Azure RBAC authorises the grounding identity, so invoking either of them
#: would mean the script tried to write Kubernetes RBAC objects.
_FAKE_KUBE = _FAKE_PRELUDE + r"""
record_call "" "$@"
die "must not be invoked: Azure RBAC authorises the grounding identity"
"""


@dataclass(frozen=True)
class AccessRun:
    """One execution of the access script against the fake CLIs."""

    process: subprocess.CompletedProcess[str]
    state: Path
    #: Byte offset of the shared call log when this run started, so a second
    #: run against the same fake cloud sees only its own calls.
    calls_offset: int = 0

    @property
    def returncode(self) -> int:
        return self.process.returncode

    @property
    def stdout(self) -> str:
        return self.process.stdout

    @property
    def stderr(self) -> str:
        return self.process.stderr

    @property
    def output(self) -> str:
        return self.process.stdout + self.process.stderr

    @property
    def calls_text(self) -> str:
        path = self.state / "calls.jsonl"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[self.calls_offset :]

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.calls_text.splitlines() if line.strip()]

    def calls_for(self, tool: str, *prefix: str) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if call["tool"] == tool and tuple(call["argv"][: len(prefix)]) == prefix
        ]

    def payload(self, call: dict[str, Any]) -> dict[str, Any]:
        assert call["payload"], f"call carried no payload file: {call['argv']}"
        body = (self.state / "payloads" / f"{call['payload']}.json").read_text(encoding="utf-8")
        parsed: dict[str, Any] = json.loads(body)
        return parsed

    def variable(self, name: str) -> str:
        return (self.state / "variables" / name).read_text(encoding="utf-8")

    def secret(self, name: str) -> str:
        return (self.state / "secrets" / name).read_text(encoding="utf-8")

    def variable_names(self) -> list[str]:
        return [call["argv"][2] for call in self.calls_for("gh", "variable", "set")]

    def secret_names(self) -> list[str]:
        return [call["argv"][2] for call in self.calls_for("gh", "secret", "set")]

    def assignments(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for call in self.calls_for("az", "role", "assignment", "create"):
            argv = call["argv"]
            pairs.append((argv[argv.index("--role") + 1], argv[argv.index("--scope") + 1]))
        return pairs


class AccessHarness:
    """A private workspace plus strict fake ``az``/``gh``/``kubectl`` executables."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.bin = root / "bin"
        self.render_root = root / "render"
        self.key_file = root / "korvid-app-private-key.pem"
        self.reflection_file = root / "reflection-credential.txt"
        for directory in (self.state, self.bin, self.render_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.key_file.write_text(PRIVATE_KEY, encoding="utf-8")
        self.reflection_file.write_text(f"{REFLECTION_CREDENTIAL_MARKER}\n", encoding="utf-8")
        for name, body in (
            ("az", _FAKE_AZ),
            ("gh", _FAKE_GH),
            ("kubectl", _FAKE_KUBE),
            ("kubelogin", _FAKE_KUBE),
        ):
            shim = self.bin / name
            shim.write_text(body, encoding="utf-8")
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # -- state seeding ----------------------------------------------------
    def seed_application(self, *, app_id: str = EXISTING_APP_ID) -> None:
        (self.state / "app_id").write_text(app_id, encoding="utf-8")
        (self.state / "app_object_id").write_text(EXISTING_APP_OBJECT_ID, encoding="utf-8")
        (self.state / "sp_id").write_text(SP_OBJECT_ID, encoding="utf-8")

    def seed_federated_credential(self, **overrides: Any) -> None:
        credential = {
            "id": "00000000-fed0-0000-0000-000000000009",
            "name": FEDERATED_NAME,
            "issuer": FEDERATED_ISSUER,
            "subject": FEDERATED_SUBJECT,
            "audiences": [FEDERATED_AUDIENCE],
        }
        credential.update(overrides)
        (self.state / "fedcred.json").write_text(json.dumps([credential]), encoding="utf-8")

    def seed_role_definitions(self, *names: str) -> None:
        roles = self.state / "roles"
        roles.mkdir(exist_ok=True)
        for name in names:
            slug = "".join(char if char.isalnum() else "-" for char in name.lower())
            while "--" in slug:
                slug = slug.replace("--", "-")
            (roles / slug).write_text(name, encoding="utf-8")

    def seed_node_pool_id(self, node_pool_id: str) -> None:
        (self.state / "nodepool_id").write_text(node_pool_id, encoding="utf-8")

    def fail(self, key: str) -> None:
        (self.state / f"fail-{key}").write_text("", encoding="utf-8")

    def fail_once(self, key: str) -> None:
        (self.state / f"fail-once-{key}").write_text("", encoding="utf-8")

    # -- execution --------------------------------------------------------
    def run(self, **overrides: str | None) -> AccessRun:
        assert _JQ is not None
        env: dict[str, str] = {
            "PATH": f"{self.bin}:{Path(_JQ).parent}:/usr/bin:/bin",
            "HOME": str(self.root),
            "TMPDIR": str(self.render_root),
            "FAKE_STATE_DIR": str(self.state),
            "FAKE_SUBSCRIPTION_ID": SUBSCRIPTION_ID,
            "FAKE_TENANT_ID": TENANT_ID,
            "FAKE_NEW_APP_ID": NEW_APP_ID,
            "FAKE_NEW_APP_OBJECT_ID": NEW_APP_OBJECT_ID,
            "FAKE_SP_OBJECT_ID": SP_OBJECT_ID,
            "FAKE_AKS_ID": AKS_ID,
            "FAKE_NODEPOOL_ID": NODE_POOL_ID,
            "FAKE_GH_USER_ID": GH_USER_ID,
            "KORVID_APP_ID": KORVID_APP_ID,
            "KORVID_APP_PRIVATE_KEY_FILE": str(self.key_file),
            "_GROUNDING_RETRY_ATTEMPTS": "2",
            "_GROUNDING_RETRY_DELAY_SECONDS": "0",
        }
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        call_log = self.state / "calls.jsonl"
        offset = len(call_log.read_text(encoding="utf-8")) if call_log.exists() else 0
        process = subprocess.run(
            ["bash", str(ACCESS_SCRIPT)],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return AccessRun(process=process, state=self.state, calls_offset=offset)


@pytest.fixture
def access(tmp_path: Path) -> AccessHarness:
    if _JQ is None:  # pragma: no cover - jq ships with the runner image
        pytest.skip("jq is required to execute the fake az and gh flows")
    return AccessHarness(tmp_path)


def assert_succeeded(run: AccessRun) -> None:
    assert run.returncode == 0, f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"


# ---------------------------------------------------------------------------
# Task 3a: the file contract the plan pins
# ---------------------------------------------------------------------------


def test_grounding_access_is_environment_bound_and_nodepool_scoped() -> None:
    body = ACCESS_SCRIPT.read_text(encoding="utf-8")
    assert "repo:hellices/korvid-prompt-lab:environment:aks-grounding" in body
    assert "agentPools/modeleval" in body
    assert "namespaces/ollama" in body
    assert "AZURE_CLIENT_SECRET" not in body
    assert "gh variable set AZURE_CLIENT_ID --env aks-grounding" in body
    assert "gh secret set KORVID_APP_PRIVATE_KEY --env aks-grounding" in body


def test_grounding_kubernetes_role_has_only_required_data_actions() -> None:
    role = json.loads(RBAC_TEMPLATE.read_text(encoding="utf-8"))
    assert role["Actions"] == []
    assert role["DataActions"] == [
        "Microsoft.ContainerService/managedClusters/apps/deployments/read",
        "Microsoft.ContainerService/managedClusters/endpoints/read",
        "Microsoft.ContainerService/managedClusters/pods/read",
        "Microsoft.ContainerService/managedClusters/pods/write",
        "Microsoft.ContainerService/managedClusters/services/read",
    ]
    assert role["NotDataActions"] == []
    assert "secrets" not in json.dumps(role)
    assert "exec/action" not in json.dumps(role)


def test_grounding_kubernetes_role_excludes_unsupported_portforward_actions() -> None:
    """``portforward/action`` is not an AKS DataAction; pods/write carries it."""
    body = RBAC_TEMPLATE.read_text(encoding="utf-8")
    assert "portforward" not in body
    role = json.loads(body)
    assert role["AssignableScopes"] == ["__SUBSCRIPTION_SCOPE__"]
    assert "RoleBinding" not in body


def test_access_script_never_places_secrets_on_cli_arguments() -> None:
    body = ACCESS_SCRIPT.read_text(encoding="utf-8")
    assert "--from-literal" not in body
    assert 'cat "$KORVID_APP_PRIVATE_KEY_FILE" | gh secret set' in body


def test_access_script_is_strict_bash_without_tracing_or_kubernetes_rbac() -> None:
    body = ACCESS_SCRIPT.read_text(encoding="utf-8")
    assert body.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in body
    assert "set -x" not in body
    assert "RoleBinding" not in body
    assert "kubectl apply" not in body
    assert "kubectl create" not in body
    assert ACCESS_SCRIPT.stat().st_mode & stat.S_IXUSR


def test_split_access_scripts_are_retired() -> None:
    """One orchestrator owns bootstrap; competing half-scripts must be gone."""
    still_present = [path.name for path in RETIRED_ACCESS_PATHS if path.exists()]
    assert still_present == [], f"obsolete grounding-access files remain: {still_present}"


# ---------------------------------------------------------------------------
# Task 3b: executed bootstrap behaviour
# ---------------------------------------------------------------------------


def test_cold_bootstrap_creates_identity_roles_and_environment(
    access: AccessHarness,
) -> None:
    run = access.run()
    assert_succeeded(run)

    assert len(run.calls_for("az", "ad", "app", "create")) == 1
    assert len(run.calls_for("az", "ad", "sp", "create")) == 1

    federated = run.calls_for("az", "ad", "app", "federated-credential", "create")
    assert len(federated) == 1
    assert run.payload(federated[0]) == {
        "name": FEDERATED_NAME,
        "issuer": FEDERATED_ISSUER,
        "subject": FEDERATED_SUBJECT,
        "audiences": [FEDERATED_AUDIENCE],
        "description": run.payload(federated[0])["description"],
    }
    assert not run.calls_for("az", "ad", "app", "federated-credential", "delete")

    assert len(run.calls_for("az", "role", "definition", "create")) == 2
    assert not run.calls_for("az", "role", "definition", "update")

    assert run.variable("AZURE_CLIENT_ID") == NEW_APP_ID
    assert run.variable("AZURE_TENANT_ID") == TENANT_ID
    assert run.variable("AZURE_SUBSCRIPTION_ID") == SUBSCRIPTION_ID
    assert run.variable("KORVID_APP_ID") == KORVID_APP_ID
    assert run.variable("KORVID_AKS_NAMESPACE") == "ollama"
    assert run.variable("KORVID_AKS_SERVICE") == "ollama"
    assert set(run.variable_names()) == REQUIRED_ENVIRONMENT_VARIABLES
    assert run.secret_names() == ["KORVID_APP_PRIVATE_KEY"]
    assert run.secret("KORVID_APP_PRIVATE_KEY") == PRIVATE_KEY


def test_bootstrap_assigns_exactly_three_scoped_roles(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    assert sorted(run.assignments()) == sorted(
        [
            (KUBERNETES_ROLE_NAME, NAMESPACE_SCOPE),
            ("Korvid Prompt Lab Grounding Node Pool Scaler", NODE_POOL_ID),
            (CLUSTER_USER_ROLE, AKS_ID),
        ]
    )
    for role, scope in run.assignments():
        if role == CLUSTER_USER_ROLE:
            assert scope == AKS_ID, "cluster user role must stay at cluster scope"
        else:
            assert scope != AKS_ID, f"{role} must not be assigned at cluster scope"


def test_rendered_role_definitions_carry_exact_actions_and_scope(
    access: AccessHarness,
) -> None:
    run = access.run()
    assert_succeeded(run)
    definitions = {
        run.payload(call)["Name"]: run.payload(call)
        for call in run.calls_for("az", "role", "definition", "create")
    }

    kubernetes = definitions[KUBERNETES_ROLE_NAME]
    assert kubernetes["Actions"] == []
    assert kubernetes["DataActions"] == [
        "Microsoft.ContainerService/managedClusters/apps/deployments/read",
        "Microsoft.ContainerService/managedClusters/endpoints/read",
        "Microsoft.ContainerService/managedClusters/pods/read",
        "Microsoft.ContainerService/managedClusters/pods/write",
        "Microsoft.ContainerService/managedClusters/services/read",
    ]
    assert kubernetes["NotDataActions"] == []
    assert kubernetes["AssignableScopes"] == [SUBSCRIPTION_SCOPE]

    scaler = definitions["Korvid Prompt Lab Grounding Node Pool Scaler"]
    assert scaler["Actions"] == [
        "Microsoft.ContainerService/managedClusters/agentPools/read",
        "Microsoft.ContainerService/managedClusters/agentPools/write",
    ]
    assert scaler["DataActions"] == []
    assert scaler["AssignableScopes"] == [SUBSCRIPTION_SCOPE]


def test_no_unexpanded_template_placeholder_reaches_azure(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    payloads = sorted((run.state / "payloads").glob("*.json"))
    assert payloads, "no rendered payload reached the fake az/gh"
    for payload in payloads:
        assert "__SUBSCRIPTION_SCOPE__" not in payload.read_text(encoding="utf-8")


def test_rendered_payloads_are_private_files(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    payload_calls = [call for call in run.calls if call["payload"]]
    assert len(payload_calls) >= 4
    for call in payload_calls:
        assert call["payload_mode"] == "600", f"{call['argv']} used mode {call['payload_mode']}"


def test_render_directory_is_removed_on_exit(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    assert list(access.render_root.iterdir()) == []


def test_second_run_reuses_every_created_object(access: AccessHarness) -> None:
    first = access.run()
    assert_succeeded(first)
    second = access.run()
    assert_succeeded(second)

    assert not second.calls_for("az", "ad", "app", "create")
    assert not second.calls_for("az", "ad", "sp", "create")
    assert not second.calls_for("az", "ad", "app", "federated-credential", "create")
    assert not second.calls_for("az", "ad", "app", "federated-credential", "delete")
    assert not second.calls_for("az", "role", "definition", "create")
    assert len(second.calls_for("az", "role", "definition", "update")) == 2
    assert second.assignments() == []
    assert set(second.variable_names()) == REQUIRED_ENVIRONMENT_VARIABLES


def test_existing_application_and_principal_are_reused(access: AccessHarness) -> None:
    access.seed_application()
    run = access.run()
    assert_succeeded(run)
    assert not run.calls_for("az", "ad", "app", "create")
    assert not run.calls_for("az", "ad", "sp", "create")
    assert run.variable("AZURE_CLIENT_ID") == EXISTING_APP_ID


def test_matching_federated_credential_is_left_untouched(access: AccessHarness) -> None:
    access.seed_application()
    access.seed_federated_credential()
    run = access.run()
    assert_succeeded(run)
    assert not run.calls_for("az", "ad", "app", "federated-credential", "create")
    assert not run.calls_for("az", "ad", "app", "federated-credential", "delete")


def test_drifted_federated_credential_is_replaced(access: AccessHarness) -> None:
    access.seed_application()
    access.seed_federated_credential(subject="repo:hellices/korvid-prompt-lab:ref:refs/heads/main")
    run = access.run()
    assert_succeeded(run)

    deletes = run.calls_for("az", "ad", "app", "federated-credential", "delete")
    creates = run.calls_for("az", "ad", "app", "federated-credential", "create")
    assert len(deletes) == 1
    assert len(creates) == 1
    assert run.calls.index(deletes[0]) < run.calls.index(creates[0])
    payload = run.payload(creates[0])
    assert payload["subject"] == FEDERATED_SUBJECT
    assert payload["audiences"] == [FEDERATED_AUDIENCE]
    assert payload["issuer"] == FEDERATED_ISSUER


def test_drifted_federated_audience_is_replaced(access: AccessHarness) -> None:
    access.seed_application()
    access.seed_federated_credential(audiences=["api://WrongAudience"])
    run = access.run()
    assert_succeeded(run)
    assert len(run.calls_for("az", "ad", "app", "federated-credential", "delete")) == 1
    assert len(run.calls_for("az", "ad", "app", "federated-credential", "create")) == 1


def test_existing_role_definitions_are_updated_not_recreated(
    access: AccessHarness,
) -> None:
    access.seed_role_definitions(
        KUBERNETES_ROLE_NAME, "Korvid Prompt Lab Grounding Node Pool Scaler"
    )
    run = access.run()
    assert_succeeded(run)
    assert not run.calls_for("az", "role", "definition", "create")
    assert len(run.calls_for("az", "role", "definition", "update")) == 2


def test_environment_requires_the_authenticated_user_as_reviewer(
    access: AccessHarness,
) -> None:
    run = access.run()
    assert_succeeded(run)
    environment_calls = [
        call
        for call in run.calls_for("gh", "api")
        if any("environments/aks-grounding" in arg for arg in call["argv"])
    ]
    assert len(environment_calls) == 1
    call = environment_calls[0]
    assert "repos/hellices/korvid-prompt-lab/environments/aks-grounding" in call["argv"]
    payload = run.payload(call)
    assert payload["reviewers"] == [{"type": "User", "id": int(GH_USER_ID)}]


def test_environment_exists_before_variables_and_secrets_are_written(
    access: AccessHarness,
) -> None:
    run = access.run()
    assert_succeeded(run)
    calls = run.calls
    environment_index = min(
        index
        for index, call in enumerate(calls)
        if call["tool"] == "gh" and any("environments/" in arg for arg in call["argv"])
    )
    writes = [
        index
        for index, call in enumerate(calls)
        if call["tool"] == "gh" and call["argv"][:1] in (["variable"], ["secret"])
    ]
    assert writes, "no Environment variable or secret was written"
    assert min(writes) > environment_index


def test_mutating_azure_commands_suppress_output(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    mutations = (
        run.calls_for("az", "role", "assignment", "create")
        + run.calls_for("az", "role", "definition", "create")
        + run.calls_for("az", "ad", "app", "federated-credential", "create")
    )
    assert mutations
    for call in mutations:
        argv = call["argv"]
        assert "--output" in argv and argv[argv.index("--output") + 1] == "none", argv


def test_kubernetes_clients_are_never_invoked(access: AccessHarness) -> None:
    run = access.run()
    assert_succeeded(run)
    assert not run.calls_for("kubectl")
    assert not run.calls_for("kubelogin")


# ---------------------------------------------------------------------------
# Task 3c: failure propagation and leakage
# ---------------------------------------------------------------------------


def test_azure_failure_stops_before_any_github_mutation(access: AccessHarness) -> None:
    access.fail("role-assignment-create")
    run = access.run()
    assert run.returncode != 0
    assert not run.calls_for("gh", "variable", "set")
    assert not run.calls_for("gh", "secret", "set")
    assert not run.calls_for("gh", "api")


def test_role_definition_failure_propagates(access: AccessHarness) -> None:
    access.fail("role-definition-create")
    run = access.run()
    assert run.returncode != 0
    assert not run.calls_for("az", "role", "assignment", "create")


def test_application_creation_failure_propagates(access: AccessHarness) -> None:
    access.fail("app-create")
    run = access.run()
    assert run.returncode != 0
    assert not run.calls_for("az", "role", "definition", "create")


def test_github_secret_failure_propagates(access: AccessHarness) -> None:
    access.fail("gh-secret-set")
    run = access.run()
    assert run.returncode != 0
    assert run.secret_names() == ["KORVID_APP_PRIVATE_KEY"]


def test_environment_failure_stops_before_variables(access: AccessHarness) -> None:
    access.fail("gh-environment")
    run = access.run()
    assert run.returncode != 0
    assert not run.calls_for("gh", "variable", "set")


def test_unreadable_private_key_fails_before_touching_the_cloud(
    access: AccessHarness,
) -> None:
    missing = access.root / "absent.pem"
    run = access.run(KORVID_APP_PRIVATE_KEY_FILE=str(missing))
    assert run.returncode != 0
    assert run.calls == []


def test_missing_korvid_app_id_fails_before_touching_the_cloud(
    access: AccessHarness,
) -> None:
    run = access.run(KORVID_APP_ID=None)
    assert run.returncode != 0
    assert run.calls == []


def test_unexpected_node_pool_id_fails_closed(access: AccessHarness) -> None:
    access.seed_node_pool_id(f"{AKS_ID}/agentPools/nodepool1")
    run = access.run()
    assert run.returncode != 0
    assert run.assignments() == []


def test_transient_principal_creation_is_retried(access: AccessHarness) -> None:
    """Entra replication lag must not turn into a bootstrap failure."""
    access.fail_once("sp-create")
    run = access.run(_GROUNDING_RETRY_ATTEMPTS="3", _GROUNDING_RETRY_DELAY_SECONDS="0")
    assert_succeeded(run)
    assert len(run.calls_for("az", "ad", "sp", "create")) == 2
    assert len(run.assignments()) == 3


def test_transient_role_assignment_failure_is_retried(access: AccessHarness) -> None:
    """A newly created principal is not visible to ARM immediately."""
    access.fail_once("role-assignment-create")
    run = access.run(_GROUNDING_RETRY_ATTEMPTS="3", _GROUNDING_RETRY_DELAY_SECONDS="0")
    assert_succeeded(run)
    assert len(run.calls_for("az", "role", "assignment", "create")) == 4


def test_retries_are_bounded_and_then_fail(access: AccessHarness) -> None:
    access.fail("sp-create")
    run = access.run(_GROUNDING_RETRY_ATTEMPTS="3", _GROUNDING_RETRY_DELAY_SECONDS="0")
    assert run.returncode != 0
    assert len(run.calls_for("az", "ad", "sp", "create")) == 3
    assert not run.calls_for("gh", "variable", "set")


def test_no_identifier_or_secret_reaches_the_console(access: AccessHarness) -> None:
    run = access.run(
        GROUNDING_REFLECTION_MODEL="openai/gpt-4.1-mini",
        GROUNDING_REFLECTION_CREDENTIAL_FILE=str(access.reflection_file),
    )
    assert_succeeded(run)
    for leak in (
        PRIVATE_KEY_MARKER,
        REFLECTION_CREDENTIAL_MARKER,
        SUBSCRIPTION_ID,
        TENANT_ID,
        NEW_APP_ID,
        SP_OBJECT_ID,
        NEW_APP_OBJECT_ID,
        AKS_ID,
    ):
        assert leak not in run.output, f"{leak} leaked to the console"
    for leak in (PRIVATE_KEY_MARKER, REFLECTION_CREDENTIAL_MARKER):
        assert leak not in run.calls_text, f"{leak} appeared on a command line"


def test_reflection_credential_is_streamed_from_a_file(access: AccessHarness) -> None:
    run = access.run(
        GROUNDING_REFLECTION_MODEL="openai/gpt-4.1-mini",
        GROUNDING_REFLECTION_CREDENTIAL_FILE=str(access.reflection_file),
    )
    assert_succeeded(run)
    assert sorted(run.secret_names()) == [
        "GROUNDING_REFLECTION_CREDENTIAL",
        "KORVID_APP_PRIVATE_KEY",
    ]
    assert run.secret("GROUNDING_REFLECTION_CREDENTIAL") == (
        access.reflection_file.read_text(encoding="utf-8")
    )
    assert run.variable("GROUNDING_REFLECTION_MODEL") == "openai/gpt-4.1-mini"


def test_reflection_model_without_a_credential_file_fails_closed(
    access: AccessHarness,
) -> None:
    run = access.run(GROUNDING_REFLECTION_MODEL="openai/gpt-4.1-mini")
    assert run.returncode != 0
    assert run.calls == []


def test_reflection_credential_file_must_be_readable(access: AccessHarness) -> None:
    run = access.run(
        GROUNDING_REFLECTION_MODEL="openai/gpt-4.1-mini",
        GROUNDING_REFLECTION_CREDENTIAL_FILE=str(access.root / "absent.txt"),
    )
    assert run.returncode != 0
    assert run.calls == []


# ---------------------------------------------------------------------------
# Task 4: install and verify the Prompt Lab runner scale set
#
# Both scripts are deployment tools, so grepping their source proves almost
# nothing.  These tests execute them against strict stateful fakes that hold
# the *live* cluster shape on disk: the AutoscalingRunnerSet the committed
# values file renders to, the Ollama deployment as it actually runs
# (``purpose=korvid-model-eval`` with the ``workload=ollama`` and spot taints),
# the ``modeleval`` pool, and the ``aks-grounding`` Environment whose endpoints
# answer with ``.variables[].name`` and ``.secrets[].name``.  Every fake
# refuses commands outside the contract, so a script that reads the wrong
# namespace, rewrites the operator kubeconfig, or mutates anything during a
# read-only audit fails the test instead of silently passing.
# ---------------------------------------------------------------------------
INSTALLER = ROOT / "scripts/install-prompt-lab-runner.sh"
VERIFIER = ROOT / "scripts/verify-grounding-deployment.sh"

RUNNER_NAMESPACE = "arc-runners-prompt-lab"
#: ARC's controller and every listener pod live here, not in the runner
#: namespace, so an installer that waits in the runner namespace hangs.
ARC_CONTROLLER_NAMESPACE = "arc-systems"
RELEASE_NAME = "prompt-lab-runners"
CHART_VERSION = "0.14.2"
CHART_REFERENCE = "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set"
RUNNER_SECRET_NAME = "prompt-lab-runners-github-app"
GITHUB_CONFIG_URL = "https://github.com/hellices/korvid-prompt-lab"
#: The reviewed runner image.  Both scripts must pin exactly this reference.
RUNNER_IMAGE = "acrpensionguard.azurecr.io/runner-base:prompt-lab-v1"

MODEL_NODE_POOL = "modeleval"
OLLAMA_NAMESPACE = "ollama"
#: Live scheduling of the model deployment — the verifier must pin exactly this.
OLLAMA_NODE_SELECTOR = {"purpose": "korvid-model-eval"}
OLLAMA_TOLERATIONS = [
    {"key": "workload", "operator": "Equal", "value": "ollama", "effect": "NoSchedule"},
    {
        "key": "kubernetes.azure.com/scalesetpriority",
        "operator": "Equal",
        "value": "spot",
        "effect": "NoSchedule",
    },
]

ARC_APP_ID = "8675309"
ARC_APP_INSTALLATION_ID = "90210901"
ARC_PRIVATE_KEY_MARKER = "arc-runner-app-private-key-marker"
ARC_PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    f"{ARC_PRIVATE_KEY_MARKER}\n"
    "-----END RSA PRIVATE KEY-----\n"
)

VARIABLE_VALUE_MARKER = "environment-variable-value-marker"
CLUSTER_ENDPOINT_MARKER = "cluster-endpoint-marker"
OPERATOR_KUBECONFIG_MARKER = "operator-kubeconfig-marker"

SAFE_EVIDENCE_DIR = "prompt-lab/artifacts/grounding-round/safe-evidence"

#: Tools the fakes must shadow.  If a real one is reachable through the system
#: path the "missing tool" tests cannot prove the preflight check runs.
_SYSTEM_PATH = "/usr/bin:/bin"
_SHADOWED_BY_SYSTEM = {
    tool
    for tool in ("az", "gh", "helm", "kubectl", "kubelogin")
    if shutil.which(tool, path=_SYSTEM_PATH) is not None
}


def _autoscaling_runner_set() -> dict[str, Any]:
    """The AutoscalingRunnerSet the committed values file renders to."""
    values = yaml.safe_load(VALUES.read_text(encoding="utf-8"))
    return {
        "apiVersion": "actions.github.com/v1alpha1",
        "kind": "AutoscalingRunnerSet",
        "metadata": {"name": RELEASE_NAME, "namespace": RUNNER_NAMESPACE},
        "spec": {
            "githubConfigUrl": values["githubConfigUrl"],
            "githubConfigSecret": values["githubConfigSecret"],
            "runnerScaleSetName": values["runnerScaleSetName"],
            "minRunners": values["minRunners"],
            "maxRunners": values["maxRunners"],
            "template": values["template"],
        },
    }


def _ollama_deployment() -> dict[str, Any]:
    """The Ollama deployment as it runs on the shared cluster today."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "ollama", "namespace": OLLAMA_NAMESPACE},
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "nodeSelector": dict(OLLAMA_NODE_SELECTOR),
                    "tolerations": [dict(toleration) for toleration in OLLAMA_TOLERATIONS],
                    "containers": [{"name": "ollama", "image": "ollama/ollama:0.12.3"}],
                }
            },
        },
    }


def _set_path(document: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``dotted`` (``a.b.0.c``) inside ``document``."""
    node: Any = document
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


#: Scale-set drift both scripts must reject, with the fragment each must report.
RUNNER_SET_DRIFT = [
    pytest.param("spec.githubConfigUrl", "https://github.com/hellices/korvid", "githubConfigUrl"),
    pytest.param("spec.minRunners", 1, "minRunners"),
    pytest.param("spec.maxRunners", 4, "maxRunners"),
    pytest.param("spec.template.spec.serviceAccountName", "default", "serviceAccountName"),
    pytest.param(
        "spec.template.spec.automountServiceAccountToken", True, "automountServiceAccountToken"
    ),
    pytest.param("spec.template.spec.nodeSelector.workload", "ollama", "nodeSelector"),
    pytest.param(
        "spec.template.spec.containers.0.securityContext.runAsNonRoot", False, "runAsNonRoot"
    ),
    pytest.param("spec.template.spec.containers.0.securityContext.runAsUser", 0, "runAsUser"),
    pytest.param("spec.template.spec.containers.0.securityContext.runAsGroup", 0, "runAsGroup"),
    pytest.param(
        "spec.template.spec.containers.0.securityContext.allowPrivilegeEscalation",
        True,
        "allowPrivilegeEscalation",
    ),
    pytest.param(
        "spec.template.spec.containers.0.image",
        "acrpensionguard.azurecr.io/runner-base:latest",
        RUNNER_IMAGE,
        id="runner-image",
    ),
]


# ---------------------------------------------------------------------------
# Task 4a: the file contract the plan pins
# ---------------------------------------------------------------------------


def test_runner_installer_pins_arc_and_handles_secrets_through_files() -> None:
    body = INSTALLER.read_text(encoding="utf-8")
    assert "gha-runner-scale-set" in body
    assert "--version 0.14.2" in body
    assert "--from-file=github_app_private_key=" in body
    assert "--from-literal" not in body
    assert "trap cleanup EXIT" in body


def test_deployment_verifier_is_read_only() -> None:
    body = VERIFIER.read_text(encoding="utf-8")
    for forbidden in ("kubectl apply", "kubectl delete", "helm upgrade", "az aks nodepool scale"):
        assert forbidden not in body
    assert "prompt-lab-runners" in body
    assert "modeleval" in body
    assert "safe-evidence" in body


def test_deployment_scripts_never_swallow_a_cli_failure() -> None:
    """A single ``|| true`` turns the audit into decoration."""
    for script in (INSTALLER, VERIFIER):
        body = script.read_text(encoding="utf-8")
        assert body.startswith("#!/usr/bin/env bash")
        assert "set -Eeuo pipefail" in body
        assert "|| true" not in body, f"{script.name} swallows a CLI failure"
        assert "set -x" not in body
        assert script.stat().st_mode & stat.S_IXUSR


def test_deployment_scripts_wait_for_the_listener_in_the_controller_namespace() -> None:
    """Listeners run in arc-systems; waiting in the runner namespace hangs."""
    body = INSTALLER.read_text(encoding="utf-8")
    assert "arc-systems" in body
    assert "arc-runners-prompt-lab" in body


def test_deployment_scripts_pin_the_reviewed_runner_image_literally() -> None:
    """A release running an unreviewed image is the drift that matters most."""
    for script in (INSTALLER, VERIFIER):
        body = script.read_text(encoding="utf-8")
        assert RUNNER_IMAGE in body, f"{script.name} does not pin the reviewed runner image"


# ---------------------------------------------------------------------------
# Task 4b: the fake cluster
# ---------------------------------------------------------------------------

#: Adds stdin capture and a read-only switch to the shared fake prelude.
_DEPLOY_PRELUDE = _FAKE_PRELUDE + r"""
record() {
  local stdin_file="$1"
  shift
  jq -cn --arg tool "$TOOL" --arg stdin "$stdin_file" --args \
    '{tool: $tool, stdin: $stdin, argv: $ARGS.positional}' -- "$@" >>"$calls"
}

capture_stdin() {
  local file
  mkdir -p "${state}/stdin"
  file="${state}/stdin/$(next_seq).txt"
  cat >"${file}"
  printf '%s' "${file}"
}

read_only() {
  [[ "${FAKE_READ_ONLY:-0}" == "1" ]]
}
"""

_FAKE_DEPLOY_AZ = _DEPLOY_PRELUDE + r"""
sub1="${1:-}"
sub2="${2:-}"
sub3="${3:-}"
out="$(opt_value --output "$@")"
if [[ -z "$out" ]]; then out="$(opt_value -o "$@")"; fi
record "" "$@"

emit() {
  case "${out:-json}" in
    none) : ;;
    json) printf '%s\n' "$1" ;;
    *) die "unsupported --output: ${out}" ;;
  esac
}

case "$sub1 $sub2" in
  "account show")
    fail_if_requested "az-account"
    emit "$(jq -cn --arg id "${FAKE_SUBSCRIPTION_ID}" '{id: $id, name: "fake-subscription"}')"
    ;;
  "aks show")
    [[ "$(opt_value --resource-group "$@")" == "rg-pension-guard" ]] || die "unexpected resource group"
    [[ "$(opt_value --name "$@")" == "aks-shared-runners" ]] || die "unexpected cluster name"
    fail_if_requested "az-aks-show"
    emit "$(jq -cn --arg id "${FAKE_AKS_ID}" '{id: $id, name: "aks-shared-runners"}')"
    ;;
  "aks get-credentials")
    file="$(opt_value --file "$@")"
    [[ -n "$file" ]] \
      || die "get-credentials without --file would rewrite the operator kubeconfig"
    [[ "$file" != "${HOME}/.kube/config" ]] \
      || die "get-credentials must never target the operator kubeconfig"
    [[ "$file" == "${TMPDIR%/}/"* ]] \
      || die "the kubeconfig must live in a private temporary directory: $file"
    [[ "$(opt_value --resource-group "$@")" == "rg-pension-guard" ]] || die "unexpected resource group"
    [[ "$(opt_value --name "$@")" == "aks-shared-runners" ]] || die "unexpected cluster name"
    fail_if_requested "az-credentials"
    printf 'apiVersion: v1\nkind: Config\ncurrent-context: aks-shared-runners\n' >"$file"
    ;;
  "aks nodepool")
    [[ "$sub3" == "show" ]] || die "unsupported nodepool command: $sub3"
    [[ "$(opt_value --resource-group "$@")" == "rg-pension-guard" ]] || die "unexpected resource group"
    [[ "$(opt_value --cluster-name "$@")" == "aks-shared-runners" ]] || die "unexpected cluster name"
    [[ "$(opt_value --name "$@")" == "modeleval" ]] || die "unexpected node pool"
    fail_if_requested "az-nodepool"
    emit "$(cat "${state}/nodepool.json")"
    ;;
  *) die "unsupported command: $*" ;;
esac
"""

_FAKE_DEPLOY_KUBELOGIN = _DEPLOY_PRELUDE + r"""
record "" "$@"
[[ "${1:-}" == "convert-kubeconfig" ]] || die "unsupported command: ${1:-}"
login_mode="$(opt_value -l "$@")"
if [[ -z "$login_mode" ]]; then login_mode="$(opt_value --login "$@")"; fi
[[ "$login_mode" == "azurecli" ]] \
  || die "convert-kubeconfig without '-l azurecli': '${login_mode}'"
[[ -n "${KUBECONFIG:-}" ]] || die "convert-kubeconfig without an exported KUBECONFIG"
[[ "${KUBECONFIG}" != "${HOME}/.kube/config" ]] \
  || die "convert-kubeconfig would rewrite the operator kubeconfig"
[[ -f "${KUBECONFIG}" ]] || die "KUBECONFIG points at a missing file: ${KUBECONFIG}"
fail_if_requested "kubelogin"
printf 'kubelogin-converted: azurecli\n' >>"${KUBECONFIG}"
"""

_FAKE_DEPLOY_KUBECTL = _DEPLOY_PRELUDE + r"""
[[ -n "${KUBECONFIG:-}" ]] \
  || die "kubectl ran without KUBECONFIG: it would use the operator's own context"
[[ "${KUBECONFIG}" != "${HOME}/.kube/config" ]] \
  || die "kubectl ran against the operator kubeconfig"
[[ -f "${KUBECONFIG}" ]] || die "KUBECONFIG points at a missing file: ${KUBECONFIG}"
kubeconfig_mode="$(file_mode "${KUBECONFIG}")"
[[ "${kubeconfig_mode}" == "600" ]] \
  || die "the kubeconfig is mode ${kubeconfig_mode}, expected 600"
grep -q 'kubelogin-converted' "${KUBECONFIG}" \
  || die "kubectl ran before kubelogin convert-kubeconfig"

LISTENER_POD="pod/prompt-lab-runners-6f8b7c9d-listener"

# The ARC controller creates the listener pod some time *after* the release is
# installed.  ``listener-appear-after`` is how many more queries answer "no
# resources"; each existence query consumes one.
listener_pending() {
  local pending=0
  if [[ -f "${state}/listener-appear-after" ]]; then
    pending="$(cat "${state}/listener-appear-after")"
  fi
  printf '%s' "$pending"
}

listener_exists() {
  [[ "$(listener_pending)" -le 0 ]]
}

listener_consume_query() {
  local pending
  pending="$(listener_pending)"
  if (( pending > 0 )); then
    printf '%s' "$(( pending - 1 ))" >"${state}/listener-appear-after"
  fi
}

namespace="$(opt_value --namespace "$@")"
if [[ -z "$namespace" ]]; then namespace="$(opt_value -n "$@")"; fi
output="$(opt_value --output "$@")"
if [[ -z "$output" ]]; then output="$(opt_value -o "$@")"; fi
selector="$(opt_value --selector "$@")"
if [[ -z "$selector" ]]; then selector="$(opt_value -l "$@")"; fi
filename="$(opt_value --filename "$@")"
if [[ -z "$filename" ]]; then filename="$(opt_value -f "$@")"; fi

p0=""
p1=""
p2=""
p3=""
index=0
skip=0
for arg in "$@"; do
  if (( skip )); then skip=0; continue; fi
  case "$arg" in
    -n|--namespace|-o|--output|-l|--selector|-f|--filename) skip=1 ;;
    -*) ;;
    *)
      case "$index" in
        0) p0="$arg" ;;
        1) p1="$arg" ;;
        2) p2="$arg" ;;
        3) p3="$arg" ;;
      esac
      index=$((index + 1))
      ;;
  esac
done

case "$p0" in
  cluster-info)
    record "" "$@"
    printf 'Kubernetes control plane is running at https://%s.hcp.koreacentral.azmk8s.io:443\n' \
      "${FAKE_CLUSTER_ENDPOINT_MARKER}"
    ;;
  get)
    record "" "$@"
    case "$p1" in
      pod|pods|po)
        [[ "$namespace" == "arc-systems" ]] \
          || die "listener pods listed in namespace '${namespace}': ARC listeners run in arc-systems"
        [[ "$selector" == *"actions.github.com/scale-set-name=prompt-lab-runners"* ]] \
          || die "listener query without the scale-set-name selector: '${selector}'"
        [[ "$selector" == *"actions.github.com/scale-set-namespace=arc-runners-prompt-lab"* ]] \
          || die "listener query without the scale-set-namespace selector: '${selector}'"
        [[ "$output" == "name" ]] \
          || die "unsupported output for a listener existence query: '${output}'"
        fail_if_requested "kubectl-listener-get"
        if listener_exists; then
          printf '%s\n' "${LISTENER_POD}"
        else
          listener_consume_query
          printf 'No resources found in arc-systems namespace.\n' >&2
        fi
        exit 0
        ;;
      autoscalingrunnersets.actions.github.com|autoscalingrunnerset)
        [[ "$namespace" == "arc-runners-prompt-lab" ]] \
          || die "AutoscalingRunnerSet read from namespace '${namespace}'"
        [[ "$p2" == "prompt-lab-runners" ]] || die "unexpected scale set: ${p2}"
        fail_if_requested "kubectl-runner-set"
        document="${state}/ars.json"
        ;;
      deployment|deployments|deploy)
        [[ "$namespace" == "ollama" ]] || die "Ollama read from namespace '${namespace}'"
        [[ "$p2" == "ollama" ]] || die "unexpected deployment: ${p2}"
        fail_if_requested "kubectl-ollama"
        document="${state}/ollama.json"
        ;;
      *) die "unsupported resource: ${p1}" ;;
    esac
    [[ -f "$document" ]] || die "resource not found: ${p1}/${p2}"
    case "$output" in
      json) cat "$document" ;;
      jsonpath=*)
        expression="${output#jsonpath=}"
        expression="${expression#\{}"
        expression="${expression%\}}"
        value="$(jq -rc "$expression" "$document")"
        if [[ "$value" == "null" ]]; then value=""; fi
        printf '%s' "$value"
        ;;
      *) die "unsupported output for get: '${output}'" ;;
    esac
    ;;
  apply)
    if read_only; then die "apply mutates the cluster"; fi
    if [[ "$filename" == "-" ]]; then
      stdin_file="$(capture_stdin)"
      record "$stdin_file" "$@"
    else
      [[ -f "$filename" ]] || die "apply of a missing manifest: '${filename}'"
      record "" "$@"
    fi
    fail_if_requested "kubectl-apply"
    printf 'applied\n'
    ;;
  create)
    if read_only; then die "create mutates the cluster"; fi
    record "" "$@"
    [[ "$p1" == "secret" ]] || die "unsupported create: ${p1}"
    [[ "$p2" == "generic" ]] || die "unsupported secret type: ${p2}"
    [[ "$p3" == "prompt-lab-runners-github-app" ]] || die "unexpected secret name: ${p3}"
    [[ "$namespace" == "arc-runners-prompt-lab" ]] \
      || die "secret created in namespace '${namespace}'"
    [[ "$output" == "yaml" ]] || die "create secret must render yaml for apply"
    dry_run=false
    for arg in "$@"; do
      case "$arg" in
        --dry-run=client) dry_run=true ;;
        --dry-run|--dry-run=*) die "wrong dry-run mode: ${arg}" ;;
        --from-literal|--from-literal=*) die "--from-literal puts a secret in argv" ;;
      esac
    done
    [[ "$dry_run" == true ]] \
      || die "create secret without --dry-run=client writes the secret directly"
    data='{}'
    for arg in "$@"; do
      case "$arg" in
        --from-file=*)
          specification="${arg#--from-file=}"
          key="${specification%%=*}"
          path="${specification#*=}"
          [[ "$key" != "$specification" ]] || die "--from-file without an explicit key"
          [[ -f "$path" ]] || die "--from-file source is missing: ${path}"
          source_mode="$(file_mode "$path")"
          [[ "$source_mode" == "600" ]] \
            || die "--from-file source is mode ${source_mode}, expected 600"
          encoded="$(jq -Rrs '@base64' <"$path")"
          data="$(printf '%s' "$data" \
            | jq -c --arg k "$key" --arg v "$encoded" '. + {($k): $v}')"
          ;;
      esac
    done
    jq -cn --arg name "$p3" --arg ns "$namespace" --argjson data "$data" \
      '{apiVersion: "v1", kind: "Secret", type: "Opaque",
        metadata: {name: $name, namespace: $ns}, data: $data}'
    ;;
  wait)
    record "" "$@"
    [[ "$namespace" == "arc-systems" ]] \
      || die "listener wait in namespace '${namespace}': ARC listeners run in arc-systems"
    [[ "$p1" == "pod" || "$p1" == "pods" ]] || die "unexpected wait resource: ${p1}"
    [[ "$selector" == *"actions.github.com/scale-set-name=prompt-lab-runners"* ]] \
      || die "listener wait without the scale-set-name selector: '${selector}'"
    [[ "$selector" == *"actions.github.com/scale-set-namespace=arc-runners-prompt-lab"* ]] \
      || die "listener wait without the scale-set-namespace selector: '${selector}'"
    [[ "$*" == *"--for=condition=Ready"* ]] || die "listener wait without a Ready condition"
    [[ "$*" == *"--timeout="* ]] || die "listener wait without a timeout"
    # Real kubectl does not wait for a resource that does not exist yet: a
    # condition wait whose selector matches nothing exits 1 immediately.
    if ! listener_exists; then
      listener_consume_query
      printf 'error: no matching resources found\n' >&2
      exit 1
    fi
    fail_if_requested "listener-wait"
    if [[ -f "${state}/listener-never-ready" ]]; then
      printf 'error: timed out waiting for the condition on pods/prompt-lab-runners-6f8b7c9d-listener\n' >&2
      exit 1
    fi
    printf '%s condition met\n' "${LISTENER_POD}"
    ;;
  *)
    record "" "$@"
    die "refused: '${p0}' is not part of the install or verify contract"
    ;;
esac
"""

_FAKE_DEPLOY_HELM = _DEPLOY_PRELUDE + r"""
namespace="$(opt_value --namespace "$@")"
if [[ -z "$namespace" ]]; then namespace="$(opt_value -n "$@")"; fi
output="$(opt_value --output "$@")"
if [[ -z "$output" ]]; then output="$(opt_value -o "$@")"; fi
record "" "$@"

p0=""
p1=""
index=0
skip=0
for arg in "$@"; do
  if (( skip )); then skip=0; continue; fi
  case "$arg" in
    -n|--namespace|-o|--output|-f|--values|--version|--timeout|--kube-context) skip=1 ;;
    -*) ;;
    *)
      case "$index" in
        0) p0="$arg" ;;
        1) p1="$arg" ;;
      esac
      index=$((index + 1))
      ;;
  esac
done

case "$p0" in
  status)
    [[ "$p1" == "prompt-lab-runners" ]] || die "unexpected release: ${p1}"
    [[ "$namespace" == "arc-runners-prompt-lab" ]] \
      || die "release read from namespace '${namespace}'"
    [[ "$output" == "json" ]] || die "helm status must request json"
    fail_if_requested "helm-status"
    jq -cn --arg status "$(cat "${state}/release-status")" \
      '{name: "prompt-lab-runners", info: {status: $status}}'
    ;;
  upgrade)
    if read_only; then die "a release upgrade mutates the cluster"; fi
    [[ "$p1" == "prompt-lab-runners" ]] || die "unexpected release: ${p1}"
    [[ "$namespace" == "arc-runners-prompt-lab" ]] \
      || die "release installed into namespace '${namespace}'"
    [[ "$(opt_value --version "$@")" == "0.14.2" ]] || die "the chart version is not pinned"
    [[ "$(opt_value --timeout "$@")" == "10m" ]] || die "install without a 10m timeout"
    values="$(opt_value --values "$@")"
    [[ -f "$values" ]] || die "values file missing: '${values}'"
    install=false
    wait_for_release=false
    chart=""
    for arg in "$@"; do
      case "$arg" in
        --install) install=true ;;
        --wait) wait_for_release=true ;;
        oci://*) chart="$arg" ;;
      esac
    done
    [[ "$install" == true ]] || die "upgrade without --install"
    [[ "$wait_for_release" == true ]] || die "upgrade without --wait"
    [[ "$chart" == "oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set" ]] \
      || die "unexpected chart reference: '${chart}'"
    fail_if_requested "helm-upgrade"
    printf 'deployed' >"${state}/release-status"
    printf 'Release "prompt-lab-runners" has been upgraded.\n'
    ;;
  *) die "unsupported helm command: ${p0}" ;;
esac
"""

_FAKE_DEPLOY_GH = _DEPLOY_PRELUDE + r"""
[[ "${1:-}" == "api" ]] || die "unsupported command: ${1:-}"
path="${2:-}"
record "" "$@"

method="$(opt_value --method "$@")"
if [[ -z "$method" ]]; then method="$(opt_value -X "$@")"; fi
if [[ -n "$method" && "$method" != "GET" ]]; then
  die "the verifier must not call the API with --method ${method}"
fi
for arg in "$@"; do
  case "$arg" in
    -f|--field|-F|--raw-field|--input) die "the verifier must not send a request body" ;;
  esac
done

jq_expression="$(opt_value --jq "$@")"
if [[ -z "$jq_expression" ]]; then jq_expression="$(opt_value -q "$@")"; fi

emit() {
  if [[ -n "$jq_expression" ]]; then
    printf '%s' "$1" | jq -r "$jq_expression"
  else
    printf '%s\n' "$1"
  fi
}

fail_if_requested "gh-api"

case "$path" in
  "repos/hellices/korvid-prompt-lab/environments/aks-grounding")
    if [[ -f "${state}/environment-missing" ]]; then
      printf 'gh: Not Found (HTTP 404)\n' >&2
      exit 1
    fi
    emit "$(jq -cn '{name: "aks-grounding",
                     protection_rules: [{type: "required_reviewers"}]}')"
    ;;
  "repos/hellices/korvid-prompt-lab/environments/aks-grounding/variables")
    emit "$(jq -Rsc --arg marker "${FAKE_VARIABLE_VALUE_MARKER}" \
      'split("\n") | map(select(length > 0))
       | {total_count: length,
          variables: map({name: ., value: ($marker + "-" + .)})}' \
      <"${state}/environment-variables")"
    ;;
  "repos/hellices/korvid-prompt-lab/environments/aks-grounding/secrets")
    emit "$(jq -Rsc \
      'split("\n") | map(select(length > 0))
       | {total_count: length,
          secrets: map({name: ., created_at: "2026-08-01T00:00:00Z"})}' \
      <"${state}/environment-secrets")"
    ;;
  *) die "unsupported api path: ${path}" ;;
esac
"""

_PYTHON_SHIM = f"""#!/usr/bin/env bash
exec {shlex.quote(sys.executable)} "$@"
"""


class DeployRun(AccessRun):
    """One execution of the installer or the verifier against the fake cluster."""

    def calls_with(self, tool: str, *tokens: str) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if call["tool"] == tool and all(token in call["argv"] for token in tokens)
        ]

    def tools(self) -> list[str]:
        return [call["tool"] for call in self.calls]

    def call_positions(self, tool: str, *tokens: str) -> list[int]:
        """Indices of the matching calls inside this run's ordered call log."""
        return [
            index
            for index, call in enumerate(self.calls)
            if call["tool"] == tool and all(token in call["argv"] for token in tokens)
        ]

    def stdin_text(self, call: dict[str, Any]) -> str:
        path = call.get("stdin") or ""
        assert path, f"call carried no stdin: {call['argv']}"
        return Path(path).read_text(encoding="utf-8")


class DeployHarness:
    """A sandbox checkout plus strict fake az/kubectl/kubelogin/helm/gh binaries."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.bin = root / "bin"
        self.tmpdir = root / "tmp"
        self.home = root / "home"
        self.repo = root / "repo"
        self.key_file = root / "arc-github-app.pem"
        for directory in (self.state, self.bin, self.tmpdir, self.home, self.repo):
            directory.mkdir(parents=True, exist_ok=True)

        (self.home / ".kube").mkdir(parents=True, exist_ok=True)
        self.operator_kubeconfig = self.home / ".kube" / "config"
        self.operator_kubeconfig.write_text(OPERATOR_KUBECONFIG_MARKER, encoding="utf-8")
        self.key_file.write_text(ARC_PRIVATE_KEY, encoding="utf-8")

        for relative in (
            "scripts/install-prompt-lab-runner.sh",
            "scripts/verify-grounding-deployment.sh",
            "infra/arc/prompt-lab-runners-values.yaml",
            "infra/arc/prompt-lab-runner-service-account.yaml",
            ".github/workflows/grounding-round.yml",
        ):
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        self.workflow = self.repo / ".github/workflows/grounding-round.yml"

        for name, body in (
            ("az", _FAKE_DEPLOY_AZ),
            ("gh", _FAKE_DEPLOY_GH),
            ("helm", _FAKE_DEPLOY_HELM),
            ("kubectl", _FAKE_DEPLOY_KUBECTL),
            ("kubelogin", _FAKE_DEPLOY_KUBELOGIN),
            ("python3", _PYTHON_SHIM),
        ):
            shim = self.bin / name
            shim.write_text(body, encoding="utf-8")
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        self.set_runner_set(_autoscaling_runner_set())
        self.set_ollama(_ollama_deployment())
        self.set_environment_variables(sorted(REQUIRED_ENVIRONMENT_VARIABLES))
        self.set_environment_secrets(["KORVID_APP_PRIVATE_KEY"])
        self.set_node_pool(count=0, provisioning_state="Succeeded")
        self.set_release_status("deployed")
        self.set_listener()

    # -- state seeding ----------------------------------------------------
    @property
    def runner_set(self) -> dict[str, Any]:
        document: dict[str, Any] = json.loads(
            (self.state / "ars.json").read_text(encoding="utf-8")
        )
        return document

    def set_runner_set(self, document: dict[str, Any]) -> None:
        (self.state / "ars.json").write_text(json.dumps(document), encoding="utf-8")

    @property
    def ollama(self) -> dict[str, Any]:
        document: dict[str, Any] = json.loads(
            (self.state / "ollama.json").read_text(encoding="utf-8")
        )
        return document

    def set_ollama(self, document: dict[str, Any]) -> None:
        (self.state / "ollama.json").write_text(json.dumps(document), encoding="utf-8")

    def set_environment_variables(self, names: list[str]) -> None:
        body = "".join(f"{name}\n" for name in names)
        (self.state / "environment-variables").write_text(body, encoding="utf-8")

    def set_environment_secrets(self, names: list[str]) -> None:
        body = "".join(f"{name}\n" for name in names)
        (self.state / "environment-secrets").write_text(body, encoding="utf-8")

    def remove_environment(self) -> None:
        (self.state / "environment-missing").write_text("", encoding="utf-8")

    def set_node_pool(self, *, count: int, provisioning_state: str) -> None:
        document = {
            "name": MODEL_NODE_POOL,
            "count": count,
            "provisioningState": provisioning_state,
            "nodeTaints": ["workload=ollama:NoSchedule"],
        }
        (self.state / "nodepool.json").write_text(json.dumps(document), encoding="utf-8")

    def set_release_status(self, status: str) -> None:
        (self.state / "release-status").write_text(status, encoding="utf-8")

    def set_listener(self, *, appear_after: int = 0, never_ready: bool = False) -> None:
        """Model the listener lifecycle the ARC controller drives.

        ``appear_after`` is how many listener queries answer "no resources"
        before the pod exists, so ``appear_after=2`` is a fresh install where
        the controller has not created the listener yet.
        """
        (self.state / "listener-appear-after").write_text(str(appear_after), encoding="utf-8")
        marker = self.state / "listener-never-ready"
        if never_ready:
            marker.write_text("", encoding="utf-8")
        elif marker.exists():
            marker.unlink()

    def fail(self, key: str) -> None:
        (self.state / f"fail-{key}").write_text("", encoding="utf-8")

    def remove_tool(self, name: str) -> None:
        (self.bin / name).unlink()

    # -- execution --------------------------------------------------------
    def _run(self, script: str, overrides: dict[str, str | None]) -> DeployRun:
        assert _JQ is not None
        env: dict[str, str] = {
            "PATH": f"{self.bin}:{Path(_JQ).parent}:{_SYSTEM_PATH}",
            "HOME": str(self.home),
            "TMPDIR": str(self.tmpdir),
            "FAKE_STATE_DIR": str(self.state),
            "FAKE_READ_ONLY": "1" if script.startswith("verify") else "0",
            "FAKE_SUBSCRIPTION_ID": SUBSCRIPTION_ID,
            "FAKE_AKS_ID": AKS_ID,
            "FAKE_VARIABLE_VALUE_MARKER": VARIABLE_VALUE_MARKER,
            "FAKE_CLUSTER_ENDPOINT_MARKER": CLUSTER_ENDPOINT_MARKER,
            "ARC_GITHUB_APP_ID": ARC_APP_ID,
            "ARC_GITHUB_APP_INSTALLATION_ID": ARC_APP_INSTALLATION_ID,
            "ARC_GITHUB_APP_PRIVATE_KEY_FILE": str(self.key_file),
        }
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        call_log = self.state / "calls.jsonl"
        offset = len(call_log.read_text(encoding="utf-8")) if call_log.exists() else 0
        process = subprocess.run(
            ["bash", str(self.repo / "scripts" / script)],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return DeployRun(process=process, state=self.state, calls_offset=offset)

    def install(self, **overrides: str | None) -> DeployRun:
        return self._run("install-prompt-lab-runner.sh", overrides)

    def verify(self, **overrides: str | None) -> DeployRun:
        return self._run("verify-grounding-deployment.sh", overrides)


@pytest.fixture
def deploy(tmp_path: Path) -> DeployHarness:
    if _JQ is None:  # pragma: no cover - jq ships with the runner image
        pytest.skip("jq is required to execute the fake cluster flows")
    return DeployHarness(tmp_path)


# ---------------------------------------------------------------------------
# Task 4c: executed installer behaviour
# ---------------------------------------------------------------------------


def test_installer_uses_a_private_kubeconfig_and_leaves_the_operator_default_alone(
    deploy: DeployHarness,
) -> None:
    run = deploy.install()
    assert_succeeded(run)

    credentials = run.calls_with("az", "get-credentials")
    assert len(credentials) == 1
    argv = credentials[0]["argv"]
    kubeconfig = argv[argv.index("--file") + 1]
    assert kubeconfig.startswith(f"{deploy.tmpdir}/")
    assert kubeconfig != str(deploy.operator_kubeconfig)
    assert "--overwrite-existing" in argv

    assert deploy.operator_kubeconfig.read_text(encoding="utf-8") == OPERATOR_KUBECONFIG_MARKER
    assert not Path(kubeconfig).exists(), "the private kubeconfig survived the run"
    assert list(deploy.tmpdir.iterdir()) == []


def test_installer_converts_the_kubeconfig_before_the_first_kubectl_call(
    deploy: DeployHarness,
) -> None:
    run = deploy.install()
    assert_succeeded(run)

    tools = run.tools()
    assert "kubelogin" in tools and "kubectl" in tools
    assert tools.index("kubelogin") < tools.index("kubectl")
    conversions = run.calls_for("kubelogin", "convert-kubeconfig")
    assert len(conversions) == 1
    assert conversions[0]["argv"] == ["convert-kubeconfig", "-l", "azurecli"]


@pytest.mark.parametrize("missing", ["az", "helm", "kubectl", "kubelogin"])
def test_installer_requires_every_tool_before_any_mutation(
    deploy: DeployHarness, missing: str
) -> None:
    if missing in _SHADOWED_BY_SYSTEM:  # pragma: no cover - depends on the host
        pytest.skip(f"a real {missing} is reachable through {_SYSTEM_PATH}")
    deploy.remove_tool(missing)
    run = deploy.install()
    assert run.returncode != 0
    assert f"required command not found: {missing}" in run.output
    assert run.calls == []


@pytest.mark.parametrize(
    "variable",
    ["ARC_GITHUB_APP_ID", "ARC_GITHUB_APP_INSTALLATION_ID", "ARC_GITHUB_APP_PRIVATE_KEY_FILE"],
)
def test_installer_requires_the_app_environment_before_any_call(
    deploy: DeployHarness, variable: str
) -> None:
    run = deploy.install(**{variable: None})
    assert run.returncode != 0
    assert run.calls == []


def test_installer_requires_a_readable_private_key(deploy: DeployHarness) -> None:
    run = deploy.install(ARC_GITHUB_APP_PRIVATE_KEY_FILE=str(deploy.root / "absent.pem"))
    assert run.returncode != 0
    assert run.calls == []


def test_installer_streams_every_secret_through_a_mode_600_file(deploy: DeployHarness) -> None:
    run = deploy.install()
    assert_succeeded(run)

    creations = run.calls_with("kubectl", "create", "secret")
    assert len(creations) == 1
    keys = sorted(
        argument.split("=", 2)[1]
        for argument in creations[0]["argv"]
        if argument.startswith("--from-file=")
    )
    assert keys == ["github_app_id", "github_app_installation_id", "github_app_private_key"]

    applied = run.calls_with("kubectl", "apply", "-")
    assert len(applied) == 1
    secret = json.loads(run.stdin_text(applied[0]))
    assert secret["kind"] == "Secret"
    assert secret["metadata"]["name"] == RUNNER_SECRET_NAME
    assert secret["metadata"]["namespace"] == RUNNER_NAMESPACE
    assert base64.b64decode(secret["data"]["github_app_private_key"]).decode() == ARC_PRIVATE_KEY
    assert base64.b64decode(secret["data"]["github_app_id"]).decode() == ARC_APP_ID
    assert (
        base64.b64decode(secret["data"]["github_app_installation_id"]).decode()
        == ARC_APP_INSTALLATION_ID
    )


def test_installer_never_puts_a_secret_in_argv_or_on_the_console(deploy: DeployHarness) -> None:
    run = deploy.install()
    assert_succeeded(run)
    for leak in (
        ARC_PRIVATE_KEY_MARKER,
        ARC_APP_ID,
        ARC_APP_INSTALLATION_ID,
        CLUSTER_ENDPOINT_MARKER,
        SUBSCRIPTION_ID,
        AKS_ID,
    ):
        assert leak not in run.output, f"{leak} reached the console"
        assert leak not in run.calls_text, f"{leak} reached a command line"


def test_installer_pins_the_chart_namespace_and_values(deploy: DeployHarness) -> None:
    run = deploy.install()
    assert_succeeded(run)

    upgrades = run.calls_with("helm", "upgrade", "--install")
    assert len(upgrades) == 1
    argv = upgrades[0]["argv"]
    assert argv[argv.index("--install") + 1] == RELEASE_NAME
    assert argv[argv.index("--version") + 1] == CHART_VERSION
    assert argv[argv.index("--namespace") + 1] == RUNNER_NAMESPACE
    assert argv[argv.index("--timeout") + 1] == "10m"
    assert CHART_REFERENCE in argv
    assert "--wait" in argv
    assert Path(argv[argv.index("--values") + 1]) == (
        deploy.repo / "infra/arc/prompt-lab-runners-values.yaml"
    )


def test_installer_mutates_only_the_service_account_secret_and_release(
    deploy: DeployHarness,
) -> None:
    run = deploy.install()
    assert_succeeded(run)

    mutations = []
    for call in run.calls:
        argv = call["argv"]
        if call["tool"] == "kubectl" and "apply" in argv:
            mutations.append("kubectl apply")
        if call["tool"] == "helm" and "upgrade" in argv:
            mutations.append("helm upgrade")
        assert not any(
            forbidden in argv for forbidden in ("delete", "scale", "patch", "replace", "edit")
        ), f"the installer issued a forbidden verb: {argv}"
    assert mutations == ["kubectl apply", "kubectl apply", "helm upgrade"]


def test_installer_waits_for_the_listener_in_the_controller_namespace(
    deploy: DeployHarness,
) -> None:
    run = deploy.install()
    assert_succeeded(run)

    waits = run.calls_with("kubectl", "wait")
    assert len(waits) == 1
    argv = waits[0]["argv"]
    assert argv[argv.index("--namespace") + 1] == ARC_CONTROLLER_NAMESPACE
    selector = argv[argv.index("--selector") + 1]
    assert f"actions.github.com/scale-set-name={RELEASE_NAME}" in selector
    assert f"actions.github.com/scale-set-namespace={RUNNER_NAMESPACE}" in selector


def test_installer_fails_when_the_listener_never_becomes_ready(deploy: DeployHarness) -> None:
    deploy.fail("listener-wait")
    run = deploy.install()
    assert run.returncode != 0
    assert "listener" in run.output


def test_installer_waits_for_the_listener_to_be_created_before_waiting_for_ready(
    deploy: DeployHarness,
) -> None:
    """A fresh install has no listener pod when helm returns.

    ``kubectl wait --for=condition=Ready`` does not wait for a resource that
    does not exist yet: a selector matching nothing exits 1 immediately with
    "no matching resources found".  The install must therefore wait for the
    listener to appear first, then wait for it to become Ready.
    """
    deploy.set_listener(appear_after=2)

    run = deploy.install(
        LISTENER_CREATE_TIMEOUT_SECONDS="30",
        LISTENER_POLL_INTERVAL_SECONDS="1",
    )
    assert_succeeded(run)

    existence = run.call_positions("kubectl", "get", "pods")
    assert len(existence) >= 3, "the installer gave up before the listener could appear"
    ready = run.call_positions("kubectl", "wait")
    assert len(ready) == 1
    assert max(existence) < ready[0], "the Ready wait ran before the listener existed"

    for call in run.calls_with("kubectl", "get", "pods"):
        argv = call["argv"]
        assert argv[argv.index("--namespace") + 1] == ARC_CONTROLLER_NAMESPACE
        selector = argv[argv.index("--selector") + 1]
        assert f"actions.github.com/scale-set-name={RELEASE_NAME}" in selector
        assert f"actions.github.com/scale-set-namespace={RUNNER_NAMESPACE}" in selector

    assert "listener" in run.output


def test_installer_reports_a_listener_that_is_never_created(deploy: DeployHarness) -> None:
    """Never-created is a different failure from created-but-not-Ready."""
    deploy.set_listener(appear_after=10_000)

    run = deploy.install(
        LISTENER_CREATE_TIMEOUT_SECONDS="2",
        LISTENER_POLL_INTERVAL_SECONDS="1",
    )
    assert run.returncode != 0
    assert f"no {RELEASE_NAME} listener pod was created" in run.output
    assert "did not become Ready" not in run.output
    assert run.calls_with("kubectl", "wait") == [], (
        "the installer waited on a Ready condition for a pod that never existed"
    )
    assert len(run.calls_with("kubectl", "get", "pods")) >= 2, "the existence wait was not bounded"
    assert list(deploy.tmpdir.iterdir()) == []
    assert deploy.operator_kubeconfig.read_text(encoding="utf-8") == OPERATOR_KUBECONFIG_MARKER


def test_installer_distinguishes_a_listener_that_is_created_but_never_ready(
    deploy: DeployHarness,
) -> None:
    deploy.set_listener(appear_after=1, never_ready=True)

    run = deploy.install(
        LISTENER_CREATE_TIMEOUT_SECONDS="30",
        LISTENER_POLL_INTERVAL_SECONDS="1",
    )
    assert run.returncode != 0
    assert "did not become Ready" in run.output
    assert "was created" not in run.output, "a not-Ready listener was reported as never created"
    assert len(run.calls_with("kubectl", "wait")) == 1
    assert list(deploy.tmpdir.iterdir()) == []
    assert deploy.operator_kubeconfig.read_text(encoding="utf-8") == OPERATOR_KUBECONFIG_MARKER


def test_installer_rejects_a_listener_timeout_that_is_not_a_number(
    deploy: DeployHarness,
) -> None:
    """A typo'd bound must be named, not silently arithmetic-evaluated to 0."""
    run = deploy.install(LISTENER_CREATE_TIMEOUT_SECONDS="two minutes")
    assert run.returncode != 0
    assert "LISTENER_CREATE_TIMEOUT_SECONDS" in run.output
    assert run.calls == [], "the installer called out before validating its bounds"


def test_installer_rejects_a_zero_listener_poll_interval(deploy: DeployHarness) -> None:
    """A zero interval would hammer the API server for the whole create timeout."""
    run = deploy.install(LISTENER_POLL_INTERVAL_SECONDS="0")
    assert run.returncode != 0
    assert "LISTENER_POLL_INTERVAL_SECONDS" in run.output
    assert run.calls == []


def test_a_ready_wait_without_a_listener_pod_fails_like_real_kubectl(
    deploy: DeployHarness,
) -> None:
    """Guard: the fresh-install tests above are not vacuous."""
    assert _JQ is not None
    deploy.set_listener(appear_after=1)
    kubeconfig = deploy.root / "ready-guard-kubeconfig"
    kubeconfig.write_text("kubelogin-converted: azurecli\n", encoding="utf-8")
    kubeconfig.chmod(0o600)
    process = subprocess.run(
        [
            "bash",
            str(deploy.bin / "kubectl"),
            "--namespace",
            ARC_CONTROLLER_NAMESPACE,
            "wait",
            "pod",
            "--selector",
            (
                f"actions.github.com/scale-set-name={RELEASE_NAME},"
                f"actions.github.com/scale-set-namespace={RUNNER_NAMESPACE}"
            ),
            "--for=condition=Ready",
            "--timeout=180s",
        ],
        env={
            "PATH": f"{deploy.bin}:{Path(_JQ).parent}:{_SYSTEM_PATH}",
            "HOME": str(deploy.home),
            "KUBECONFIG": str(kubeconfig),
            "FAKE_STATE_DIR": str(deploy.state),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode != 0
    assert "no matching resources found" in process.stderr


def test_a_listener_wait_outside_the_controller_namespace_is_rejected(
    deploy: DeployHarness,
) -> None:
    """Guard: the arc-systems assertion above is not vacuous."""
    assert _JQ is not None
    kubeconfig = deploy.root / "guard-kubeconfig"
    kubeconfig.write_text("kubelogin-converted: azurecli\n", encoding="utf-8")
    kubeconfig.chmod(0o600)
    process = subprocess.run(
        [
            "bash",
            str(deploy.bin / "kubectl"),
            "--namespace",
            RUNNER_NAMESPACE,
            "wait",
            "pod",
            "--selector",
            (
                f"actions.github.com/scale-set-name={RELEASE_NAME},"
                f"actions.github.com/scale-set-namespace={RUNNER_NAMESPACE}"
            ),
            "--for=condition=Ready",
            "--timeout=1s",
        ],
        env={
            "PATH": f"{deploy.bin}:{Path(_JQ).parent}:{_SYSTEM_PATH}",
            "HOME": str(deploy.home),
            "KUBECONFIG": str(kubeconfig),
            "FAKE_STATE_DIR": str(deploy.state),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode != 0
    assert "arc-systems" in process.stderr


@pytest.mark.parametrize(("path", "value", "fragment"), RUNNER_SET_DRIFT)
def test_installer_fails_when_the_installed_scale_set_drifts(
    deploy: DeployHarness, path: str, value: Any, fragment: str
) -> None:
    document = deploy.runner_set
    _set_path(document, path, value)
    deploy.set_runner_set(document)

    run = deploy.install()
    assert run.returncode != 0
    assert fragment in run.output


@pytest.mark.parametrize(
    "toleration", [pytest.param(item, id=item["key"]) for item in OLLAMA_TOLERATIONS]
)
def test_installer_rejects_a_runner_that_tolerates_a_model_taint(
    deploy: DeployHarness, toleration: dict[str, str]
) -> None:
    document = deploy.runner_set
    document["spec"]["template"]["spec"]["tolerations"].append(dict(toleration))
    deploy.set_runner_set(document)

    run = deploy.install()
    assert run.returncode != 0
    assert toleration["key"] in run.output


def test_installer_removes_its_temporary_directory_when_the_install_fails(
    deploy: DeployHarness,
) -> None:
    deploy.fail("helm-upgrade")
    run = deploy.install()
    assert run.returncode != 0
    assert list(deploy.tmpdir.iterdir()) == []
    assert deploy.operator_kubeconfig.read_text(encoding="utf-8") == OPERATOR_KUBECONFIG_MARKER


def test_installer_stops_when_the_cluster_credentials_cannot_be_downloaded(
    deploy: DeployHarness,
) -> None:
    deploy.fail("az-credentials")
    run = deploy.install()
    assert run.returncode != 0
    assert run.calls_with("kubectl") == []
    assert run.calls_with("helm") == []


# ---------------------------------------------------------------------------
# Task 4d: executed verifier behaviour
# ---------------------------------------------------------------------------


def test_verifier_passes_against_the_live_cluster_shape(deploy: DeployHarness) -> None:
    run = deploy.verify()
    assert_succeeded(run)
    assert run.calls, "the verifier checked nothing"


def test_verifier_only_reads(deploy: DeployHarness) -> None:
    run = deploy.verify()
    assert_succeeded(run)
    for call in run.calls:
        argv = call["argv"]
        assert not any(
            forbidden in argv
            for forbidden in ("apply", "create", "delete", "scale", "patch", "upgrade", "set")
        ), f"the verifier issued a mutating command: {argv}"


def test_verifier_uses_a_private_kubeconfig_and_cleans_it_up(deploy: DeployHarness) -> None:
    run = deploy.verify()
    assert_succeeded(run)
    assert list(deploy.tmpdir.iterdir()) == []
    assert deploy.operator_kubeconfig.read_text(encoding="utf-8") == OPERATOR_KUBECONFIG_MARKER


@pytest.mark.parametrize("missing", ["az", "gh", "helm", "kubectl", "kubelogin"])
def test_verifier_requires_every_tool(deploy: DeployHarness, missing: str) -> None:
    if missing in _SHADOWED_BY_SYSTEM:  # pragma: no cover - depends on the host
        pytest.skip(f"a real {missing} is reachable through {_SYSTEM_PATH}")
    deploy.remove_tool(missing)
    run = deploy.verify()
    assert run.returncode != 0
    assert f"required command not found: {missing}" in run.output
    assert run.calls == []


def test_verifier_fails_when_the_github_api_is_unavailable(deploy: DeployHarness) -> None:
    deploy.fail("gh-api")
    run = deploy.verify()
    assert run.returncode != 0
    assert "verify-grounding-deployment: FAIL" in run.output


def test_verifier_fails_when_the_environment_is_absent(deploy: DeployHarness) -> None:
    deploy.remove_environment()
    run = deploy.verify()
    assert run.returncode != 0
    assert "aks-grounding" in run.output


@pytest.mark.parametrize("variable", sorted(REQUIRED_ENVIRONMENT_VARIABLES))
def test_verifier_requires_every_grounding_variable(
    deploy: DeployHarness, variable: str
) -> None:
    deploy.set_environment_variables(sorted(REQUIRED_ENVIRONMENT_VARIABLES - {variable}))
    run = deploy.verify()
    assert run.returncode != 0
    assert f"environment variable missing: {variable}" in run.output


def test_verifier_requires_the_korvid_app_private_key_secret(deploy: DeployHarness) -> None:
    deploy.set_environment_secrets([])
    run = deploy.verify()
    assert run.returncode != 0
    assert "environment secret missing: KORVID_APP_PRIVATE_KEY" in run.output


def test_verifier_treats_the_reflection_credential_as_optional(deploy: DeployHarness) -> None:
    assert_succeeded(deploy.verify())
    deploy.set_environment_secrets(["GROUNDING_REFLECTION_CREDENTIAL", "KORVID_APP_PRIVATE_KEY"])
    assert_succeeded(deploy.verify())


def test_verifier_prints_names_but_never_a_value(deploy: DeployHarness) -> None:
    run = deploy.verify()
    assert_succeeded(run)
    assert "KORVID_APP_PRIVATE_KEY" in run.output
    assert VARIABLE_VALUE_MARKER not in run.output
    for leak in (SUBSCRIPTION_ID, AKS_ID, CLUSTER_ENDPOINT_MARKER):
        assert leak not in run.output


def test_verifier_fails_when_the_release_is_not_deployed(deploy: DeployHarness) -> None:
    deploy.set_release_status("failed")
    run = deploy.verify()
    assert run.returncode != 0
    assert "deployed" in run.output


@pytest.mark.parametrize(("path", "value", "fragment"), RUNNER_SET_DRIFT)
def test_verifier_fails_when_the_scale_set_drifts(
    deploy: DeployHarness, path: str, value: Any, fragment: str
) -> None:
    document = deploy.runner_set
    _set_path(document, path, value)
    deploy.set_runner_set(document)

    run = deploy.verify()
    assert run.returncode != 0
    assert fragment in run.output


@pytest.mark.parametrize(
    "toleration", [pytest.param(item, id=item["key"]) for item in OLLAMA_TOLERATIONS]
)
def test_verifier_rejects_a_runner_that_tolerates_a_model_taint(
    deploy: DeployHarness, toleration: dict[str, str]
) -> None:
    document = deploy.runner_set
    document["spec"]["template"]["spec"]["tolerations"].append(dict(toleration))
    deploy.set_runner_set(document)

    run = deploy.verify()
    assert run.returncode != 0
    assert toleration["key"] in run.output


def test_deployment_scripts_require_a_container_named_runner(deploy: DeployHarness) -> None:
    """Both scripts select the runner container by name, never by index."""
    document = deploy.runner_set
    document["spec"]["template"]["spec"]["containers"][0]["name"] = "sidecar"
    deploy.set_runner_set(document)

    install = deploy.install()
    assert install.returncode != 0
    assert "no container named 'runner'" in install.output

    verify = deploy.verify()
    assert verify.returncode != 0
    assert "no container named 'runner'" in verify.output


def test_deployment_scripts_read_the_runner_image_by_name_not_by_index(
    deploy: DeployHarness,
) -> None:
    """A container the controller prepends must not shift the image check."""
    document = deploy.runner_set
    containers = document["spec"]["template"]["spec"]["containers"]
    assert containers[0]["name"] == "runner"
    containers.insert(0, {"name": "init-dind", "image": "docker:27-dind"})
    deploy.set_runner_set(document)

    assert_succeeded(deploy.install())
    assert_succeeded(deploy.verify())


def test_deployment_scripts_reject_a_runner_container_on_an_unreviewed_image(
    deploy: DeployHarness,
) -> None:
    """Even at the right index, a drifted image must fail both scripts."""
    document = deploy.runner_set
    containers = document["spec"]["template"]["spec"]["containers"]
    containers.insert(0, {"name": "init-dind", "image": RUNNER_IMAGE})
    containers[1]["image"] = "acrpensionguard.azurecr.io/runner-base:v1"
    deploy.set_runner_set(document)

    install = deploy.install()
    assert install.returncode != 0
    assert RUNNER_IMAGE in install.output
    assert "image" in install.output

    verify = deploy.verify()
    assert verify.returncode != 0
    assert RUNNER_IMAGE in verify.output
    assert "image" in verify.output


def test_verifier_pins_the_live_ollama_node_selector(deploy: DeployHarness) -> None:
    document = deploy.ollama
    document["spec"]["template"]["spec"]["nodeSelector"] = {"workload": "modeleval"}
    deploy.set_ollama(document)

    run = deploy.verify()
    assert run.returncode != 0
    assert "korvid-model-eval" in run.output


@pytest.mark.parametrize(
    "dropped", [pytest.param(item["key"], id=item["key"]) for item in OLLAMA_TOLERATIONS]
)
def test_verifier_requires_both_model_tolerations_on_ollama(
    deploy: DeployHarness, dropped: str
) -> None:
    document = deploy.ollama
    pod = document["spec"]["template"]["spec"]
    pod["tolerations"] = [item for item in pod["tolerations"] if item["key"] != dropped]
    deploy.set_ollama(document)

    run = deploy.verify()
    assert run.returncode != 0
    assert f"does not tolerate {dropped}" in run.output


@pytest.mark.parametrize(
    ("count", "provisioning_state", "fragment"),
    [(2, "Succeeded", "count"), (0, "Failed", "provisioningState")],
)
def test_verifier_rejects_an_unhealthy_model_node_pool(
    deploy: DeployHarness, count: int, provisioning_state: str, fragment: str
) -> None:
    deploy.set_node_pool(count=count, provisioning_state=provisioning_state)
    run = deploy.verify()
    assert run.returncode != 0
    assert fragment in run.output


def test_verifier_rejects_a_workflow_that_targets_another_runner(
    deploy: DeployHarness,
) -> None:
    body = deploy.workflow.read_text(encoding="utf-8")
    updated = body.replace(
        f"runs-on: {RELEASE_NAME}\n", "runs-on: korvid-runners\n"
    )
    assert updated != body
    deploy.workflow.write_text(updated, encoding="utf-8")

    run = deploy.verify()
    assert run.returncode != 0
    assert "runs-on" in run.output


def test_verifier_rejects_an_artifact_upload_outside_the_safe_evidence_directory(
    deploy: DeployHarness,
) -> None:
    body = deploy.workflow.read_text(encoding="utf-8")
    updated = body.replace(
        f"          path: {SAFE_EVIDENCE_DIR}/\n",
        "          path: prompt-lab/artifacts/grounding-round/\n",
    )
    assert updated != body
    deploy.workflow.write_text(updated, encoding="utf-8")

    run = deploy.verify()
    assert run.returncode != 0
    assert "safe-evidence" in run.output


def test_verifier_ignores_an_unrelated_workflow_that_mentions_the_scale_set(
    deploy: DeployHarness,
) -> None:
    """A decoy workflow must not satisfy the grounding-round contract."""
    body = deploy.workflow.read_text(encoding="utf-8")
    deploy.workflow.write_text(
        body.replace(f"runs-on: {RELEASE_NAME}\n", "runs-on: ubuntu-latest\n"), encoding="utf-8"
    )
    decoy = deploy.repo / ".github/workflows/decoy.yml"
    decoy.write_text(
        "name: decoy\n"
        "on: push\n"
        "jobs:\n"
        "  decoy:\n"
        f"    runs-on: {RELEASE_NAME}\n"
        "    steps:\n"
        "      - uses: actions/upload-artifact@v4\n"
        "        with:\n"
        f"          path: {SAFE_EVIDENCE_DIR}/\n",
        encoding="utf-8",
    )

    run = deploy.verify()
    assert run.returncode != 0
    assert "runs-on" in run.output
