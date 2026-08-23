"""Offline contract tests for the Prompt Lab ARC runner and grounding-access files."""
from __future__ import annotations

import json
import shutil
import stat
import subprocess
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
    assert "--kubectl-version v1.35.6" in body
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
# Task 4 — installer and verifier script contract tests
# ---------------------------------------------------------------------------


def test_runner_installer_pins_arc_and_handles_secrets_through_files() -> None:
    body = (ROOT / "scripts/install-prompt-lab-runner.sh").read_text(encoding="utf-8")
    assert "gha-runner-scale-set" in body
    assert "--version 0.14.2" in body
    assert "--from-file=github_app_private_key=" in body
    assert "--from-literal" not in body
    assert "trap cleanup EXIT" in body


def test_deployment_verifier_is_read_only() -> None:
    body = (ROOT / "scripts/verify-grounding-deployment.sh").read_text(encoding="utf-8")
    for forbidden in ("kubectl apply", "kubectl delete", "helm upgrade", "az aks nodepool scale"):
        assert forbidden not in body
    assert "prompt-lab-runners" in body
    assert "modeleval" in body
    assert "safe-evidence" in body
