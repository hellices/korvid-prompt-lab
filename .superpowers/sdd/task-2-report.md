# Task 2 Report — Add Disjoint Qualification Inputs

## Status
COMPLETE

## Commit
- `80500397c4b0dbf33048552fef7715d05d6ed301` — `feat(campaigns): define bounded qualification inputs`
- `8d75196d59f878d6605c08c059b97ccce8d37b70` — `fix(campaigns): canonicalize model digests`

## Files Changed
- `examples/campaigns/aks-small-operator-qualification.yaml`
- `examples/optimization-campaigns/qwen3-small-operator.yaml`
- `src/korvid_prompt_lab/campaigns.py`
- `tests/test_campaigns.py`
- `tests/test_contracts.py`

## Summary
- Added the 12-case AKS qualification evaluation campaign with the exact Korvid template IDs/prompts from the brief.
- Added a strict optimization-campaign manifest domain with frozen/slotted dataclasses, fail-closed key validation, disjoint case-set enforcement, staged budgets/seeds, and immutable model-tier digests.
- Added a live `/api/tags` digest validator for model tiers that canonicalizes bare Ollama SHA-256 bytes to the shared publication-domain form before comparison.
- Resolved the live AKS Ollama `qwen3:0.6b` digest and wrote the canonical `sha256:<64 lowercase hex>` form into the manifest while preserving the exact live hash bytes.

## RED Evidence
Command:
```bash
uv run --python 3.12 pytest tests/test_campaigns.py tests/test_contracts.py -q
```
Result:
```text
ERROR tests/test_campaigns.py
ModuleNotFoundError: No module named 'korvid_prompt_lab.campaigns'
```
This was the expected RED state before implementation.

## GREEN Evidence
Command:
```bash
uv run --python 3.12 pytest tests/test_campaigns.py tests/test_contracts.py -q
```
Result:
```text
59 passed in 0.19s
```

Static checks:
```bash
uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaigns.py
uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
```
Result:
```text
All checks passed!
Success: no issues found in 1 source file
```

## Exact Digest Provenance
Live lookup was performed against the deployed AKS Ollama service after bringing the `modeleval` pool from `0` to `1`, then using the existing AKS port-forward preflight path to reach loopback and query `/api/tags`.

Commands used:
```bash
az aks nodepool show --resource-group rg-pension-guard --cluster-name aks-shared-runners --name modeleval --query count --output tsv --only-show-errors
az aks nodepool scale --resource-group rg-pension-guard --cluster-name aks-shared-runners --name modeleval --node-count 1 --only-show-errors
uv run --python 3.12 python - <<'PY'
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from korvid_prompt_lab.aks import AKSPortForward, AKSPreflightTransientError
from korvid_prompt_lab.contracts import AKSPortForwardServing

serving = AKSPortForwardServing(
    backend='aks_port_forward',
    resource_group='rg-pension-guard',
    cluster_name='aks-shared-runners',
    namespace='ollama',
    service='ollama',
    model='qwen3:0.6b',
    command=('korvid-bridge', '--request', '{request}', '--response', '{response}', '--turn-timeout', '300'),
)
last_error = None
for attempt in range(1, 19):
    try:
        with AKSPortForward(serving, workspace_dir=Path('.superpowers/sdd')) as forward:
            with urllib.request.urlopen(f'{forward.base_url}/api/tags', timeout=5) as response:
                payload = json.load(response)
            matches = [model.get('digest') for model in payload.get('models', []) if model.get('name') == 'qwen3:0.6b']
            if len(matches) != 1 or not isinstance(matches[0], str):
                raise SystemExit(f'Unexpected qwen3:0.6b digest entries: {len(matches)}')
            print(matches[0])
            raise SystemExit(0)
    except AKSPreflightTransientError as exc:
        last_error = exc
        time.sleep(10)
if last_error is not None:
    raise SystemExit(f'Transient preflight never stabilized: {last_error}')
raise SystemExit('No attempt completed')
PY
uv run --python 3.12 python - <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from korvid_prompt_lab.aks import AKSPortForward
from korvid_prompt_lab.contracts import AKSPortForwardServing

serving = AKSPortForwardServing(
    backend='aks_port_forward',
    resource_group='rg-pension-guard',
    cluster_name='aks-shared-runners',
    namespace='ollama',
    service='ollama',
    model='qwen3:0.6b',
    command=('korvid-bridge', '--request', '{request}', '--response', '{response}', '--turn-timeout', '300'),
)
with AKSPortForward(serving, workspace_dir=Path('.superpowers/sdd')) as forward:
    with urllib.request.urlopen(f'{forward.base_url}/api/tags', timeout=5) as response:
        payload = json.load(response)
match = [
    {
        'name': model.get('name'),
        'digest': model.get('digest'),
        'digest_type': type(model.get('digest')).__name__,
        'digest_len': len(model.get('digest')) if isinstance(model.get('digest'), str) else None,
    }
    for model in payload.get('models', [])
    if model.get('name') == 'qwen3:0.6b'
]
print(json.dumps(match, ensure_ascii=False))
PY
```
Observed results:
```text
initial node count: 0
resolved digest: 7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435
metadata: [{"name": "qwen3:0.6b", "digest": "7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435", "digest_type": "str", "digest_len": 64}]
```
Canonicalization decision: the deployed Ollama `/api/tags` returned the exact SHA-256 bytes as a bare 64-hex string. Task 2 now canonicalizes that wire value to `sha256:<64 lowercase hex>` for manifest storage and comparison, also accepts an already-prefixed live value, and rejects missing, duplicate, uppercase, wrong-length, unsupported-prefix, and non-string live digests as configuration errors.

## Cleanup Evidence
Commands:
```bash
az aks nodepool scale --resource-group rg-pension-guard --cluster-name aks-shared-runners --name modeleval --node-count 0 --only-show-errors
az aks nodepool show --resource-group rg-pension-guard --cluster-name aks-shared-runners --name modeleval --query count --output tsv --only-show-errors
find .superpowers/sdd -maxdepth 1 -name '.kubeconfig-*.yaml' -print
ps -ax -o pid=,command= | grep 'kubectl.*port-forward.*service/ollama' | grep -v grep || true
```
Results:
```text
restored node count: 0
no kubeconfig temp files left
no matching port-forward processes left
```

## Exact Commands / Results
1. RED test run: missing `korvid_prompt_lab.campaigns` module as expected.
2. Live AKS digest retrieval: resolved bare live digest `7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`, then stored and exposed it canonically as `sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`.
3. Final verification:
   - `uv run --python 3.12 pytest tests/test_campaigns.py tests/test_contracts.py -q` → `59 passed in 0.19s`
   - `uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaigns.py` → `All checks passed!`
   - `uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py` → `Success: no issues found in 1 source file`
4. Commit:
   - `git commit -m "feat(campaigns): define bounded qualification inputs" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"`
   - Result: created commit `80500397c4b0dbf33048552fef7715d05d6ed301`
5. Review-fix commit:
   - `git commit -m "fix(campaigns): canonicalize model digests" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"`
   - Result: created commit `8d75196d59f878d6605c08c059b97ccce8d37b70`

## Self-Review
- Confirmed the evaluation campaign contains exactly the 12 required case IDs, template IDs, and prompts.
- Confirmed train/validation/milestone sets are pairwise disjoint and exactly cover the evaluation campaign.
- Confirmed manifest validation is fail-closed for unknown keys, missing required fields, empty sets, duplicate seeds, invalid budgets, mutable/invalid manifest digests, and unknown keys inside `model_tiers[0]`.
- Confirmed the committed canonical model-tier digest matches the exact live AKS `/api/tags` hash bytes after canonicalization, whether the live endpoint returns bare or canonical SHA-256 text.

## Concerns
- The live digest verifier is implemented as a dedicated helper in `src/korvid_prompt_lab/campaigns.py`; later controller/orchestrator tasks will need to call it at the pre-allocation decision point.

## Review Fix Follow-up
### Canonicalization adjudication
- Manifest and public campaign state use only `sha256:<64 lowercase hex>`.
- Live Ollama `/api/tags` digests are canonicalized from either bare `64`-hex bytes or already-prefixed `sha256:<64 lowercase hex>` before comparison.
- Uppercase, wrong-length, unsupported-prefix, duplicate, missing, and non-string live digests are configuration errors.

### RED Evidence for review fixes
Command:
```bash
uv run --python 3.12 pytest tests/test_campaigns.py tests/test_contracts.py -q
```
Observed failure after tightening tests but before code changes:
```text
FAILED tests/test_campaigns.py::test_loads_bounded_disjoint_campaign
AssertionError: assert '7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435' == 'sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435'
```

### GREEN Evidence for review fixes
Commands:
```bash
uv run --python 3.12 pytest tests/test_campaigns.py tests/test_contracts.py -q
uv run --python 3.12 ruff check src/korvid_prompt_lab/campaigns.py tests/test_campaigns.py
uv run --python 3.12 mypy src/korvid_prompt_lab/campaigns.py
```
Results:
```text
66 passed in 0.34s
All checks passed!
Success: no issues found in 1 source file
```
