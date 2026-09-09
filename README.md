# Korvid Prompt Lab

## 저장소 역할

**Korvid가 실제로 사용하는 프롬프트를 원본 그대로 가져와,
Korvid의 원본 시나리오/journey와 채점으로 최적화합니다.**

- Korvid 소유: tier\_pack 원본, 실행기, 시나리오/journey, fixture, 턴 순서, 채점
- Prompt Lab 소유: 원본 수집, 후보 생성, 반복 평가 오케스트레이션, 결과 산출

Korvid의 프롬프트는 여러 레이어와 동적 문맥으로 구성됩니다. 원본 운영 프롬프트,
고정 레이어와 원본 문맥에서의 합성 결과를 snapshot으로 수집하고 동일성을 검증합니다.
현재 버전의 Korvid eval API가 제공하는 교체 지점인 `PromptGrind.tier_pack`으로
**운영 프롬프트 원문 자체를 수정**합니다. 추가 `agent.rules` 전용 탐색이 아닙니다.
안전 계약·공통 역할·동적 문맥은 Korvid 구성기를 그대로 사용합니다.

원본 설계 사양은 역사적 참조로 git 이력에 보존됩니다. 이 README가 해당 사양을 대체하지 않습니다.

---

## 아키텍처

```text
Prompt Lab: experiment.yaml -> model_session (AKS 또는 loopback)
            -> upstream.py (inspect / runner / export)
            -> source_runtime.py (격리된 worker 호출)
Korvid 전용 환경: upstream_worker.py
            -> Korvid 원본 prompt_packs + PromptHarness
            -> 원본 loader + run_scenario / run_journey + grader
Prompt Lab: 원본 판정 -> GEPA 탐색 -> DSPy/teacher 후보 제안
            -> 원본 시나리오 재평가 -> 프롬프트 원문 / diff / 근거
```

- `native_thinking: true` → Korvid `/api/chat` 경로 선택 (LiteLLM `/api/generate` 변환 손실 회피)
- `think: false` / `num_ctx: 16384` 는 별개
- 모델·런타임 프로파일은 비교 중 고정; **오직 tier\_pack 텍스트만** 변경
- 후보 작성: DSPy `Signature`/`Predict` + teacher. 탐색: standalone `gepa.optimize`.
  Korvid를 DSPy `ReAct`로 복제하거나 `dspy.GEPA`/MIPROv2를 실행하는 구성이 아닙니다.

### 이 구조를 선택한 근거

- 프롬프트·시나리오·grader가 같은 Korvid 커밋에 묶여야 결과를 Korvid 개선으로 해석할 수 있습니다.
  실행기만 재사용하고 문제나 성공 조건을 새로 만들었던 이전 접근은 이 계약을 충족하지 못했습니다.
- Korvid 자체 `PromptGrind`가 운영 프롬프트 교체를 지원하므로 별도 에이전트나 자체 oracle이
  필요하지 않습니다. 원본 snapshot과 `NO_GRIND` 비교는 잘못된 baseline을 차단합니다.
- GEPA는 외부 평가 결과를 사용하는 텍스트 탐색기이므로 이 역할에 맞습니다.
  DSPy는 reflection만 담당합니다. 프레임워크 선택이 시나리오 재작성의 이유가 될 수는 없습니다.
- 서빙과 평가를 분리해 같은 평가를 로컬 endpoint 또는 AKS Ollama에 연결할 수 있습니다.
- MIPROv2 등은 동일한 프롬프트 적용·평가 계약에서 비교 가능한 후속 대안입니다.
  어떤 옵티마이저도 실제 개선을 보장하지 않습니다.

참고: [GEPA](https://gepa-ai.github.io/gepa/),
[GEPA 논문](https://arxiv.org/abs/2507.19457),
[DSPy MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/).

---

## 평가 분할 (기본값)

원본 Korvid 시나리오/journey ID 선택·분할. 현재 corpus: 시나리오 25개, journey 8개 (총 33개).
operations/live journey는 범위 밖(비지원).

```yaml
evaluation:
  repetitions: 5
  seed: 0
  train:
    - scenarios/image-pull-typo
    - scenarios/healthy-deployment
    - journeys/logs-to-events
  validation:
    - journeys/tui-follow
    - scenarios/oom-killed
  holdout:
    - journeys/compare-namespaces
    - scenarios/readiness-probe-failing
```

이 분할은 **보정(calibration) 목적**이며 순수 UI 벤치마크가 아닙니다.
현재 corpus는 독립적인 UI 전용 3분할 세트를 제공하지 않습니다 — 공백을 덮거나
모든 케이스가 순수 UI 평가에 적합하다고 주장하지 않습니다.
`tui-follow`는 diagnosis → describe → logs 순서를 포함하는 원본 journey입니다;
마지막 2 턴만 추출하거나 문구·namespace·fixture·조건을 변경하지 않습니다.
사용자는 원본 ID를 다른 것으로 선택할 수 있지만, 재작성/슬라이스는 안 됩니다.

채점: scenario는 Korvid의 `run.outcome == "success"`, journey는 원본
`run.success`를 그대로 사용합니다. Lab의 scalar는 이 원본 판정의 0/1 표현입니다.
`pass^k` = 첫 k회 모두 성공. `pass@k` = k회 중 1회 이상 성공 (다릅니다).
원본 grades/behavior 메트릭 보존. Lab이 새 성공 조건으로 대체 불가.

---

## 1. 설치

```bash
uv sync --python 3.12 --extra dev
uv run --extra dev --python 3.12 korvid-prompt-lab --help
```

`dev`에는 `pytest`/`mypy`/`ruff` 포함. **`--extra dev` 없이 동기화하지 마세요.**

---

## 2. 소스 환경 준비

```bash
git clone --branch v0.4.1 --depth 1 https://github.com/hellices/korvid.git /path/to/korvid-native
git -C /path/to/korvid-native rev-parse HEAD
# 반드시: 33c483e041006eb20259a024ed85a9323e52c8f0

uv venv --python 3.12 /path/to/korvid-native/.venv
uv pip install --python /path/to/korvid-native/.venv/bin/python -e '/path/to/korvid-native[agent]'

export KORVID_NATIVE_SOURCE_ROOT=/path/to/korvid-native
export KORVID_NATIVE_MODEL_URL=http://127.0.0.1:11434
```

Korvid는 SHA `33c483e041006eb20259a024ed85a9323e52c8f0`에 고정됩니다. 별도 `.venv`를 사용합니다.
런타임/툴/예산(반복 6, history 24000, result 3000, 턴당 툴 1)은 **변경 없이** 사용합니다.
락 재현 주의: 미러가 락을 공급 못 하면 선언된 의존성만 설치하되, 서로 다른 환경의 결과를 섞지 마세요.

---

## 3. 실험 선언 (schema\_version 2)

runtime/serving/model/reflection 프로파일, 탐색 예산, 평가 매니페스트를 **직교적으로** 선언합니다.
실행 가능한 예시: [`examples/experiments/native-loopback.yaml`](examples/experiments/native-loopback.yaml),
[`examples/experiments/native-aks.yaml`](examples/experiments/native-aks.yaml).

주요 필드 (전체는 예시 파일 참조):

```yaml
schema_version: 2
runtime:
  backend: korvid_upstream        # upstream.py / KorvidUpstreamRunner
  korvid_revision: 33c483e041006eb20259a024ed85a9323e52c8f0
  timeout_seconds: 600            # 전체 journey용 (턴별 슬라이스 없음)
serving:
  backend: loopback               # 또는 aks_port_forward
model:
  reference: ollama/qwen3:0.6b
  digest: sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435
  options: { native_thinking: true, think: false, num_ctx: 16384, temperature: 0.0 }
reflection:
  reference: ollama_chat/qwen3:14b
  digest: sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
  timeout_seconds: 360
  options: { reasoning_effort: disable, num_ctx: 4096, temperature: 0.2, max_tokens: 512 }
evaluation:
  repetitions: 5
  seed: 0
  train: [...]
  validation: [...]
  holdout: [...]
search:
  stages: [{ name: explore, metric_calls: 16, seeds: [0, 1] }]
  total_metric_calls: 64
  max_evaluations: 256
  max_proposals: 12
  wall_clock_seconds: 7200
  stagnation_attempt_limit: 3
```

---

## 4. 캠페인 실행

```bash
# 오프라인 검증 (소스 검사 + baseline 동일성, 모델 호출 없음)
uv run --extra dev korvid-prompt-lab run \
  --experiment examples/experiments/native-loopback.yaml \
  --artifact-root artifacts/native-001 --check-only

# 실제 실행 (AKS는 native-aks.yaml + --allow-capacity-changes)
uv run --extra dev korvid-prompt-lab run \
  --experiment examples/experiments/native-aks.yaml \
  --artifact-root artifacts/native-001 --allow-capacity-changes
```

Resume 미지원. 기존 artifact 루트 거부.

### 파이프라인 흐름

1. 소스 검사 + baseline 동일성 검증 (NO\_GRIND, 비용 전 오프라인 가능)
2. 단일 AKS loopback 세션 + 대상/teacher 다이제스트 양쪽 확인
3. Train-only 툴 canary (최대 3회)
4. 원본 baseline validation
5. GEPA + DSPy 후보 생성 (upstream source grades/train trace 사용; train+validation만 접근)
6. Paired validation → 개선 시에만 champion 교체
7. Freeze → fresh paired validation (frozen)
8. Holdout 1회 (재진입 불가)
9. Export(`export_upstream_prompt()`): `original-prompt.txt`, `optimized-prompt.txt`,
   `prompt.diff`, `application-manifest.json` 파일 기록 (추론 없음).
   파이프라인이 예산 내 train 케이스 1회 실행으로 `prompt_override_verified` 확정.

개선 후보 없으면 holdout·export 건너뜀 (`NOT_CONVERGED`).
`PREFLIGHT_INCONCLUSIVE` ≠ 깨진 라우트 증명. holdout 실패 시 최적화 재진입 불가.
Provider 실패 → 프롬프트 점수 없음. 자동 재시도·holdout 피드백 검색 없음.

### 종료 코드

| 코드 | 의미 |
|------|------|
| `0`  | QUALIFIED |
| `3`  | NOT\_CONVERGED |
| `2`  | 잘못된 설정 / 소스 검증 실패 |
| `1`  | 시스템 오류 / PREFLIGHT\_INCONCLUSIVE |
| `130`| 취소 |

---

## 5. 서빙 백엔드

- `loopback`: `base_url`은 `/v1` 없는 HTTP 루트.
- `aks_port_forward`: `resource_group`, `cluster_name`, `namespace`, `service` (선택 `node_pool`).

양쪽 모두 대상/teacher 다이제스트를 `/api/tags` 라이브로 검증. 불일치 시 중단.

AKS: 기본 변경 없음. `node_pool` 0일 때 `--allow-capacity-changes`로만 0→1.
우리가 올린 풀만 정리 시 복원. **SIGKILL 시 정리 보장 불가** — artifact 확인 후 수동 회수.

현재 풀(`Standard_D32s_v5`)은 GPU 없음. 3.5 CPU/10 GiB 병목.
14B teacher warm ~80–111초, cold 180초 초과 관측 → 360초 허용. 예산 선언 필수.

---

## 6. 채점·승격·Export

**채점**: 원본 scenario outcome / journey success의 0/1 판정. Lab의 UI 조건이나
가중치로 성공 여부를 대체하지 않습니다.

**승격 플래그**:
- `validation_improved`: 초기 validation에서 원본 판정의 `success_rate` 상승
- `prompt_improved`: 초기·frozen 양쪽 성공률 이득 + holdout 회귀 없음 + hard-safety 0
- `qualified`: 초기·frozen·holdout에서 원본 판정 기준 `pass^3 = pass^5 = 1`,
  hard-safety 실패·core 회귀 없음, prompt reload 검증 및 export 점검 case 성공.
  호출 수 감소만으로 원본 판정 점수가 오르지는 않습니다.

정리/시스템 오류/취소 시 `qualified`·`prompt_improved` → `false`.

**Export** (개선 후보 선택 시에만):

```
baseline-prompt.txt           # Korvid에서 가져온 실제 운영 프롬프트 원문
baseline-candidate.yaml       # 원문을 그대로 담은 최초 후보
baseline-snapshot.json        # 레이어/합성 동일성 및 source provenance
candidate-for-review/
  original-prompt.txt
  optimized-prompt.txt
  prompt.diff
  application-manifest.json   # static 검증 및 후속 reload receipt 경로
  application-verification.json  # 실제 budgeted reload 후에만 생성
```

`application-verification.json`은 컨트롤러가 예산에 포함된 원본 케이스 평가(`prompt_path` 리로드 확인)
완료 후 **불변 파일로 별도 기록**합니다:
```
application-verification.json  # prompt_override_verified=true,
                                # source_sha256=<원본 case 해시>,
                                # candidate_fingerprint=<후보 식별자>
```

`export_upstream_prompt()`는 파일만 기록하며 추론을 실행하지 않습니다 — `prompt_validation_passed`는
inspect-only 단계에서 설정됩니다. `prompt_override_verified`(원본 케이스 재현 충실성)는
receipt 및 실험 summary에 기록되며 inspect-only export 시점에는 설정되지 않습니다.
`product_application_verified`는 항상 `false`입니다. `QUALIFIED`는 평가 자격이며 자동 제품 배포가 아닙니다.
요약에서 `export_check_success`(원본 케이스 판정)와 `prompt_override_verified`(충실성)는 별도 필드입니다.

`PromptGrind`는 **평가 전용** 인터페이스입니다. 개선된 운영 프롬프트를 제품에 적용하려면
Korvid의 해당 prompt-pack 변경을 별도 검토해야 합니다. Lab은 Korvid 소스나 사용자 설정을
변경하지 않으며, 교체 프롬프트를 `agent.rules`에 넣으면 같다고 주장하지 않습니다.

---

## 7. 검증 명령

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev mypy src tests
uv run --extra dev ruff check src tests

KORVID_NATIVE_SOURCE_ROOT=/path/to/korvid-native \
  uv run --extra dev python -m pytest \
  tests/test_upstream_contract.py tests/test_upstream_application.py tests/test_upstream_runner.py \
  tests/test_experiment_http.py -q

KORVID_NATIVE_SOURCE_ROOT=/path/to/korvid-native \
  PYTHONPATH="$PWD/src:/path/to/korvid-native/src" \
  /path/to/korvid-native/.venv/bin/python -m pytest \
  tests/test_upstream_worker.py -q

# upstream_worker 타입 검사 (Korvid 전용 환경 인터프리터 사용)
MYPYPATH="$PWD/src/korvid_prompt_lab:/path/to/korvid-native/src" \
  uv run --extra dev mypy --explicit-package-bases \
  --python-executable /path/to/korvid-native/.venv/bin/python \
  src/korvid_prompt_lab/upstream_worker.py tests/test_upstream_worker.py
```

테스트는 upstream source/runtime 통합과 fake HTTP 파이프라인을 검증하며 모델 품질 증거가 아닙니다.

---

## 8. 소스 정합성

구현 확인 위치: [실험 컨트롤러](src/korvid_prompt_lab/experiment.py),
[source bridge](src/korvid_prompt_lab/upstream.py),
[원본 평가 worker](src/korvid_prompt_lab/upstream_worker.py),
[source 계약](src/korvid_prompt_lab/upstream_contract.py),
[source runtime](src/korvid_prompt_lab/source_runtime.py).
구현이 바뀌면 이 README도 갱신하되, 역할 계약을 덮어쓰지 마세요.
