# Full Qwen 1P2D cluster rerun

The offline replay is portable to any machine. This runbook recreates the real
serving experiment and therefore assumes equivalent cluster infrastructure.

## 1. Prerequisites

Install `oc`, `tkn`, Ruby with YAML support, and `python3`. The namespace must
provide:

- Tekton pipeline `sim2real` from llm-d-benchmark commit
  `76473d05a0f546a6bc0c4662ed7e34104da73891`;
- a Tekton service account allowed to create the serving resources;
- shared data and source PVCs;
- a Hugging Face token secret;
- Multus plus an RDMA network/resource usable by NIXL/UCX; and
- three schedulable H100 80 GB GPUs.

Set names for your cluster and verify them before creating anything:

```bash
cd reproduction

export NAMESPACE=my-namespace
export PIPELINE_NAME=sim2real
export SERVICE_ACCOUNT=my-tekton-service-account
export DATA_PVC=my-results-pvc
export SOURCE_PVC=my-source-pvc
export HF_SECRET=my-huggingface-secret
export RUN_NAME="qwen-causal-$(date +%Y%m%d-%H%M%S)"

oc whoami
oc -n "$NAMESPACE" get pipeline "$PIPELINE_NAME"
oc -n "$NAMESPACE" get serviceaccount "$SERVICE_ACCOUNT"
oc -n "$NAMESPACE" get pvc "$DATA_PVC" "$SOURCE_PVC"
oc -n "$NAMESPACE" get secret "$HF_SECRET"
oc -n "$NAMESPACE" get pods -o wide
oc -n "$NAMESPACE" get pipelineruns
```

Do not delete or modify another user's PipelineRuns, deployments, images,
PVCs, secrets, or service accounts.

## 2. Generate and validate a unique PipelineRun

The public EPP and sidecar images are the defaults. Override `EPP_REGISTRY`,
`EPP_REPOSITORY`, `EPP_TAG`, `SIDECAR_REPOSITORY`, or `SIDECAR_TAG` if using
images from your own registry.

```bash
mkdir -p out

PIPELINERUN_TEMPLATE="$PWD/scripts/causal-qwen-overlap-20260827-151229.yaml" \
NAMESPACE="$NAMESPACE" \
PIPELINE_NAME="$PIPELINE_NAME" \
SERVICE_ACCOUNT="$SERVICE_ACCOUNT" \
DATA_PVC="$DATA_PVC" \
SOURCE_PVC="$SOURCE_PVC" \
HF_SECRET="$HF_SECRET" \
RATE=1 REQUESTS=20 INPUT_TOKENS=15000 OUTPUT_TOKENS=512 SEED=4242 \
ruby scripts/generate-qwen-iterlaw-run.rb \
  causal "$RUN_NAME" "out/$RUN_NAME.yaml"

ruby -ryaml -e 'YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true); puts "YAML: PASS"' \
  "out/$RUN_NAME.yaml"

oc -n "$NAMESPACE" create --dry-run=server \
  -f "out/$RUN_NAME.yaml" -o name
```

The model repository name is pinned, but its Hugging Face revision was not
pinned in R5. Pin and record a revision before using a rerun for a bit-exact or
longitudinal comparison.

## 3. Install the log task and submit

The PipelineRun overrides its `stream-epp-logs` task with the Task below. It
captures both EPP and `routing-proxy` events without vLLM instrumentation.

```bash
oc -n "$NAMESPACE" apply -f manifests/stream-epp-and-sidecar-logs-task.yaml
oc -n "$NAMESPACE" create -f "out/$RUN_NAME.yaml"
tkn -n "$NAMESPACE" pipelinerun logs "$RUN_NAME" -f
```

In another terminal:

```bash
oc -n "$NAMESPACE" get taskruns \
  -l "tekton.dev/pipelineRun=$RUN_NAME" -w
```

Acceptance requires all PipelineRun tasks and all 20 requests to succeed,
exactly 512 generated tokens per request, no missing remote proxy IDs, and no
silent remote-to-local KV fallback.

## 4. Copy the result directory from the data PVC

Create a private reader pod that mounts only your configured data PVC:

```bash
oc -n "$NAMESPACE" run "$RUN_NAME-results-reader" \
  --image=registry.access.redhat.com/ubi9/ubi-minimal \
  --restart=Never \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"reader\",\"image\":\"registry.access.redhat.com/ubi9/ubi-minimal\",\"command\":[\"sleep\",\"86400\"],\"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"}]}],\"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"$DATA_PVC\"}}]}}"

oc -n "$NAMESPACE" wait --for=condition=Ready \
  "pod/$RUN_NAME-results-reader" --timeout=120s

oc -n "$NAMESPACE" exec "$RUN_NAME-results-reader" -- \
  find /data -type f -path "*$RUN_NAME*" -name trace_data.csv
```

Use the printed directory containing `trace_data.csv` as `RESULT_REL`:

```bash
export RESULT_REL=path/under/data/printed/above
mkdir -p "out/$RUN_NAME-results"
oc -n "$NAMESPACE" cp \
  "$RUN_NAME-results-reader:/data/$RESULT_REL/." \
  "out/$RUN_NAME-results"
```

Keep the generated PipelineRun beside the copied results. Delete only the
temporary reader pod you created:

```bash
oc -n "$NAMESPACE" delete pod "$RUN_NAME-results-reader"
```

## 5. Analyze the new run

The copied directory must include `trace_data.csv`, EPP logs, and sidecar logs.
If your collector names them differently, pass the actual files/directories as
the positional event arguments.

```bash
export RESULT_DIR="$PWD/out/$RUN_NAME-results"

python3 scripts/analyze-causal-observability.py \
  "$RESULT_DIR/epp_logs" \
  "$RESULT_DIR/sidecar_logs" \
  --client-trace "$RESULT_DIR/trace_data.csv" \
  --report-out "$RESULT_DIR/prediction-validation.json" \
  --rows-out "$RESULT_DIR/prediction-residuals.csv" \
  --externality-rows-out "$RESULT_DIR/resident-prediction-residuals.csv" \
  --strict
```

Pass `--transfer-windows FILE.csv` only when the before/after NIXL counters
were sampled around individual requests. The analyzer attributes a transfer
only when that window contains exactly one successful transfer.

The R5 expected values and claim boundary are in
[`../docs/VALIDATION-R5.md`](../docs/VALIDATION-R5.md). A different cluster or
model revision need not reproduce identical latencies; it must reproduce the
join coverage and algebraic identities before its residuals are interpreted.
