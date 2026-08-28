# llm-d causal SLO-externality reproduction

This repository is a pinned research snapshot of the llm-d Router port of the
INFOCOM causal SLO-externality policy, its proxy-side observability, and the
first Qwen3-14B real-stack prediction-versus-observation experiment.

It is based on upstream `llm-d/llm-d-router` tag `v0.9.0`, commit
`5f4e762f341a5196393ce79f8a57c3e1900c4a6b`. It is not an upstream llm-d
branch. The exact code, images, dependencies, model caveat, and platform are in
[reproduction/VERSIONS.md](reproduction/VERSIONS.md).

## 60-second status

- Michael's proxy-style port provides the causal joint P/D candidate scorer,
  shadow state, and llm-d integration without modifying vLLM.
- This snapshot adds request-keyed EPP/sidecar events and an offline analyzer
  so each prediction can be joined to client-, proxy-, and NIXL-observed time.
- R5 ran Qwen3-14B BF16, TP=1, with one prefiller and two decoders on three
  H100 80 GB GPUs. All 20 requests completed; the policy chose 17 local and 3
  disaggregated placements.
- Strict replay joins 20/20 requests. All 76 candidate rows satisfy the logged
  arithmetic identities; maximum error is `3.64e-12 us`.
- Local client TTFT MAPE is 6.87%, remote prefill MAPE is 3.68%, and early
  decode-iteration MAPE is 5.72% local / 8.62% remote.
- The important negative result is the remote path: the one unambiguous
  transfer was 599.502 ms observed versus 200.627 ms predicted, and the remote
  post-prefill tail has 1.058 s MAE.
- This demonstrates real-stack execution and validates several component
  predictions. It does **not** yet prove counterfactual SLO externality or an
  end-to-end policy advantage over a baseline.

## Reproduce R5 locally

Only Python 3, `make`, and standard Unix utilities are needed. No GPU, cluster,
Python packages, or access to the original environment is required.

```bash
git clone https://github.com/vishakha-ramani/llmd-causal-slo-externality-repro.git
cd llmd-causal-slo-externality-repro
make -C reproduction verify
```

Expected final lines:

```text
Ran 5 tests in ...
OK
R5 strict joins and byte-for-byte result replay: PASS
```

The command:

1. tests the analyzer, including compressed-log ingestion;
2. replays all 10,736 packaged causal events in strict mode;
3. requires complete client/EPP/sidecar joins and exact logged identities; and
4. compares the regenerated JSON and both CSVs byte-for-byte with the committed
   R5 results.

Generated files are written under `reproduction/out/r5/`. To run the analyzer
directly:

```bash
python3 reproduction/scripts/analyze-causal-observability.py \
  reproduction/data/r5/events.jsonl.gz \
  --client-trace reproduction/data/r5/trace_data.csv \
  --transfer-windows reproduction/data/r5/nixl-transfer-windows.csv \
  --report-out /tmp/prediction-validation.json \
  --rows-out /tmp/prediction-residuals.csv \
  --externality-rows-out /tmp/resident-prediction-residuals.csv \
  --strict
```

## Verify and build the source

The focused tests cover the causal policy, timing identities, event schema,
sidecar request-ID propagation, and the `min_tokens`/`max_tokens` P/D fix:

```bash
go test -race \
  ./pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality \
  ./pkg/sidecar/proxy
```

The upstream build also supports containerized tests and image builds:

```bash
make test-unit
IMAGE_REGISTRY=quay.io/YOUR_QUAY_NAMESPACE \
  EPP_TAG=causal-qwen-repro \
  SIDECAR_TAG=causal-qwen-repro \
  make image-build
```

The published R5 images are already public, so rebuilding is optional. Do not
push to `llm-d/llm-d-router`; use a registry and repository you own.

## Generate the real-cluster experiment

Ruby is needed only to generate a new PipelineRun:

```bash
make -C reproduction verify-manifest
```

For a cluster-specific manifest, supply your own namespace, service account,
PVCs, secret, and optional image/repository overrides:

```bash
cd reproduction
export NAMESPACE=my-namespace
export SERVICE_ACCOUNT=my-tekton-service-account
export DATA_PVC=my-results-pvc
export SOURCE_PVC=my-source-pvc
export HF_SECRET=my-huggingface-secret
export PIPELINE_NAME=sim2real
export RUN_NAME="qwen-causal-$(date +%Y%m%d-%H%M%S)"

PIPELINERUN_TEMPLATE="$PWD/scripts/causal-qwen-overlap-20260827-151229.yaml" \
RATE=1 REQUESTS=20 INPUT_TOKENS=15000 OUTPUT_TOKENS=512 \
ruby scripts/generate-qwen-iterlaw-run.rb \
  causal "$RUN_NAME" "out/$RUN_NAME.yaml"
```

The complete preflight, submission, collection, and analysis commands are in
[reproduction/CLUSTER.md](reproduction/CLUSTER.md). A full R5-equivalent run
requires the llm-d-benchmark `sim2real` Tekton pipeline, two shared PVCs,
RDMA/NIXL/Multus support, and three H100 80 GB GPUs. Those are infrastructure
requirements, not bundled software dependencies.

## Iteration-time law being tested

For one GPU iteration, the implementation adds the marginal work of every
scheduled decode request and every scheduled prefill grant:

```text
T_iter = alpha
       + C0 * B_decode
       + C1 * sum(decode context tokens)
       + C_pf * sum(prefill grant tokens)
       + C_attn * sum(grant * (prefix + grant/2))
```

The intercept is charged once for the complete batch. A dedicated prefiller
sets the two decode terms to zero; a mixed decoder iteration retains all five
terms. The implementation is in
[`coeffs.go`](pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality/coeffs.go),
and the direct marginal-sum test is in
[`score_test.go`](pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality/score_test.go).

R5 deliberately accepted only clean snapshots and therefore did not itself
exercise live mixed resident-prefill terms. Independent calibration used
decode-only and prefill-only data for fitting and held 72 mixed rows out for
validation (6.96% MAPE). See [docs/VALIDATION-R5.md](docs/VALIDATION-R5.md) and
[reproduction/TODO.md](reproduction/TODO.md) for the evidence and remaining
coefficient study.

## Repository map

- `pkg/epp/.../causalsloexternality/`: joint P/D policy and decomposed logging.
- `pkg/sidecar/proxy/`: proxy boundary, token-event, and transfer correlation.
- `reproduction/`: one-command replay, portable generator, exact manifests,
  minimum event data, committed outputs, versions, and follow-up plan.
- `docs/PORTING.md`: how the BLIS policy maps into llm-d.
- `docs/VALIDATION-R5.md`: measurement contract and result interpretation.
- `docs/ITERATION-LAW-AUDIT.md`: Qwen coefficient and marginal-law audit.
- `docs/SESSION-CHECKPOINT.md`: detailed experiment history and operational
  notes.
- `reproduction/patches/`: standalone sidecar fix for comparison with the
  pinned upstream tree; the fix is already applied in this repository.
- `docs/UPSTREAM-README.md`: README from the pinned upstream router snapshot.

Here, **stock** means the comparison arm using the upstream/default llm-d
routing policy and configuration. It does not mean unmodified vLLM, and it
does not refer to Michael's port or the causal policy.

## Claim boundary and next experiment

The current evidence supports the plumbing, logged formula identities, local
first-token prediction, remote prefill prediction, and early iteration model.
It also identifies two real gaps: loaded transfer/post-prefill timing and the
unobserved counterfactual for resident requests.

The next claim-bearing experiment is a matched causal-versus-stock study with
an explicit output-estimator warmup, repeated seeds, controlled overlap, and a
fixed-snapshot replay or paired intervention for resident externality. Report
TTFT/ITL/E2E goodput, placement decisions, confidence intervals, and the
prediction residuals alongside policy outcomes.
