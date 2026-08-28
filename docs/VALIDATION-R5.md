# Causal SLO prediction validation on llm-d

## What this experiment observes

One request first reaches the EPP, where the policy scores every local and
remote placement. The chosen request then reaches the decode pod's routing
sidecar. For a remote placement, that sidecar sends a prefill request, waits
for its HTTP response, and then sends the decode request. For a local
placement, it sends the request directly to the decoder. The response stream
then passes back through the sidecar and EPP to the client.

The logging added in the local worktree records those boundaries without
modifying vLLM. It can therefore test whether the policy predicts the
client-visible and proxy-visible timeline. It cannot expose a hidden vLLM
scheduler decision as if that decision had been measured.

All emitted records use schema `causal-slo-externality/v1` and carry:

```text
event
requestID          incoming EPP/client x-request-id
proxyRequestID     sidecar UUID used on both remote vLLM legs
wallTimeUnixNs     cross-process join timestamp
elapsedUs          time since the local component's request/decision start
placement
decodeEndpoint
prefillEndpoint
```

The EPP and sidecar clocks need not have identical monotonic origins.
Durations are computed inside one process. Cross-process joins use the two
request IDs and wall time only.

## What is predicted and what is observed

| Quantity under test | Policy prediction | Observation | Interpretation |
|---|---|---|---|
| TTFT | `projectedTTFTUs` | client TTFT and EPP first-token callback | End-to-end or callback composite, not pure engine time |
| Remote prefill completion | `prefillAdmissionUs + prefillChunks * prefillIterUs + prefillWorkUs` | first `prefill_dispatched` through `prefill_completed` | Sidecar HTTP round trip, including vLLM admission and response handling |
| KV transfer | `transferUs` | NIXL transfer-time counter delta, only if exactly one transfer completed in the sampling window | Request-attributable only in an unambiguous window; the full `remoteLeadUs` is not directly observable |
| Local dispatch to first event | `projectedTTFTUs` | `decode_dispatched` through `decode_first_event` | Includes local admission, chunked prefill, engine work, HTTP, and buffering |
| Remote post-prefill tail | `max(transferUs, decodeAdmissionUs - prefillCompletionUs, 0) + firstDecodeIterUs + outputTokenProcessingUs` | `decode_dispatched` through `decode_first_event` | The sidecar dispatches before KV synchronization, so this boundary includes transfer, remaining queueing, engine work, HTTP, and buffering |
| Early decode iteration | `postJoinIterUs` | median of the first 16 positive SSE token gaps, divided by token delta | Closest non-invasive proxy for the post-join batch; still not a pure engine measurement |
| Lifetime decode iteration | `postJoinIterUs` | median over all positive SSE token gaps | Diagnostic for later batch changes, not a same-snapshot validation point |
| Completion | `projectedE2EUs` | EPP completion callback and client E2E | Client is authoritative for the user-visible result |
| Resident externality | per-resident baseline and candidate target times and values | later EPP first-token/completion callback for that resident | The selected-world residual includes later arrivals; it is not the missing counterfactual |

The iteration prediction is the five-term law used by the decision itself:

```text
T_iter = alpha
       + C0 * B_decode
       + C1 * sum(decode context tokens)
       + C_pf * sum(prefill grant tokens)
       + C_attn * sum(prefill attention work)
```

The logger records every term separately for the current decode iteration,
the arrival's first post-join decode iteration, and the remote prefill
iteration. The analyzer checks that each total equals the sum of its terms.
This identity check catches implementation or logging mistakes; agreement
with observed time is the separate coefficient-validity test.

For a remote candidate, the policy predicts:

```text
prefill completion = prefill admission
                   + number of chunks * prefill iteration
                   + prompt work

remote lead = prefill completion + transfer
decode join = max(remote lead, decode admission)
```

The maximum is essential. Remote prefill and decode-queue drainage advance at
the same time; adding them would count the decode wait twice.

## Event stream

EPP emits:

- `decision_candidate`: one complete candidate decomposition.
- `decision_resident_prediction`: one resident or pre-first-token occupant
  under one candidate, with baseline value, candidate value, and contribution.
- `decision_selected`: the chosen candidate with the same decomposition.
- `epp_first_token`: first positive completion-token count seen by EPP.
- `epp_complete`: final completion count and EPP-observed duration.

The sidecar emits:

- `proxy_request_received`.
- `prefill_dispatched` and `prefill_completed` for remote placement.
- `decode_dispatched`.
- `decode_first_event`.
- `decode_token_event` at debug verbosity for each complete SSE data event.
- `decode_completed`, or `proxy_completed` when no decode dispatch occurred.

The sidecar preserves the incoming ID in `requestID`. It assigns a separate
UUID to a disaggregated request, sets that UUID—not an additional duplicate
header—on both vLLM legs, and records it as `proxyRequestID`. vLLM echoes that
proxy UUID to the workload client on a remote request. The analyzer therefore
resolves client response IDs through the recorded proxy-to-EPP mapping. Strict
mode now fails if any successful client row is unmatched, ambiguous, or
duplicated; filtering the EPP stream to the intersection alone is not enough.

## Run the analyzer

Collect JSON-formatted EPP and `routing-proxy` container logs. EPP verbosity 5
and sidecar `--zap-log-level=4 --zap-encoder=json` expose candidates,
resident predictions, and SSE event gaps. The string value `debug` suppresses
the sidecar's `V(4)` token events and is insufficient for this study.

```bash
python3 analyze-causal-observability.py \
  results/epp_logs \
  results/sidecar_logs \
  --client-trace results/trace_data.csv \
  --report-out results/prediction-validation.json \
  --rows-out results/prediction-residuals.csv \
  --externality-rows-out results/resident-prediction-residuals.csv \
  --strict
```

The script accepts individual files or directories. It ignores unrelated
JSON log records. `--strict` fails when request joins are incomplete, selected
decisions are duplicated, or a logged algebraic identity fails.

An optional NIXL window file has this CSV schema:

```text
requestID,successfulTransfersBefore,successfulTransfersAfter,transferTimeSecondsBefore,transferTimeSecondsAfter
```

Pass it with `--transfer-windows`. A row is used only when the successful
transfer counter increases by exactly one. Larger deltas are reported as
ambiguous and excluded from request-level transfer residuals.

## First deployed prediction-versus-observation results

The r4 smoke used six fixed-512 Qwen3-14B requests at TP=1 on one prefiller and
two decoders. All six were local, all six strict joins passed, and all 32
candidate identities were exact within `1.82e-12` microseconds. Client TTFT
MAE was 13.38 ms. Its sidecar verbosity did not emit per-token events, so r4
does not validate the iteration prediction.

Run `qwen-causal-observe-r5-20260827` then sent 20 requests at one request/s,
each with 15,000 generated input tokens and exactly 512 output tokens. It used
Qwen3-14B, vLLM 0.23.0, TP=1, one prefiller, two decoders, disabled prefix
caching, 2,048 scheduled tokens per iteration, and the two public v2 images in
Vishakha's Quay namespace. All 11 pipeline tasks and all 20 requests
succeeded. The complete local evidence is under
`qwen-causal-observe-r5-20260827-results-full/`.

Strict coverage is complete: 20/20 workload decisions joined through client,
EPP, and sidecar events; 17 were local and three were disaggregated. The three
remote response IDs were correctly resolved through their proxy UUIDs. Each
remote request has exactly one receive, prefill dispatch/completion, decode
dispatch/first event/completion, a single shared proxy UUID, and 513 SSE event
records (512 token-bearing events plus the final metadata event). All 76
candidate rows satisfy every iteration, TTFT, completion, externality, and
score identity; the largest discrepancy is `3.64e-12` microseconds.

This run intentionally exercised only clean decision snapshots: all 76
candidate rows had zero resident-prefill grant and attention terms. It
therefore validates that the five-term implementation reduces correctly in
that regime, but it is not another live mixed-batch test. The independent
Qwen calibration remains the mixed-structure evidence: 6,048 decode samples
over 24 cells gave 0.405% MAPE after cell-median fitting, 27 prefill samples
gave 16.36% MAPE, and 72 held-out mixed rows gave 6.96% MAPE. The mixed rows
were never used to fit the coefficients. Prefill repetition and broader
decode-context/mixed-batch coverage are still required before calling the
coefficient vector high confidence.

The following numbers test predictions against proxy- or client-observed time;
they are separate from those exact identities.

| Component | Population | Result |
|---|---:|---:|
| Local dispatch to first event | 17 | MAE 40.20 ms; MAPE 4.54% |
| Local client TTFT | 17 | MAE 60.82 ms; MAPE 6.87% |
| Remote prefill completion | 3 | MAE 31.35 ms; MAPE 3.68%; prediction was high by 31.35 ms on average |
| Early decode iteration, local | 17 | MAE 0.844 ms; MAPE 5.72% |
| Early decode iteration, remote | 3 | MAE 1.373 ms; MAPE 8.62% |
| Remote post-prefill tail | 3 | MAE 1.058 s; MAPE 487.36%; the model is missing most of this tail |
| Client completion after output estimator reached 512 | 6 local / 3 remote | MAPE 16.40% / 15.21% |

One decoder handled exactly one remote request, making its NIXL counter delta
unambiguous. The model predicted 200.627 ms; vLLM measured 599.502 ms, a
398.875 ms residual and a 198.8% relative error. The other decoder handled two
transfers, so its aggregate was not assigned to either request. This one point
does not justify a new fit, but it shows that the standalone 12.214 GB/s
transfer calibration did not carry into this loaded 1P2D run.

The first 11 workload decisions used `nHatOut=1`, the deliberate one-token
cold start of the per-class output-length mean, even though every request
produced 512 tokens. Their client completion MAE was 11.027 s. After the first
workload completion updated the mean, the remaining nine decisions used
`nHatOut=512` and completion MAE fell to 1.429 s. Cold-start rows must be
excluded from coefficient-quality claims, and future matched trials need an
explicit same-class warmup before the measurement window.

There are 69 selected resident completion predictions, all for decoder
residents. Sixty-one have later logged arrivals before the resident completes;
their 3.698 s MAE is confounded by work that was unknown at the decision. The
eight tail rows with no later logged arrival have 151.96 ms MAE and -118.61 ms
bias for the selected-world completion target. That is useful evidence for
absolute target-time prediction in a quiet tail, but it is not an observation
of the absent-arrival baseline and therefore does not validate the causal
externality difference itself.

The defensible result is consequently narrower than “the algorithm works.”
The request-keyed port, five-term arithmetic, local first-token model, remote
prefill model, and early decode-iteration model now have real-stack evidence.
The remote transfer/post-prefill path is not calibrated under load, output
length needs warmup, and resident counterfactual externality still requires a
fixed-snapshot replay or paired run.

## Experiment sequence

The first run should be a low-rate observability smoke, not a policy claim:

1. Send one local and one forced or naturally selected remote request.
2. Require one `decision_selected`, one sidecar receive, one decode dispatch,
   one first event, and one completion per successful request.
3. For the remote request, additionally require prefill dispatch/completion
   and one shared proxy UUID on both vLLM legs.
4. Run the analyzer in strict mode and require every algebraic identity to be
   within floating-point tolerance.

Then validate the Qwen coefficients before comparing policies:

1. Run decode-only batches across many batch sizes and context lengths.
2. Run prefill-only batches across prompt lengths and chunk positions.
3. Run mixed batches spanning both decode residents and prefill grants.
4. Fit on a training subset and report residuals on held-out points.
5. Report error by regime, not only one aggregate score: idle, moderate load,
   near saturation, short prompt, long prompt, and mixed batch.

The prefill-only data identifies `alphaP`, `CPf`, and `CAttn`; decode-only data
identifies `alphaD`, `C0`, and `C1`; mixed batches test whether the two fitted
parts add as the model assumes. A fit using only mixed points can look good
while trading error between coefficients, so it is not enough.

After coefficient validation, run repeated causal and stock-policy trials at
the same offered loads, request seeds, model, topology, cache setting, token
limits, and sidecar image. Compare both composite goodput and component
residuals. The algorithm is supported only if its identities hold, its timing
predictions are calibrated out of sample, and its selected placements improve
the preregistered workload metric against the matched baseline.

## What remains unobservable

Without vLLM instrumentation, this study cannot directly recover:

- scheduler admission time;
- the exact requests in each engine batch;
- pure GPU iteration duration;
- a request-specific NIXL duration during overlapping transfers; or
- the realized baseline and candidate resident outcomes from one execution.

The smallest counterexample is two overlapping transfers: the aggregate NIXL
counter increases by two, so either request may have consumed either part of
the measured time. Assigning that delta to one request would manufacture
ground truth. The analyzer excludes it.

Likewise, after the chosen arrival runs, later arrivals may delay a resident.
The resident's eventual completion is useful as a selected-world residual but
does not reveal how that resident would have completed without the chosen
arrival. A causal externality claim therefore needs a fixed-snapshot replay or
paired counterfactual run in addition to these logs.
