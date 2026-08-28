# Qwen real-system iteration-law audit

Date: 2026-08-27

## Status update: the full law is now deployed

The implementation gap described below was corrected in the public EPP image
`quay.io/vishakharamani/llm-d-router:causal-qwen-iterlaw-20260827`, runtime
digest `sha256:7d7a9ba382d36bc6673fabf426efdf5bb2d0d257d5953bed7670c7bbec1716fd`.
The request-keyed candidate trace now exposes all five terms. In the matched
40-request Qwen run, all 136 workload candidate rows reconstructed the
iteration sum with at most `1.82e-12` microseconds of floating-point error.
See `QWEN-SLO-EXTERNALITY-COMPARISON-2026-08-27.md` for the real-system result.

## What one engine iteration must predict

In one vLLM iteration, the scheduler chooses a mixed batch. Every decode
request advances by one token and rereads its current KV context. Every prefill
request advances by a prompt chunk and attends from those new queries to the
context already visible to that request. The model predicts the duration of
that complete forward pass; it does not assign independent wall-clock service
times and then serialize them.

For decode request `r`, let `H_r` be the context it rereads. For prefill request
`r`, let `kappa_r` be its scheduled chunk and `P_r` the context visible before
that chunk. The INFOCOM slide and paper give the request-wise law

```text
T_iter(B) = alpha
          + sum over decode r of [c_dec + c_kv * H_r]
          + sum over prefill r of [c_pf * kappa_r
                                  + c_attn * kappa_r * (P_r + kappa_r/2)].
```

Equivalently, collect four sufficient batch totals:

```text
B_dec = number of decoding requests
KV    = sum of H_r over decoding requests
S_pf  = sum of kappa_r over prefilling requests
U_pf  = sum of kappa_r * (P_r + kappa_r/2) over prefilling requests

T_iter = alpha + c_dec*B_dec + c_kv*KV + c_pf*S_pf + c_attn*U_pf.
```

The baseline `alpha` is charged once per engine iteration. Every other term is
the marginal contribution of a request scheduled in that iteration. On a
dedicated prefiller, `B_dec=KV=0`. On a mixed decoder, both sums can be nonzero.

This is the law shown in
`infocom/presentation/from-mixed-batch-to-iteration-latency.png` and stated as
Equations `marginal` and `titer` in `infocom/main.tex`. The smooth
`kappa_r/2` convention differs from the exact causal-pair count by a term
linear in `kappa_r`; the fitted `c_pf` absorbs that convention.

## What the Qwen calibration fit

The Qwen3-14B TP=1 calibration in `pd-model-validation/calibration/` uses this
same law:

- Decode-only cells fit `alpha + c_dec*B + c_kv*B*(n+k)` while independently
  varying batch size and context.
- Prefill-only TTFT fits the sum of all prompt chunks. With no cached prefix,
  the attention terms telescope to `c_attn*n^2/2`, so
  `TTFT = n_chunks*alpha + c_pf*n + c_attn*n^2/2`.
- The held-out mixed check evaluates
  `alpha + c_dec*B + c_kv*resident_context_sum + c_pf*kappa
  + c_attn*kappa*(P+kappa/2)` for each observed prefill-bearing iteration.

The full calibration produced 6,048 decode rows, 27 repeated prefill samples,
and 72 held-out mixed rows. All coefficients are positive:

```text
alpha  = 0.010884573061102548 s/iteration
c_dec  = 1.0419827361034679e-05 s/decode request
c_kv   = 5.267373344102902e-08 s/resident KV token
c_pf   = 3.9806418327287684e-05 s/prefill token
c_attn = 1.4236690050459293e-09 s/attention-basis unit
```

Decode cell-median R-squared is 0.9987 with 0.405% MAPE. Prefill R-squared is
0.99986. The mixed rows were not used to fit the coefficients and have 6.96%
MAPE. This supports the additive law for batches containing multiple decodes
and one prefill chunk. It does not yet validate batches with multiple
simultaneous prefill requests; a teacher-forced per-iteration Qwen trace should
cover that case.

## Historical pre-fix gap: the earlier EPP did not receive these statistics

The port audit in `llm-d-porting-explanation.md` records the current generated
plugin clocks as

```text
T_D = alpha_D + C0*B + C1*K + C_pf*S_pf
T_P = alpha_P + C_pf*S_pf.
```

These are not the full INFOCOM iteration law:

1. Both clocks omit the `C_attn*U_pf` contribution of resident prefill work.
2. `B` comes from `vllm:num_requests_running`, which includes requests that are
   still prefilling rather than only decode requests.
3. `K` is aggregate KV occupancy and can include prefill KV, whereas the fitted
   `c_kv` regressor is only context reread by decode requests.
4. `S_pf` is reconstructed from the EPP shadow table and is not the prompt work
   actually granted by the vLLM scheduler in the current iteration.

The arriving request's own `W_p` does include its attention term. That does not
repair the current-iteration clock for resident requests; it makes the model
asymmetric instead.

Therefore the newly fitted Qwen coefficients could not simply replace the
Llama values in that earlier image. The corrected image above resolves this
specific regressor mismatch; the coefficient-coverage limitations remain.

## Gate used before the causal-versus-stock experiment

For a comparison that represents the INFOCOM algorithm, the EPP needs the four
phase-specific aggregates `B_dec`, `KV`, `S_pf`, and `U_pf`, or equivalent
per-request state from which they can be summed. Its iteration clock must then
evaluate the full marginal law above. A request-keyed trace must show those
inputs, every term, and the final sum for every candidate.

If exact scheduler grants cannot be exported, the fallback is to fit a new
proxy directly on the aggregates that EPP can observe. That may be useful as a
systems heuristic, but it is no longer a direct real-system test of the
INFOCOM iteration law and must be labeled separately.

The clock was corrected before measuring policy goodput. The remaining checks
for an INFOCOM-strength claim are:

1. Replay fixed endpoint snapshots through the EPP and simulator and require
   term-by-term agreement.
2. Capture teacher-forced Qwen engine iterations with arbitrary mixed batches
   and validate frozen predictions from the actual scheduled `B_dec`, `KV`,
   `S_pf`, and `U_pf` values.

The transfer calibration is independent of this formula issue and can proceed.
The matched causal-versus-stock load experiment is gated on fixing or explicitly
reclassifying the EPP clock.
