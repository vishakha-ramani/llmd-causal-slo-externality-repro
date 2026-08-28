# Causal SLO Externality Placement

**Types:** `causal-slo-externality` (policy, profile handler, request lifecycle) and
`causal-slo-externality-picker` (the argmin)

Joint P/D placement by a single argmin over (decode endpoint, prefill placement) of

```
J(d, p) = V * ( externality(d, p) - ownGood(d, p) )
```

Each candidate is priced by the smooth SLO value it destroys among the residents already on the
endpoint, evaluated **causally**: every completion model is gated on an admission window, so a
resident that finishes before the arrival is admitted contributes exactly zero. That gating is what
makes the term an externality rather than another load proxy.

The consequence is that two endpoints with identical queue depth and identical KV occupancy are
**not** interchangeable. One may hold residents just inside their end-to-end deadline, where an
extra co-scheduled prefill flips them to a miss; the other residents with comfortable slack, where
the same prefill costs nothing recoverable. No load-shaped signal can see that difference, because
it is a property of the residents' deadlines rather than of the endpoint's load.

## Why two plugin types

The scheduling `Picker` and `ProfileHandler` interfaces both declare a method named `Pick` with
different signatures, so no single Go type can satisfy both. The policy is one shared state object
behind two registered types:

| type | interfaces | responsibility |
|---|---|---|
| `causal-slo-externality` | `ProfileHandler`, `PreRequest`, `ResponseBodyProcessor`, `ConsumerPlugin` | owns configuration, the shadow table and the per-class output-length means; runs the role-blind profile; splits the chosen pair; maintains request state |
| `causal-slo-externality-picker` | `Filter`, `Picker`, `ConsumerPlugin` | attaches the request to candidates, then computes the argmin |

**The policy must be declared before the picker.** The picker resolves it through the plugin handle
and factories run in configuration declaration order. Reversing them fails at startup.

The profile handler is **not** referenced from `schedulingProfiles`; the loader finds it by scanning
instantiated plugins for the interface. Only the picker is referenced.

## Why not a Scorer

Two independent structural reasons:

1. A `Scorer` returns one scalar per endpoint, so no (decode, prefill) **pair** can be named.
2. Every scorer contribution passes through a `[0,1]` clamp, while `J` is signed and unbounded.
   Clamping would collapse every candidate with `J <= 0` to a single score, destroying the ranking
   rather than degrading it. Measured on realistic inputs, `J` ranges from about `-1.7` for an
   endpoint with ample slack to about `+2.9` for one whose residents sit at their deadline.

## Required shape

A **single role-blind profile**. Every profile receives the same candidate slice, so a profile
carrying no role filter sees both pools and the picker can enumerate the full cross product: `D`
local candidates plus `D*P` disaggregated ones, scored on one scale in one argmin.

Adding `decode-filter` or `prefill-filter` to that profile splits the pools and silently reduces
this to a decode-first decomposition, which is a **different algorithm** with a measured cost of
-0.0485 equal-cell mean goodput and -0.1555 worst cell. It would still run and still report goodput.
The `decomposed` parameter exists to reproduce that arm deliberately.

## Configuration

```yaml
plugins:
# Declared first: the policy reads the tokenized prompt and the prefix attribute these produce.
# Omit `name:` on the token producer -- see "The two tests that must never be relaxed".
- type: token-producer
  parameters:
    modelName: "meta-llama/Llama-3.3-70B-Instruct"
    vllm:
      url: "http://localhost:8000"
- type: approx-prefix-cache-producer
  parameters:
    autoTune: false            # the prefix block size is clamped up to 64; it is NOT the engine's 16
    blockSizeTokens: 64
- type: causal-slo-externality
  parameters:
    v: 8.0
    # Every workload's triple, keyed by workload name, with exactly one active.
    sloTargets:
      interactive:   {tauTTFTUs: 1000000,  tauITLUs: 50000,  tauE2EUs: 16000000}
      reasoning:     {tauTTFTUs: 2000000,  tauITLUs: 100000, tauE2EUs: 802000000}
      deep_research: {tauTTFTUs: 10000000, tauITLUs: 100000, tauE2EUs: 40000000}
    # The one line to change per workload run. An unresolvable value fails at startup rather than
    # resolving to a zero triple, which would flatten the policy rather than loosen it.
    activeSLOTargets: interactive
    chunkTokens: 2048          # must equal the engine's --max-num-batched-tokens
    blockSize: 16              # must equal the engine's --block-size
    maxBatchSize: 256          # must equal the engine's --max-num-seqs
    outputTokenProcessingUs: 45
    cXferSizeAware: true
    xferBaseUs: 50.0           # MEASURE on the target fabric
    xferBandwidthGBps: 25.0    # MEASURE on the target fabric
    kvBytesPerTokenPerGPU: 81920.0
    gpuTypeLabelKey: pd-infocomm.io/gpu-type
    coefficients:
      NVIDIA-H100-80GB-HBM3:
        alphaDUs: 16613.537554540144
        alphaPUs: 16617.85321583337
        c0UsPerReq: 5.347316038602452
        c1UsPerToken: 0.04761401141756073
        cPfUsPerToken: 6.144687138665833
        cAttnUsPerUnit: 0.00010075247918809842
    roleLabelKey: llm-d.ai/role
    prefixMatchInfoProducerName: approx-prefix-cache-producer
    profiles: {joint: joint, decode: decode, prefill: prefill}
    noCapacity: true
- type: causal-slo-externality-picker
  parameters:
    policyPluginName: causal-slo-externality
schedulingProfiles:
- name: joint
  plugins:
  - pluginRef: causal-slo-externality-picker
```

### Parameters that fail loudly rather than degrade

Validation rejects each of these at startup, because each produces a plausible-looking score rather
than an error:

| setting | why |
|---|---|
| all-zero SLO triple | Every factor of the value kernel is guarded on its threshold being positive, so the value becomes identically 1, every charge `1-1 = 0`, and the argmin degenerates to enumeration order. This does not loosen the policy, it flattens it. |
| `maxBatchSize: 0` | Makes the admission estimator's zero-wait fast path unreachable and its free-slot count start negative, so every endpoint takes the wave form on every decision. |
| `tauITLUs <= alphaDUs` | An ITL target at or below the per-iteration intercept is physically unachievable on that hardware, indicating either a wrong target or coefficients fitted on other hardware. |
| `abs(alphaDUs-alphaPUs)/alphaDUs > 0.10` | Prefill and decode share the intercept on one device; more divergence means the halves were fitted on mismatched hardware. |

An endpoint whose GPU-type label is absent or unrecognised is **excluded** from the candidate set
rather than scored under defaulted coefficients, and the exclusion is counted. Heterogeneity rides
the per-iteration intercept, present on every iteration regardless of KV state, so defaulting would
be wrong on every decision rather than only under load.

### Runtime preconditions

- **One EPP replica.** Replicas partition the shadow table, so a second replica sees a strict subset
  of residents and under-counts the externality.
- **Per-chunk usage** (`stream_options.continuous_usage_stats`, or `--enable-force-include-usage` on
  the server). Without it the completion count arrives only in the final chunk, the progress
  estimate stays zero for a request's whole lifetime, and every remaining-steps estimate is wrong.
- **A GPU-type pod label.** Plugins see pod labels, never node labels, so the accelerator
  `nodeSelector` is not visible here.
- **One `InferencePool` for all decode endpoints**, or no joint argmin exists.

## Metrics

| metric | why it matters |
|---|---|
| `..._ttft_estimator_substitution_total` | The published TTFT model is unavailable here and the closed form is substituted. Upstream substitutes silently; this counter is a requirement, not instrumentation. |
| `..._placement_total` | Local against disaggregated placements. |
| `..._candidate_rejected_total` | Candidates removed before scoring, by reason. Without it a GPU-label mistake looks like a routing preference. |
| `..._block_size_mismatch_total` | A scraped KV block size disagreeing with the configured one. |
| `..._shadow_table_requests` | In-flight requests backing the resident estimates. Near zero under load means every externality is zero and the policy has degraded to its own tie-break. |
| `..._shadow_table_reaped_total` | Entries dropped by the staleness sweep. |
| `..._prefix_unobserved_total` | Candidates scored with no prefix residency. Nothing is rejected: the uncached length falls back to the full prompt, which on a well-cached endpoint is the difference between no prefill work and the entire prompt, so it re-prices the candidate and moves the local/remote boundary while leaving it in the argmin. |
| `..._tokenization_unavailable_total` | Requests refused because tokenization produced nothing. On a cold EPP or a dead render sidecar this is every request. |
| `..._argmin_duration_seconds` | Duration of the argmin itself, including endpoint-state reconstruction. This policy adds EPP-side work inside the measured TTFT, and this makes the contribution quantifiable. |

The tokenizer's own round trip is **not** re-instrumented here: the framework already records it as
`llm_d_epp_plugin_duration_seconds{extension_point="DataProducer",plugin_type="token-producer"}`.
Note its buckets top out at 0.1 s, so a slow render lands in `+Inf`; use `_sum / _count` for the
mean and the `+Inf` fraction for the exceedance rate.

## Declared degradations

Each changes what a measured number means, and the direction of each bias is stated because they do
not all point the same way.

| id | degradation | direction |
|---|---|---|
| substituted TTFT estimator | The published estimator replays the engine scheduler over its ordered wait queue; those per-request inputs are unobservable here. The closed-form rollforward estimator is used and every substitution counted. | Its wave-form branch understates deep-queue delay, so residents are charged for more interference and local placement is over-priced: biased toward **remote**. |
| shadow-table resident state | Per-resident remaining steps, arrival and first token come from an EPP-side table rather than the engine. | One replica only; traffic bypassing the EPP is invisible. Both under-count the externality: biased toward **local**. |
| late first token | The recorded first token is a dequeue time, because non-final chunks drain on a per-request goroutine. It fixes a common positive multiplier on every resident's charge. | Shrinks every charge, so the externality shrinks: biased toward **local**. |
| KV occupancy | Derived from the usage fraction, and includes blocks held by requests still prefilling, which the fitted coefficient definition excludes. | Over-counts KV while a prefill is in flight. Inherited from the simulation rather than introduced. |
| free KV blocks | Derived from the same fraction, quantised by the exporter and truncated on conversion. | A floor, never an over-count, so admission delay is over-stated: acts **against** the substituted estimator's bias. |
| prefilling-token estimate | The router cannot know which requests the engine scheduled this step. | An over-estimate, inflating the decode iteration time: biased **away from local prefill**. |
| unobserved prefill pool | Where a pool's occupants are invisible its charge is zero. | Under-prices remote prefill: biased toward **disaggregation**. |
| transfer model | `xferBaseUs` and `xferBandwidthGBps` are the simulator's executor settings, not target measurements, and `kvBytesPerTokenPerGPU` is derived from the model config rather than measured. | Mis-prices every disaggregated candidate in one direction. **Measure before trusting any magnitude.** |
| approximate prefix residency | The comparison baseline's prefix-threshold decider reads the bare prefix data key, whose identity includes its producer name, so it can only see the approximate producer. A precise-only configuration would never disaggregate: loud in the logs, silent in the results. | Both arms bound to one producer, so the error is common-mode. |
| one SLO triple per deployment | The thresholds are per **workload**, but run assembly writes one scenario per package and emits pipeline runs per (workload, package) with no per-workload override. So exactly one triple is live per assemble, and `activeSLOTargets` selects it. The committed value is `interactive` (1,000 / 50 / 16,000 ms). | **The reasoning and deep-research runs are mispriced unless the selector is changed and re-assembled.** Their E2E targets are 802,000 ms and 40,000 ms against interactive's 16,000 ms, and a resident's charge scales as roughly `1/tau_e2e`, so the difference exceeds 10x -- larger than several other entries in this table. Producing those runs correctly means one assemble each. |
| in-band tokenization | The configured token producer uses the vLLM render backend, so the input length is a real tokenization. This **adds** work the simulation never had: an uncached round trip on the request path, inside the measured TTFT. | Common-mode across arms, so the **ordering claim is unaffected**, but it inflates absolute TTFT in both arms against a 1,000 ms interactive target. Magnitude unquantified until the argmin histogram and producer timings are read from a run. |
| (avoided) byte-estimate token count | Omitting the token producer would auto-create a byte-packing estimate at four bytes per pseudo-token, whose IDs are not engine tokens. Under it, `residentKVBlocks` sums a pseudo-token prompt length with a real-token completion count and `reqKVNeed` divides pseudo-tokens by a real-token block size. | Those are **correctness** faults, not accuracy ones. The configured backend removes both. This entry records what the configuration avoids, and it is **linked** to the approximate-prefix row: real token IDs make the approximate index hash the same token stream the engine caches. |

**The net direction is not known and is not claimed.** These biases do not point the same way. Read
the register as the list of things to measure on the target, not as a set of errors that cancel.

## Token producer coverage, and what it does not cover

The configured token producer supplies the input length every term of the objective depends on, so
`tokenproducer_test.go` drives the **real** producer through its public factory against a local
`httptest` render server. Covered: construction from the overlays' exact parameter shape, the
negotiated produce timeout (asserted at a configured value, not the default, so it proves the
plumbing), the produced data key, a successful render yielding real token IDs, and a render outage
leading to this policy refusing to score.

`TestTokenProducer_PublishesTheBareDataKey` is one of the two tests that must never be relaxed; see
below for why.

**Not covered:** the render backend's warmup retry loop and its multi-minute ceiling, real vLLM
response shapes beyond `token_ids`, and behaviour against the actual sidecar. The backend's own test
seams are unexported and in-package, so those paths are reachable only from the tokenizer package or
on a cluster.

## The test that must never be relaxed, and one that does less than it appears to

The first guards the mechanism and is load-bearing: fix it by repairing what broke, never by relaxing
the assertion. The second is included here because an earlier revision of this section claimed it
guarded the overlay, and it does not — the correction matters more than the claim did, because a reader
who trusts it would stop looking.

**`TestExternality_DiscriminatesBetweenEndpointsWithIdenticalLoad` guards the mechanism.** It asserts
that batch size, queue depth, KV occupancy, free blocks, resident count and uncached prompt length are
all **equal** between two endpoints, then requires the endpoint whose residents sit at their deadline
to price more than 10x the one whose residents are already lost. That is the entire claim of this
policy: the signal is a property of resident deadlines, not of load. If it fails, the port has become
a load proxy no matter what else still passes.

**`TestTokenProducer_PublishesTheBareDataKey` documents the hazard but DOES NOT GUARD THE OVERLAY.**
Read this before relying on it. It asserts that a producer constructed with the default name publishes
exactly `TokenizedPrompt/token-producer`. That is a property of the framework's key construction
(`dataproducer/tokenizer/tokenizer.go:202`, returned by `Produces()` at `:230-232`), not a property of
the shipped overlays: its helper calls `tokenproducer.PluginFactory(tokenproducer.PluginType, ...)`,
so the name is hardcoded to the safe value and the test never reads either overlay. **Adding a `name:`
to the YAML does not fail this test.** Verified by probe: `name="my-tokenizer"` yields
`TokenizedPrompt/my-tokenizer`, the key no consumer asks for, and the test still passes.

The hazard itself is real and unchanged. A data key's identity is `DataType/ProducerName`, so a named
producer publishes a key nobody requests; `CreateMissingDataProducers`
(`pkg/epp/datalayer/data_graph.go:65-125`) then auto-creates a byte-estimate producer to supply the key
the prefix producer actually reads, and the configured render backend is built, warms up, performs its
per-request round trip, and is IGNORED. The deployment pays the full latency cost of a real tokenizer
and gets `ceil(bytes/4)` semantics. Deterministic, not a race.

**What actually protects that decision today is only that both overlays omit `name:`.** The test that
would pin it must read the artifact — parse each overlay's embedded `custom-plugins.yaml` and assert
the `token-producer` entry carries no `name:` key. That test is NOT YET PRESENT in this package; it is
a known gap, recorded here rather than implied away. Until it exists, treat the omitted `name:` as a
manually maintained invariant, and use the preflight check in the baseline overlay: confirm the EPP log
does NOT contain "auto-created default producer" naming `token-producer`.

A related caution in the same file: `TestTokenProducer_ConstructsFromTheOverlayParameters` asserts
`plg.TypedName().Name == tokenproducer.PluginType`, which restates the argument its own helper passed
to the factory. Its comment points at name-defaulting, but that defaulting happens in the config loader
(`pkg/epp/config/loader/defaults.go:108-113`), which this test never exercises.

Two companion facts, both measured rather than argued:

- `J` spans about `-1.7` to `+2.9` on realistic inputs. That is why the objective cannot be a
  `Scorer`: the `[0,1]` clamp would map both ends to the same score.
- `TestExternality_PeaksAtTheDeadlineAndFallsOffBothSides` records the shape the numbers support, and
  its comment carries the measured values. The peak is **broad** -- pricing stays near maximum until
  roughly six seconds past the deadline. Do not round that back to a sharp peak.

## Terms deliberately absent

Each is a term a reimplementation is likely to add back:

- **No standalone transfer penalty.** Transfer cost is priced causally through timing only, entering
  the remote lead and hence both the arrival's projected TTFT and the resident admission window.
  Adding an additive penalty double-prices every transfer.
- **No TTFT/ITL deficit term** and **no per-decision normalization.** Both belong to a different
  rule. This policy compares raw smooth SLO value across candidates, which is exactly why the score
  is directly comparable between candidates.
- **No mean-ITL routing factor.** The routing value is a smooth TTFT x E2E surrogate; mean ITL is an
  evaluation gate. Adding it changes the arm.
- **No capacity account.** Dead under `noCapacity`, which is forced true. Setting `noCapacity: false`
  is REJECTED AT CONSTRUCTION (`config.go:261-262`), so the EPP fails at config load rather than
  scoring without a term it never computed; `score.go:128-131` leaves `capacityTotal` at zero. An
  earlier revision returned NaN from the score instead, which was strictly worse and was removed: in
  Go every comparison involving NaN is false, so `d.score < best.score-eps` would never fire and the
  first enumerated candidate would win silently — enumeration order masquerading as a decision. The
  uncalled capacity algebra is retained and its closed forms are checked by
  `TestUncalledLawTerms_AlgebraIsStillCorrect`, so a future `noCapacity: false` port inherits tested
  algebra rather than merely preserved algebra.
- **No empty-prompt or fully-cached guard.** With no uncached tokens the chunk terms return zero and
  a fully-cached request still scores on every candidate, so the argmin still chooses which decode
  endpoint to use. A guard would throw that choice away.
