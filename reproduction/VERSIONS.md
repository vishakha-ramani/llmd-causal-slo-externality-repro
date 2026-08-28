# Exact provenance and versions

This file separates immutable inputs from facts that were not pinned. The R5
results should not be presented without the caveats in the last section.

## Router source and images

| Component | Exact version used |
|---|---|
| Upstream repository | `https://github.com/llm-d/llm-d-router` |
| Upstream tag | `v0.9.0` |
| Upstream/base commit | `5f4e762f341a5196393ce79f8a57c3e1900c4a6b` |
| EPP image | `quay.io/vishakharamani/llm-d-router:causal-qwen-observability-v2-20260827` |
| EPP image digest | `sha256:00634bf69e44bdc79b6e0253b864003eb4f752cd1304d5afd24578fda0f3efc1` |
| Sidecar image | `quay.io/vishakharamani/llm-d-router-disagg-sidecar:causal-qwen-observability-v2-20260827` |
| Sidecar image digest | `sha256:bee5b968a5892782c7cb3cb03a9b26fb109391896c6113d5d0010b1ab234e134` |

Both images were built from the pinned base plus the causal policy, the P/D
sidecar `min_tokens` fix, and the request-keyed observability changes in this
repository. The base commit alone does not contain those changes. The commit
checked out after cloning this repository is the exact source snapshot.

The translated policy lineage recorded by the porting notes is:

- source BLIS/inference-sim implementation commit
  `69c3ccb2a9f0e1f2f5c4ab80cef5ec8cc176eae9`;
- generated translation identity
  `07f0062216a475f9d9931fd8f752b11dc25ea52abdd38dc76dea7382482297ba`.

## Workload and serving stack

| Component | Exact version/configuration used |
|---|---|
| llm-d-benchmark source | commit `76473d05a0f546a6bc0c4662ed7e34104da73891` |
| inference-sim/BLIS source | commit `cc8846cc41e71488cc32004778dde83dc747a2f7` |
| vLLM | `0.23.0` |
| vLLM image digest | `sha256:3a1e7f5904e1a1192a02aa0086ceaffc33985d7044c7bb25b3a43d61bdbe3ac0` |
| Model | `Qwen/Qwen3-14B`, BF16, tensor parallelism 1 |
| Topology | one prefiller, two decoders; one GPU per replica |
| Scheduler limits | `max-num-seqs=256`, `max-num-batched-tokens=2048`, block size 16 |
| Prefix cache | disabled |
| Workload | 20 requests, constant 1 request/s, 15,000 generated input tokens, fixed 512 output tokens |
| NIXL | `1.2.0`, git `e128059a` |
| UCX | `1.21.0` |

The generator pins the two source commits and the published image tags. The
exact rendered R5 object is
[`manifests/qwen-causal-observe-r5-20260827.yaml`](manifests/qwen-causal-observe-r5-20260827.yaml).

## Hardware and cluster

| Component | Version used |
|---|---|
| GPUs | three NVIDIA H100 80 GB HBM3 |
| NVIDIA driver | `580.105.08` |
| CUDA | `13.0` |
| OpenShift | `4.19.19` |
| Kubernetes server | `v1.32.9` |

The full experiment also depended on the cluster's `sim2real` Tekton pipeline,
RDMA device/network configuration, Multus attachment, shared data/source PVCs,
and a Hugging Face secret. These cluster objects are prerequisites rather than
contents of this repository.

## Qwen coefficient vector used by R5

For `NVIDIA-H100-80GB-HBM3`:

| Coefficient | Value (microseconds) |
|---|---:|
| `alphaDUs` | 10884.573061102548 |
| `alphaPUs` | 10884.573061102548 |
| `c0UsPerReq` | 10.419827361034679 |
| `c1UsPerToken` | 0.05267373344102902 |
| `cPfUsPerToken` | 39.806418327287684 |
| `cAttnUsPerUnit` | 0.0014236690050459293 |

Transfer parameters were `xferBaseUs=595.162`,
`xferBandwidthGBps=12.213979`, and
`kvBytesPerTokenPerGPU=163840.0`.

## Reproducibility caveats

- The Hugging Face model revision was **not pinned**. vLLM recorded
  `revision=None`; `Qwen/Qwen3-14B` may therefore resolve to a newer model
  revision in a future rerun. The packaged R5 replay remains exact, but a new
  cluster run is not bit-for-bit model reproducible until a revision is added.
- Image tags are mutable identifiers. Use the digests above when exact runtime
  identity matters; the portable generator uses tags because the scenario
  schema separates repository and tag.
- R5 has 76 clean-snapshot candidate rows and zero live mixed-state rows. The
  additive mixed iteration law is supported by a separate 72-row held-out
  calibration set, not by this workload.
- `nHatOut` starts at one. The first 11 R5 requests used `nHatOut=1`; the final
  nine used 512. A claim-bearing rerun must warm up the same class before the
  measurement window.
