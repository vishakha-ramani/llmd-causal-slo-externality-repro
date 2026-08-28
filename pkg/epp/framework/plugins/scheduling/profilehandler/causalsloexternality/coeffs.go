package causalsloexternality

import "math"

const (
	// minMu floors the nominal drain rate so predictor denominators cannot collapse to
	// zero at a degenerate operating point where alpha >= T_iter.
	minMu = 1e-3

	// maxAlphaDivergence bounds |AlphaD-AlphaP|/AlphaD. Prefill and decode share the
	// per-iteration intercept on one device.
	maxAlphaDivergence = 0.10
)

// clampMu bounds a drain rate to [minMu, 1].
func clampMu(mu float64) float64 {
	if mu < minMu {
		return minMu
	}
	if mu > 1.0 {
		return 1.0
	}
	return mu
}

// Wp is the prefill demand of ap uncached tokens within a prompt of full length ar, in
// microseconds. It is the trajectory sum of the causal per-step charge
// CPf*s + CAttn*s*(prefix + s/2) integrated over the prefill from prefix ar-ap to ar,
// which is why the second factor is (ar - ap/2). At ap == ar it reduces to
// CPf*ar + 0.5*CAttn*ar^2.
func (c Coeffs) Wp(ap, ar int) float64 {
	a := float64(ap)
	r := float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// Wd is the decode demand for a prompt of length ar generating o output tokens: the
// exact discrete sum over k in [0,o) of C0 + C1*(ar+k), closed to
// C0*o + C1*o*(ar + (o-1)/2).
func (c Coeffs) Wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}

// iterationTerms is the five-term INFOCOM iteration law. Each workload-dependent
// field is the sum of the request marginals scheduled in this iteration; alpha is
// charged once for the complete batch.
type iterationTerms struct {
	alpha            float64
	decodeRequests   float64
	decodeKV         float64
	prefillLinear    float64
	prefillAttention float64
	total            float64
}

func (c Coeffs) iterationTerms(alpha float64, bDec int, kv, sPf int64, uPf float64) iterationTerms {
	t := iterationTerms{
		alpha:            alpha,
		decodeRequests:   c.C0 * float64(bDec),
		decodeKV:         c.C1 * float64(kv),
		prefillLinear:    c.CPf * float64(sPf),
		prefillAttention: c.CAttn * uPf,
	}
	t.total = t.alpha + t.decodeRequests + t.decodeKV + t.prefillLinear + t.prefillAttention
	return t
}

// tIterDecode evaluates one potentially mixed decoder iteration. bDec and kv
// contain decode requests only. sPf and uPf contain the scheduled prefill grants.
func (c Coeffs) tIterDecode(bDec int, kv, sPf int64, uPf float64) float64 {
	return c.iterationTerms(c.AlphaD, bDec, kv, sPf, uPf).total
}

// tIterPrefill is the prefill iteration time. A dedicated prefill server runs no
// decode work, so B_dec and KV are both zero.
func (c Coeffs) tIterPrefill(sPf int64, uPf float64) float64 {
	return c.iterationTerms(c.AlphaP, 0, 0, sPf, uPf).total
}

// muDecode is the live decode drain rate mu = 1 - alpha/T_iter.
func (c Coeffs) muDecode(bDec int, kv, sPf int64, uPf float64) float64 {
	return clampMu(1.0 - c.AlphaD/c.tIterDecode(bDec, kv, sPf, uPf))
}

// muPrefill is the live prefill drain rate.
func (c Coeffs) muPrefill(sPf int64, uPf float64) float64 {
	return clampMu(1.0 - c.AlphaP/c.tIterPrefill(sPf, uPf))
}

// THE THREE FUNCTIONS BELOW ARE UNCALLED BY CONSTRUCTION, NOT BY OMISSION.
//
// They are stated because the specification's completeness rule requires the latency law to be
// stated in full, and because deleting them would discard algebra a future port needs. Each names
// the term it belongs to and why that term is unreachable here. They are covered by tests, so the
// stated algebra is verified rather than merely preserved.
//
// If any of these ever becomes reachable, it must be through a call site, not by relaxing a guard.

// muDNom is the nominal decode drain rate at the SLO-critical batch, where the iteration time
// equals tau_itl.
//
// It belongs to the CAPACITY ACCOUNT: it scales the virtual workload queues that the capacity term
// reads. Configuration rejects enabling that term, so this is unreachable here. Getting the drain
// floor wrong by a factor of a thousand would be latent for exactly that reason, and would divide by
// a floor far too small at precisely the degenerate points the floor exists to protect.
//
//nolint:unused // capacity-account algebra, unreachable while noCapacity is forced true
func (c Coeffs) muDNom(tauITLUs float64) float64 {
	return clampMu(1.0 - c.AlphaD/tauITLUs)
}

// muPNom is the nominal prefill drain rate at the nominal operating chunk. Same account, same
// reason.
//
//nolint:unused // capacity-account algebra, unreachable while noCapacity is forced true
func (c Coeffs) muPNom(sPfNom int) float64 {
	return clampMu(1.0 - c.AlphaP/(c.AlphaP+c.CPf*float64(sPfNom)))
}

// deltaBarDecode is the marginal decode work per step at a context length.
//
// It belongs to the HISTORICAL DRIFT OBJECTIVE, feeding its normalized ITL virtual queue. This arm
// carries no part of that objective: the candidate score reads no virtual queue and no historical
// deficit term. Stated for completeness of the law.
//
//nolint:unused // drift-objective algebra; this arm carries no part of that objective
func (c Coeffs) deltaBarDecode(ctxLen float64) float64 {
	return c.C0 + c.C1*ctxLen
}

// ceilBlocks is ceil(tokens/blockSize) with a zero floor and a blockSize guard.
func ceilBlocks(tokens, blockSize int64) int64 {
	if tokens <= 0 {
		return 0
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	return (tokens + blockSize - 1) / blockSize
}

// atLeastZero floors a token count at zero. The uncached prompt length is deliberately not
// floored where it is computed, because a non-positive value means "fully cached" and the chunk
// terms must see that sign; it is floored only at the use sites that need a magnitude.
func atLeastZero(a int) int {
	if a > 0 {
		return a
	}
	return 0
}

// kvTokensInUse derives resident KV tokens from the scraped usage fraction.
//
// KVCacheUsagePercent is a FRACTION in [0,1] despite the name, and the target's own
// consumers treat it that way. Dividing by 100 would under-count KV by 100x and
// collapse the C1*KV term that carries hardware heterogeneity.
//
// The occupancy includes blocks held by requests still prefilling, which the fitted
// coefficient definition excludes, so KV is over-counted while a prefill is in flight
// and that context is charged twice. The simulation's own consumed value has the same
// property, so this is inherited rather than introduced.
func kvTokensInUse(usageFraction float64, cacheBlockSize, cacheNumBlocks int) int64 {
	if usageFraction < 0 {
		usageFraction = 0
	}
	return int64(usageFraction * float64(cacheBlockSize) * float64(cacheNumBlocks))
}

// freeKVBlocks derives the free block count from the same fraction. Two independent
// losses make the result a FLOOR rather than an unbiased estimate: the scraped fraction
// is quantised by the exporter, and the conversion truncates. It gates the admission
// estimator's zero-wait fast path, so a systematic floor makes the estimator decline
// that path at a marginally-free endpoint, over-stating admission delay.
func freeKVBlocks(usageFraction float64, cacheNumBlocks int) int64 {
	free := (1.0 - usageFraction) * float64(cacheNumBlocks)
	if free < 0 {
		free = 0
	}
	return int64(free)
}

// chunkTerms returns the chunk count and the per-chunk prefill charge for ap uncached
// tokens. ap <= 0 returns (0,0) exactly: no prefill work and no per-chunk inflation,
// which is what lets a fully-cached request still score on every candidate.
func (cfg *Config) chunkTerms(theta Coeffs, ap int) (nChunks, deltaPfChunk float64) {
	if ap <= 0 {
		return 0, 0
	}
	chunk := cfg.chunkFor(ap)
	return math.Ceil(float64(ap) / float64(chunk)), theta.CPf * float64(chunk)
}

// chunkFor is the overlap chunk width used by the completion model: min(ap,
// ChunkTokens), not floored at 1 here because the sum helpers apply that floor.
func (cfg *Config) chunkFor(ap int) int {
	chunk := ap
	if cfg.ChunkTokens > 0 && cfg.ChunkTokens < chunk {
		chunk = cfg.ChunkTokens
	}
	return chunk
}

// reqKVNeed is ceil(a_r / BlockSize) in engine KV blocks, using integer ceiling
// arithmetic.
//
// This is the unit hazard the port introduces. Upstream has ONE KV block size, so its
// admission test compares accumulated resident blocks against a need in the same unit.
// Here BlockSize is an operator constant while resident block counts are derived from
// the scraped Metrics.CacheBlockSize, and if the two disagree the comparison is wrong
// by their ratio. The plugin validates them equal and fails loudly rather than
// converting between them: a port with two quantities that merely reconcile is a port
// with a latent unit bug.
//
// This is also a DIFFERENT block size from the prefix producer's, which is used only to
// convert cached blocks to cached tokens and is independently configurable.
func (cfg *Config) reqKVNeed(promptTokens int) int64 {
	if cfg.BlockSize <= 0 {
		return 0
	}
	return int64((promptTokens + cfg.BlockSize - 1) / cfg.BlockSize)
}

// cXferUs is the assumed KV-transfer cost for one request: base + bytes/bandwidth,
// where bytes is the block-aligned prompt size times the per-token per-GPU KV
// footprint. The bandwidth conversion is GB/s to bytes/us via *1000.
//
// XferBaseUs and XferBandwidthGBps must be measured on the target fabric; inheriting
// the simulator's executor settings mis-prices every disaggregated candidate in the
// same direction.
func (cfg *Config) cXferUs(promptTokens int) float64 {
	if !cfg.CXferSizeAware {
		return 0
	}
	bwBytesPerUs := cfg.XferBandwidthGBps * 1000.0
	if bwBytesPerUs <= 0 || cfg.BlockSize <= 0 {
		return cfg.XferBaseUs
	}
	transferBytes := float64(cfg.reqKVNeed(promptTokens)) * float64(cfg.BlockSize) * cfg.KVBytesPerTokenPerGPU
	return cfg.XferBaseUs + transferBytes/bwBytesPerUs
}

// projectedLocalTTFT is the arriving request's TTFT under a local placement. Local
// execution samples the first token when prefill completes, so there is no separate
// decode iteration before post-processing.
func (cfg *Config) projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc float64) float64 {
	return tAdmD + nChunks*tIterD + wpLoc + cfg.OutputTokenProcessingUs
}

// projectedDisaggTTFT is the arriving request's TTFT under a disaggregated placement.
// decodeJoinUs already covers prefill admission, prefill work, transfer, and decode
// admission; the first B+1 decode iteration and post-processing follow.
func (cfg *Config) projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64 {
	return decodeJoinUs + tIterFirstDecode + cfg.OutputTokenProcessingUs
}
