package causalsloexternality

import (
	"math"
	"sort"
)

// AdmissionContext is everything the admission estimator may read for one pool.
type AdmissionContext struct {
	// Mu is the live drain rate. It is assembled to mirror the specification's context even though
	// the rollforward estimator does not read it; the estimators that do are not selected here.
	Mu                float64
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64
	TIter             float64
	QueueDepth        int
	RemainingStepsEst float64
	Running           []ResidentKV
}

// ResidentKV is the per-resident slice the estimator walks: remaining steps and the KV
// blocks that departure would free.
//
// A negative Rem means "unknown", which the decode path enforces by censoring before the
// estimator sees it, so the estimator falls back to RemainingStepsEst. That censoring is
// preserved even though the shadow table could supply per-resident values: the published
// arm's estimator did not read them, and letting it would be a different estimator.
type ResidentKV struct {
	Rem      int64
	KVBlocks int64
}

// flooredTAdm lower-bounds an estimate by one iteration: even with a free slot, a request
// waits for the current step to finish before the next batch formation admits it.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
	if ctx.TIter > est {
		return ctx.TIter
	}
	return est
}

// rollforwardTAdm is the closed-form deterministic look-ahead. Each running request departs
// after its remaining steps, freeing its KV; accumulate until BOTH a slot and enough free KV
// exist.
//
// THIS IS THE SUBSTITUTE, NOT THE PUBLISHED ESTIMATOR. Every cited margin was produced with
// a scheduler rollout that replays the engine's scheduler over its ordered wait queue. That
// replay needs per-request wait-queue contents, which this target does not expose: Metrics
// carries a single WaitingQueueSize integer and there is no route to the running set's
// per-request prompt and computed token counts. Callers must record that the substitution
// happened; upstream substitutes silently, and silence is the defect rather than the
// fallback.
//
// Four behaviours are load-bearing:
//
//  1. Fast path: a free slot AND enough already-free KV means no wait beyond the
//     one-iteration floor.
//  2. needSlots is QueueDepth+1, not QueueDepth: the requests ahead take the first
//     QueueDepth freed slots and ours takes the next.
//  3. The departure sort is STABLE. An unstable sort makes the estimate depend on iteration
//     order, which would make the same fleet state score differently across runs.
//  4. When departures are exhausted before freeing QueueDepth+1 slots, it falls back to a
//     fluid WAVE form: slots free in waves of BatchSize every ~RemainingStepsEst iterations.
//     This branch is the biased one, and it UNDERSTATES deep-queue delay. A smaller
//     admission estimate means a smaller admission window, which means residents are charged
//     for MORE of the arrival's interference, so the local candidate is over-priced and the
//     policy is biased toward remote prefill relative to the published arm.
func rollforwardTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}

	type dep struct {
		step, kv int64
	}
	deps := make([]dep, 0, len(ctx.Running))
	for _, r := range ctx.Running {
		rem := r.Rem
		if rem < 0 {
			// Truncation, not rounding.
			rem = int64(ctx.RemainingStepsEst)
			if rem < 1 {
				rem = 1
			}
		}
		deps = append(deps, dep{step: rem, kv: r.KVBlocks})
	}
	sort.SliceStable(deps, func(i, j int) bool { return deps[i].step < deps[j].step })

	needSlots := int64(ctx.QueueDepth + 1)
	freeSlots := int64(ctx.MaxBatchSize - ctx.BatchSize)
	free := ctx.FreeKVBlocks
	for _, d := range deps {
		freeSlots++
		free += d.kv
		if freeSlots >= needSlots && free >= ctx.ReqKVNeed {
			return flooredTAdm(float64(d.step)*ctx.TIter, ctx)
		}
	}
	if ctx.BatchSize > 0 {
		waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
		return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
	}
	if len(deps) > 0 {
		return flooredTAdm(float64(deps[len(deps)-1].step)*ctx.TIter, ctx)
	}
	return flooredTAdm(0, ctx)
}

// runningMean is the per-class running mean of realized output lengths. The zero-completion
// seed is 1, a conservative one-token estimate rather than zero, which would make the decode
// demand vanish and every remaining-steps floor collapse.
type runningMean struct {
	n   int64
	sum float64
}

func (r *runningMean) update(v float64) { r.n++; r.sum += v }

func (r *runningMean) mean() float64 {
	if r.n == 0 {
		return 1
	}
	return r.sum / float64(r.n)
}

// decodeRemStepsEst is the CENSORED per-endpoint mean remaining-steps estimate. It is not a
// plain mean:
//
//  1. An empty endpoint returns 1.0, not 0.
//  2. nHatEff = max(class estimate, longest in-flight StepsDone). A resident that has
//     produced StepsDone tokens has output length at least StepsDone, so the class estimate
//     is floored by the longest elapsed. This is the censoring.
//  3. Each resident contributes max(nHatEff - StepsDone, 1), floored because it is still
//     decoding, and the result is averaged over the resident count.
//
// A plain mean goes negative once residents outlive the class estimate, which is exactly what
// the flooring prevents.
func decodeRemStepsEst(residents []Resident, nHatOut float64) float64 {
	n := len(residents)
	if n == 0 {
		return 1.0
	}
	var maxSteps int64
	for _, r := range residents {
		if r.StepsDone > maxSteps {
			maxSteps = r.StepsDone
		}
	}
	nHatEff := math.Max(nHatOut, float64(maxSteps))
	var sum float64
	for _, r := range residents {
		sum += math.Max(nHatEff-float64(r.StepsDone), 1)
	}
	return sum / float64(n)
}

// prefillRemStepsEst is the symmetric estimate over prefill occupants: each occupant's
// remaining chunk count floored at 1, averaged; an empty pool returns 1.0.
func prefillRemStepsEst(occupants []PrefillOccupant) float64 {
	n := len(occupants)
	if n == 0 {
		return 1.0
	}
	var sum float64
	for _, o := range occupants {
		rem := o.RemPrefillTokens
		if rem < 1 {
			rem = 1
		}
		sum += float64(rem)
	}
	return sum / float64(n)
}

// residentRem is the DEPLOYABLE censored remaining-steps value for one decode resident:
//
//	rem = max( max(nHatOut, StepsDone) - StepsDone, 1 )
//
// The inner max is the censoring, flooring the class estimate by what this resident has
// already produced before subtracting so the difference cannot go negative. The outer floor is
// 1 because the resident is still decoding.
//
// The focal arm always takes this branch: the deployable estimate is forced whenever the
// policy is active, because an oracle remaining is not a deployable policy.
func residentRem(nHatOut float64, stepsDone int64) int64 {
	return int64(math.Max(math.Max(nHatOut, float64(stepsDone))-float64(stepsDone), 1))
}

// occupantRemDecodeSteps is the decode-phase horizon of a pre-first-token occupant. It has
// produced no output, so the deployable estimate is the full class mean with nothing to
// subtract, floored at 1.
func occupantRemDecodeSteps(nHatOut float64) int64 {
	return int64(math.Max(nHatOut, 1))
}

// residentPrefillTokens is the capped estimate of what is being prefilled THIS STEP, which is
// not the outstanding prompt backlog.
//
// Summing each occupant's whole remaining prompt over-estimates by roughly the chunk count,
// about 9.8x at a 45k mean input. Each occupant is capped at one chunk and the total at the
// engine's batched-token budget, which is what the scheduler could actually admit in one step.
//
// Still an over-estimate, because the router cannot know which requests the engine scheduled,
// so the bias inflates the decode iteration time and points away from local prefill. On a
// clean fleet this is 0 either way, so the error only fires after the policy has already
// elected local prefill at least once.
func (cfg *Config) residentPrefillTokens(occupants []PrefillOccupant) int64 {
	chunk := int64(cfg.ChunkTokens)
	if chunk < 1 {
		chunk = 1
	}
	var sum int64
	for _, o := range occupants {
		rem := o.RemPrefillTokens
		if rem <= 0 {
			continue
		}
		if rem > chunk {
			rem = chunk
		}
		sum += rem
	}
	if sum > int64(cfg.ChunkTokens) {
		sum = int64(cfg.ChunkTokens)
	}
	return sum
}
