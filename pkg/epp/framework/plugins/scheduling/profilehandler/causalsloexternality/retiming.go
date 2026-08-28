package causalsloexternality

import "math"

// reTiming holds the BATCH-LEVEL per-iteration decode times. Every resident in one decode
// batch shares one per-iteration time, so these are batch-level and not per-resident.
//
// "Full re-timing" means tIterAfter is RECOMPUTED at B+1 with the arrival's context
// added, not derived as a marginal addition to tIter0.
//
// The exact marginal-prefill model is always armed. The focal arm forces it on an internal
// copy of the decider rather than exposing it, so the legacy inexact form is unreachable
// and is not implemented here.
type reTiming struct {
	// tIter0 is the current batch's per-iteration time, the undisturbed baseline.
	tIter0 float64
	// tIterAfter is the per-iteration time once the arrival has joined the decode batch.
	tIterAfter float64

	cAttn float64
	chunk float64
	cPf   float64
	// ap is floored at 0 while ar is the raw input length.
	ap float64
	ar float64
}

// newReTiming builds the batch-level times. deltaKv is the arrival's FULL input length,
// its resident context once it joins the batch, so no output-length estimate is needed.
func newReTiming(theta Coeffs, bDec int, kv, sPf int64, uPf float64, chunk int, deltaKv int64) reTiming {
	return reTiming{
		tIter0:     theta.tIterDecode(bDec, kv, sPf, uPf),
		tIterAfter: theta.tIterDecode(bDec+1, kv+deltaKv, sPf, uPf),
		cAttn:      theta.CAttn,
		chunk:      float64(chunk),
	}
}

// withExactOverlap arms the exact marginal-prefill model.
func (rt reTiming) withExactOverlap(theta Coeffs, ap, ar int) reTiming {
	rt.cPf = theta.CPf
	rt.ap = float64(atLeastZero(ap))
	rt.ar = float64(ar)
	return rt
}

// cBase is a resident's baseline completion: rem steps at the current per-iteration time.
func (rt reTiming) cBase(now float64, rem int64) float64 {
	return now + float64(rem)*rt.tIter0
}

// cLocalAfter is a resident's completion under a LOCAL placement, where the arrival waits
// admissionSteps baseline iterations before joining.
//
// Three windows, in order:
//
//  1. pre = min(max(admissionSteps,0), rem) iterations run UNDISTURBED. This is the causal
//     gate: a resident that finishes inside the admission window has cLocalAfter == cBase
//     and contributes exactly zero. That gating is what makes the term an externality
//     rather than another load proxy.
//  2. overlap = min(nChunks, rem-pre) iterations run while the arrival's prefill is
//     co-scheduled, charged at tIter0 PLUS the marginal prefill work only. The baseline
//     iteration time is excluded because residents would pay it even if the arriving
//     request were absent, which is what makes the charge marginal.
//  3. the remaining tail runs at the B+1 re-timed rate.
func (rt reTiming) cLocalAfter(now float64, rem int64, admissionSteps, nChunks float64) float64 {
	pre := math.Min(math.Max(admissionSteps, 0), float64(rem))
	remaining := float64(rem) - pre
	overlap := math.Min(nChunks, remaining)
	return now + pre*rt.tIter0 + overlap*rt.tIter0 +
		prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
		(remaining-overlap)*rt.tIterAfter
}

// cDisagg is a resident's completion under a DISAGGREGATED placement. The arrival prefills
// remotely, so the decode endpoint is undisturbed for arrivalSteps iterations and only the
// tail is re-timed. There is no prefill-overlap inflation on this endpoint, and that
// asymmetry against cLocalAfter is the mechanism's lever.
func (rt reTiming) cDisagg(now float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(arrivalSteps, float64(rem))
	return now + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

// prefillMarginalWork is the exact work added by the first `iterations` chunks of the
// arrival's uncached prefill:
//
//	processed    = min(ap, iterations*chunk)
//	cachedPrefix = max(ar - ap, 0)
//	work         = CPf*processed + CAttn*processed*(cachedPrefix + processed/2)
//
// It EXCLUDES baseline iteration time.
func prefillMarginalWork(cPf, cAttn, ap, ar, chunk, iterations float64) float64 {
	if ap <= 0 || chunk <= 0 || iterations <= 0 {
		return 0
	}
	processed := math.Min(ap, iterations*chunk)
	cachedPrefix := math.Max(ar-ap, 0)
	return cPf*processed + cAttn*processed*(cachedPrefix+processed/2.0)
}
