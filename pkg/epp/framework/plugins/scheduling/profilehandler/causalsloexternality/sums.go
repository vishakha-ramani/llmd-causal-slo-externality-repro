package causalsloexternality

import "math"

// breakdown separates the three populations a placement can harm. The objective uses only
// total(); the split is kept explicit so an aggregate is auditable in a decision trace.
type breakdown struct {
	decode        float64
	collocPrefill float64
	prefillPool   float64
}

func (b breakdown) total() float64 { return b.decode + b.collocPrefill + b.prefillPool }

// decodeLocalSum charges the decode endpoint's residents under a local placement.
// Residents with unknown remaining steps are skipped, contributing zero.
func decodeLocalSum(now float64, residents []Resident, rt reTiming, nChunks, admissionSteps float64) float64 {
	var sum float64
	for _, r := range residents {
		if r.Rem < 0 {
			continue
		}
		cb := rt.cBase(now, r.Rem)
		cp := rt.cLocalAfter(now, r.Rem, admissionSteps, nChunks)
		sum += residentContribution(r, cb, cp)
	}
	return sum
}

// decodeDisaggSum is the disaggregated mirror.
func decodeDisaggSum(now float64, residents []Resident, rt reTiming, arrivalSteps float64) float64 {
	var sum float64
	for _, r := range residents {
		if r.Rem < 0 {
			continue
		}
		cb := rt.cBase(now, r.Rem)
		cp := rt.cDisagg(now, r.Rem, arrivalSteps)
		sum += residentContribution(r, cb, cp)
	}
	return sum
}

// collocLocalSum charges the DECODE endpoint's pre-first-token occupants under a local
// placement.
//
// Such an occupant was placed here by a prior local decision and has not produced its first
// token, so the decode-side sums skip it entirely and would miss it. A local placement harms
// it twice: the arrival's co-scheduled prefill slows the iterations it still needs to reach
// first token, and then the arrival joins the decode batch and slows its decode steps too.
//
// Both horizons project through the same cBase/cLocalAfter model. When the decode horizon is
// unknown only the first-token risk is priced.
func collocLocalSum(now float64, occupants []PrefillOccupant, rt reTiming, chunk, nChunks, admissionSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, o := range occupants {
		if o.RemPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(o.RemPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cLocalAfter(now, remPf, admissionSteps, nChunks)
		eB, eP := ftB, ftP
		if o.RemDecodeSteps > 0 {
			total := remPf + o.RemDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cLocalAfter(now, total, admissionSteps, nChunks)
		}
		sum += occupantContribution(o, ftB, ftP, eB, eP)
	}
	return sum
}

// collocDisaggSum is the disaggregated mirror. An occupant that reaches its first token
// within the arrival window sees cDisagg == cBase and contributes exactly zero.
func collocDisaggSum(now float64, occupants []PrefillOccupant, rt reTiming, chunk, arrivalSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, o := range occupants {
		if o.RemPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(o.RemPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cDisagg(now, remPf, arrivalSteps)
		eB, eP := ftB, ftP
		if o.RemDecodeSteps > 0 {
			total := remPf + o.RemDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cDisagg(now, total, arrivalSteps)
		}
		sum += occupantContribution(o, ftB, ftP, eB, eP)
	}
	return sum
}

// prefillPoolSum charges the PREFILL POOL's occupants for the arrival's remote prefill.
//
// The correction over the naive form matters: after the arrival is admitted, an occupant is
// delayed only by the arrival's chunks that execute BEFORE that occupant reaches its own
// first token. Baseline prefill iterations are not charged again, and an occupant with one
// remaining iteration is not charged the arrival's entire multi-chunk prompt. The naive form
// over-charges the pool by roughly the occupant count.
//
// Where the pool's occupants are unobserved this sum is 0, which under-prices remote prefill.
func prefillPoolSum(
	now float64, occupants []PrefillOccupant,
	tIterP, chunkP, admissionSteps float64,
	rAp, rAr int, theta Coeffs,
) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	rChunks := math.Ceil(float64(atLeastZero(rAp)) / chunkP)
	var sum float64
	for _, o := range occupants {
		if o.RemPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(o.RemPrefillTokens) / chunkP)
		remainingAfterAdmission := math.Max(remIters-admissionSteps, 0)
		overlap := math.Min(rChunks, remainingAfterAdmission)
		cb := now + remIters*tIterP
		cp := cb + prefillMarginalWork(theta.CPf, theta.CAttn, float64(rAp), float64(rAr), chunkP, overlap)
		sum += prefillTTFTContribution(o, cb, cp)
	}
	return sum
}
