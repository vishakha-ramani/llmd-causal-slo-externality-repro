package causalsloexternality

import "math"

// Resident is one decode resident's inputs, sourced from the shadow table. A negative
// Rem marks a resident whose remaining steps are unknown; it contributes zero and is
// skipped rather than guessed at.
type Resident struct {
	RequestID    string
	Rem          int64
	ArrivalUs    int64
	FirstTokenUs int64
	TTFTSet      bool
	Slo          SLO
	// StepsDone and InputLen support the censored remaining-steps estimate and the
	// per-resident KV reconstruction.
	StepsDone int64
	InputLen  int64
}

// PrefillOccupant is one pre-first-token occupant. RemDecodeSteps is its decode-phase
// horizon once it reaches first token; a non-positive value leaves the decode phase
// unpriced, recovering TTFT-only behaviour.
type PrefillOccupant struct {
	RequestID        string
	RemPrefillTokens int64
	RemDecodeSteps   int64
	ArrivalUs        int64
	Slo              SLO
	InputLen         int64
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// sloCompositeValue is the bounded routing value. Each ENABLED dimension uses its own
// SLO threshold as the transition bandwidth, so the argument is a dimensionless relative
// slack. A disabled target contributes a factor of exactly 1, not 0.
//
// Mean ITL is deliberately absent. The routing value is a smooth TTFT x E2E surrogate;
// reported goodput remains the hard conjunction of TTFT, mean ITL and E2E. Adding an ITL
// factor here would change the arm.
func sloCompositeValue(slo SLO, ttftUs, e2eUs float64) float64 {
	u := 1.0
	if slo.TauTTFTUs > 0 {
		u *= sigmoid((slo.TauTTFTUs - ttftUs) / slo.TauTTFTUs)
	}
	if slo.TauE2EUs > 0 {
		u *= sigmoid((slo.TauE2EUs - e2eUs) / slo.TauE2EUs)
	}
	return u
}

// gDecodeComposite is a decode resident's value at a projected completion. Its TTFT is
// REALIZED and therefore fixed under this decision, so only the E2E factor moves.
//
// A resident with no first token yet returns exactly 0, so it contributes nothing to the
// before-minus-after difference. Such a request is picked up instead by the collocated
// occupant term, whose TTFT is still live.
func gDecodeComposite(r Resident, cUs float64) float64 {
	if !r.TTFTSet {
		return 0
	}
	ttft := float64(r.FirstTokenUs - r.ArrivalUs)
	return sloCompositeValue(r.Slo, ttft, cUs-float64(r.ArrivalUs))
}

// gCollocComposite is a pre-first-token occupant's value. Deliberately TTFT-only, and
// the end-to-end argument is ignored: a request still prefilling has no assigned decoder
// state in the routing view, so pricing a synthetic decode horizon would make the
// one-step potential depend on state the router does not observe.
func gCollocComposite(o PrefillOccupant, ftUs, _ float64) float64 {
	if o.Slo.TauTTFTUs <= 0 {
		return 1
	}
	return sigmoid((o.Slo.TauTTFTUs - (ftUs - float64(o.ArrivalUs))) / o.Slo.TauTTFTUs)
}

// residentContribution is one decode resident's charge: value before minus value after.
// Positive means the placement destroys value.
//
// The realized-TTFT factor is a common positive multiplier on both terms, which is why a
// late-recorded first token shrinks every resident's charge rather than biasing it in a
// direction that depends on the resident.
func residentContribution(r Resident, cb, cp float64) float64 {
	return gDecodeComposite(r, cb) - gDecodeComposite(r, cp)
}

// occupantContribution is one collocated occupant's charge. The end-to-end completions
// are threaded through even though the kernel ignores them, so a future kernel that does
// use them is a one-line change rather than a re-derivation.
func occupantContribution(o PrefillOccupant, ftB, ftP, eB, eP float64) float64 {
	return gCollocComposite(o, ftB, eB) - gCollocComposite(o, ftP, eP)
}

// prefillTTFTContribution is a prefill-pool occupant's first-token charge. This is NOT
// gCollocComposite: the deadline is built as arrival + tau_ttft and the sigmoid argument
// is (deadline - completion)/scale. Algebraically the same quantity, written against an
// absolute deadline to match the form it is derived from.
func prefillTTFTContribution(o PrefillOccupant, cb, cp float64) float64 {
	deadline := float64(o.ArrivalUs) + o.Slo.TauTTFTUs
	scale := o.Slo.TauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return sigmoid((deadline-cb)/scale) - sigmoid((deadline-cp)/scale)
}

// goodSelf is the arriving request's own smoothed value under a candidate. The arrival is
// at the decision instant, so its TTFT measured from arrival IS tHatFromNow. It then
// decodes nOut steps at tIterAfter, the B+1 re-timed rate it will experience once it
// joins the batch.
//
// The own-good term is not attributable: the ablation cannot distinguish it from noise,
// and the arm with it removed scores a higher equal-cell mean. It is computed because the
// focal arm computes it, and removing it would be a different arm.
func goodSelf(slo SLO, tHatFromNow, tIterAfter, nOut float64) float64 {
	ttft := tHatFromNow
	e2e := tHatFromNow + nOut*tIterAfter
	return sloCompositeValue(slo, ttft, e2e)
}
