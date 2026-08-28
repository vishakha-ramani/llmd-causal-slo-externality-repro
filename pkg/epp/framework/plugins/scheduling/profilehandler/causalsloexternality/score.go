package causalsloexternality

import (
	"math"
	"sort"
)

// candidateScore is one candidate's decomposed objective.
//
// capacityTotal is retained and always zero for this arm. The ablation cohort's validity gate
// asserts capacity terms are exactly zero in every arm, and a port that drops the field cannot
// report that.
type candidateScore struct {
	decodeIteration      iterationTerms
	firstDecodeIteration iterationTerms
	prefillIteration     iterationTerms
	decodeAdmissionUs    float64
	prefillAdmissionUs   float64
	prefillWorkUs        float64
	transferUs           float64
	decodeJoinUs         float64
	projectedTTFTUs      float64
	prefillChunks        float64

	externalityBreakdown breakdown
	externality          float64
	ownGood              float64
	netGoodCost          float64
	capacityTotal        float64
	total                float64

	// rolloutSubstituted counts how many times this candidate's scoring fell back from the
	// published rollout estimator to the closed form.
	rolloutSubstituted int
}

// TERMS DELIBERATELY ABSENT FROM THIS SCORE. Each is a term a reimplementation is likely to
// add back, so each is named with what adding it would do.
//
// A1. NO STANDALONE TRANSFER PENALTY. The transfer cost is priced CAUSALLY THROUGH TIMING
// ONLY: it enters the remote lead, hence the decode-join clock, hence both the arrival's
// projected TTFT and the resident admission window. Adding an additive V*(c_xfer/tau_ttft)*
// (tau_ref/tau_ttft) penalty double-prices every transfer and biases the argmin toward local
// prefill by an amount scaling with V and with 1/tau_ttft^2.
//
// A2. NO TTFT/ITL VIRTUAL-QUEUE DEFICIT TERM. The upstream decision path still maintains
// normalized virtual queues and a deficit accumulator, but the candidate score reads neither.
// They are instrumentation, not the objective.
//
// A3. NO PER-DECISION NORMALIZATION. Min-max normalization of congestion against value-at-risk
// belongs to the drift-plus-VaR rule. This arm compares raw smooth SLO value across
// candidates, which is exactly why the score is directly comparable between candidates and why
// clamping it to [0,1] would destroy the ranking.

// scoreCandidate is the whole objective for one (decode, placement) candidate. A nil prefill
// state selects the local candidate.
//
// SCORE IDENTITY, gated by the registered campaign: with capacity disabled,
//
//	total = V * (externality - ownGood)
//
// and every traced candidate row must satisfy exactly that.
func (cfg *Config) scoreCandidate(
	promptTokens int, now float64, dec *endpointState, pre *endpointState,
	slo SLO, nHatOut float64,
) candidateScore {
	thetaD := dec.theta
	ar := promptTokens
	bDec, kv, sPfD := dec.batchSize, dec.kvTokensInUse, dec.residentPrefillTokens
	uPfD := dec.residentPrefillAttentionUnits
	decodeIteration := thetaD.iterationTerms(thetaD.AlphaD, bDec, kv, sPfD, uPfD)
	tIterD := decodeIteration.total
	// The first decode iteration the arrival itself experiences: B+1 with its own context
	// added.
	firstDecodeIteration := thetaD.iterationTerms(thetaD.AlphaD, bDec+1, kv+int64(ar), sPfD, uPfD)
	tIterFirstDecode := firstDecodeIteration.total

	tAdmD := rollforwardTAdm(cfg.decodeAdmissionCtx(promptTokens, dec, nHatOut))

	score := candidateScore{
		decodeIteration:      decodeIteration,
		firstDecodeIteration: firstDecodeIteration,
		decodeAdmissionUs:    tAdmD,
	}

	if pre == nil {
		// LOCAL.
		apLoc := dec.ap
		nChunksLoc, _ := cfg.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(atLeastZero(apLoc), ar)
		tHatLocal := cfg.projectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		score.prefillChunks = nChunksLoc
		score.prefillWorkUs = wpLoc
		score.projectedTTFTUs = tHatLocal

		// The published path replaces BOTH the admission delay and the TTFT when the engine's
		// scheduler state is observed. It never is at this pin, so the closed-form values
		// stand and the substitution is counted rather than hidden.
		if !dec.schedulerObserved {
			score.rolloutSubstituted++
		}

		score.externalityBreakdown = cfg.externalityLocal(promptTokens, now, dec, tAdmD)
		if !cfg.NoOwnGood {
			score.ownGood = cfg.ownGoodFor(promptTokens, dec, slo, tHatLocal, nHatOut)
		}
	} else {
		// DISAGGREGATED.
		//
		// When available the rollout would replace ONLY the decode admission here, modelling a
		// transferred request whose prompt is already computed but whose KV must still fit at
		// the candidate decoder.
		if !dec.schedulerObserved {
			score.rolloutSubstituted++
		}

		thetaP := pre.theta
		apP := pre.ap
		nChunksP, _ := cfg.chunkTerms(thetaP, apP)
		wpP := thetaP.Wp(atLeastZero(apP), ar)
		prefillIteration := thetaP.iterationTerms(thetaP.AlphaP, 0, 0,
			pre.residentPrefillTokens, pre.residentPrefillAttentionUnits)
		tIterP := prefillIteration.total
		tAdmP := rollforwardTAdm(cfg.prefillAdmissionCtx(promptTokens, pre))
		score.prefillIteration = prefillIteration
		score.prefillAdmissionUs = tAdmP
		score.prefillWorkUs = wpP
		score.prefillChunks = nChunksP

		prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
		if !pre.schedulerObserved {
			score.rolloutSubstituted++
		}

		score.transferUs = cfg.cXferUs(promptTokens)
		remoteLeadUs := prefillCompletionUs + score.transferUs

		// CONCURRENCY, NOT SERIALIZATION. The decode admission estimate is an ABSOLUTE wait
		// measured at the routing instant. Remote prefill and transfer consume part or all of
		// that interval while the decode queue keeps draining, so they must not be serialized
		// with the full estimate a second time. Writing this as remoteLeadUs + tAdmD
		// double-counts the decode wait and systematically over-prices every remote candidate.
		decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
		tHatDisagg := cfg.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)
		score.decodeJoinUs = decodeJoinUs
		score.projectedTTFTUs = tHatDisagg

		score.externalityBreakdown = cfg.externalityDisagg(promptTokens, now, dec, pre, decodeJoinUs, tAdmP)
		if !cfg.NoOwnGood {
			score.ownGood = cfg.ownGoodFor(promptTokens, dec, slo, tHatDisagg, nHatOut)
		}
	}

	if !cfg.NoExternality {
		score.externality = score.externalityBreakdown.total()
	}
	score.netGoodCost = score.externality - score.ownGood
	// capacityTotal stays zero: the capacity account is not reconstructed, and configuration
	// validation rejects a setting that would enable the term, so a zero here is the computed
	// value for this arm rather than a stand-in for an unimplemented one.
	score.total = cfg.V*score.netGoodCost + score.capacityTotal
	return score
}

// externalityLocal is the local candidate's externality.
//
// admissionSteps = ceil(decodeAdmissionUs / max(tIter0, 1)); the guard prevents a division
// blow-up at a degenerate iteration time.
func (cfg *Config) externalityLocal(
	promptTokens int, now float64, ep *endpointState, decodeAdmissionUs float64,
) breakdown {
	theta := ep.theta
	ar := promptTokens
	apLoc := ep.ap
	chunkLoc := cfg.chunkFor(apLoc)

	rt := newReTiming(theta, ep.batchSize, ep.kvTokensInUse, ep.residentPrefillTokens,
		ep.residentPrefillAttentionUnits, chunkLoc, int64(ar))
	rt = rt.withExactOverlap(theta, apLoc, ar)

	nChunksLoc, _ := cfg.chunkTerms(theta, apLoc)
	admissionSteps := math.Ceil(decodeAdmissionUs / math.Max(rt.tIter0, 1))

	v := breakdown{
		decode: decodeLocalSum(now, ep.runningDecode, rt, nChunksLoc, admissionSteps),
	}
	// Collocated-prefill pricing is forced on for this arm.
	if len(ep.runningPrefill) > 0 {
		v.collocPrefill = collocLocalSum(now, ep.runningPrefill, rt, float64(chunkLoc), nChunksLoc, admissionSteps)
	}
	return v
}

// externalityDisagg is the disaggregated candidate's externality.
//
// TWO CLOCKS, DELIBERATELY DISTINCT. decodeJoinUs is when the arrival BEGINS INTERFERING with
// the decode endpoint, supplied already composed as max(remoteLead, tAdmD) because remote
// prefill and decode-queue drainage are concurrent clocks from the routing instant. The
// client-visible TTFT is a different quantity and is not used here.
//
// The decode-side re-timing uses the DECODE endpoint's coefficients even though the prefill
// happens elsewhere, because decode happens on the decode endpoint under both placements. The
// chunk width, however, is the prefill pool's.
func (cfg *Config) externalityDisagg(
	promptTokens int, now float64, dec *endpointState, pre *endpointState,
	decodeJoinUs, prefillAdmissionUs float64,
) breakdown {
	thetaD := dec.theta
	thetaP := pre.theta
	ar := promptTokens
	apP := pre.ap
	chunkP := cfg.chunkFor(apP)

	rt := newReTiming(thetaD, dec.batchSize, dec.kvTokensInUse, dec.residentPrefillTokens,
		dec.residentPrefillAttentionUnits, chunkP, int64(ar))
	rt = rt.withExactOverlap(thetaD, apP, ar)

	tIterP := thetaP.tIterPrefill(pre.residentPrefillTokens, pre.residentPrefillAttentionUnits)

	arrivalSteps := math.Ceil(decodeJoinUs / math.Max(rt.tIter0, 1))
	v := breakdown{
		decode: decodeDisaggSum(now, dec.runningDecode, rt, arrivalSteps),
	}
	// Collocated occupants on the decode endpoint are undisturbed until the arrival comes over
	// from the pool.
	if len(dec.runningPrefill) > 0 {
		v.collocPrefill = collocDisaggSum(now, dec.runningPrefill, rt, float64(chunkP), arrivalSteps)
	}
	prefillAdmissionSteps := math.Ceil(prefillAdmissionUs / math.Max(tIterP, 1))
	v.prefillPool = prefillPoolSum(now, pre.runningPrefill, tIterP, float64(chunkP), prefillAdmissionSteps,
		atLeastZero(apP), ar, thetaP)

	// TWO UPSTREAM TERMS ARE DEAD ON THIS ARM'S PATH, and neither is an omission. The
	// arrival's full serial prefill duration on the pool feeds only the naive pool-charging
	// branch, which the exact per-occupant marginal overlap replaces; and the serial
	// reconstruction of the decode-join clock is overridden by the concurrent value the caller
	// supplies. Reintroducing the first over-charges the pool by roughly the occupant count;
	// reintroducing the second serializes two concurrent clocks.
	return v
}

// ownGoodFor evaluates the arriving request's own value at a candidate.
//
// The re-timing is built with chunk = 0: the prefill chunk affects only the overlap window, not
// the post-join iteration time, so passing the real chunk here would be wrong. Decode always
// happens on the decode endpoint, so its coefficients set the post-join rate for both the local
// and the disaggregated candidate.
func (cfg *Config) ownGoodFor(
	promptTokens int, ep *endpointState, slo SLO, tHat, nHatOut float64,
) float64 {
	rt := newReTiming(ep.theta, ep.batchSize, ep.kvTokensInUse, ep.residentPrefillTokens,
		ep.residentPrefillAttentionUnits, 0, int64(promptTokens))
	return goodSelf(slo, tHat, rt.tIterAfter, nHatOut)
}

// decodeAdmissionCtx assembles the decode-pool admission context.
//
// Running is CENSORED: per-resident remaining steps are withheld so the estimator falls back to
// the aggregate estimate. That is preserved even though the shadow table could supply
// per-resident values, because the published arm's estimator did not read them.
func (cfg *Config) decodeAdmissionCtx(promptTokens int, ep *endpointState, nHatOut float64) AdmissionContext {
	theta := ep.theta
	bDec, kv, sPf := ep.batchSize, ep.kvTokensInUse, ep.residentPrefillTokens
	uPf := ep.residentPrefillAttentionUnits
	censored := make([]ResidentKV, len(ep.runningDecode))
	for i, r := range ep.runningDecode {
		censored[i] = ResidentKV{Rem: -1, KVBlocks: residentKVBlocks(r, ep.engineBlockSize)}
	}
	return AdmissionContext{
		Mu:                theta.muDecode(bDec, kv, sPf, uPf),
		BatchSize:         ep.batchSize,
		MaxBatchSize:      int(ep.maxBatchSize),
		FreeKVBlocks:      ep.freeKVBlocks,
		ReqKVNeed:         cfg.reqKVNeed(promptTokens),
		TIter:             theta.tIterDecode(bDec, kv, sPf, uPf),
		QueueDepth:        ep.queueDepth,
		RemainingStepsEst: decodeRemStepsEst(ep.runningDecode, nHatOut),
		Running:           censored,
	}
}

// prefillAdmissionCtx is the prefill-pool counterpart. The asymmetry against the decode side is
// deliberate: prefill remaining is KNOWN INPUT, being remaining prompt tokens, so it is not
// censored.
func (cfg *Config) prefillAdmissionCtx(promptTokens int, ep *endpointState) AdmissionContext {
	theta := ep.theta
	sPf := ep.residentPrefillTokens
	uPf := ep.residentPrefillAttentionUnits
	running := make([]ResidentKV, 0, len(ep.runningPrefill))
	for _, o := range ep.runningPrefill {
		running = append(running, ResidentKV{Rem: o.RemPrefillTokens, KVBlocks: occupantKVBlocks(o, ep.engineBlockSize)})
	}
	return AdmissionContext{
		Mu:                theta.muPrefill(sPf, uPf),
		BatchSize:         ep.batchSize,
		MaxBatchSize:      int(ep.maxBatchSize),
		FreeKVBlocks:      ep.freeKVBlocks,
		ReqKVNeed:         cfg.reqKVNeed(promptTokens),
		TIter:             theta.tIterPrefill(sPf, uPf),
		QueueDepth:        ep.queueDepth,
		RemainingStepsEst: prefillRemStepsEst(ep.runningPrefill),
		Running:           running,
	}
}

// residentKVBlocks is the KV pages a decode resident holds, and therefore the pages its
// departure would free.
//
// Derived, not observed: no per-request KV figure is exported, so it is reconstructed as
// ceil((InputLen + StepsDone) / engineBlockSize) using the same integer ceiling the admission
// accounting uses, so the two agree.
//
// The block size is the ENGINE's, not the prefix producer's: this accounts for engine KV pages
// and the two block sizes are independently configurable.
func residentKVBlocks(r Resident, engineBlockSize int64) int64 {
	return ceilBlocks(r.InputLen+r.StepsDone, engineBlockSize)
}

// occupantKVBlocks is the same reconstruction for a pre-first-token occupant, which has streamed
// no tokens, so only its already-computed prompt span is resident.
func occupantKVBlocks(o PrefillOccupant, engineBlockSize int64) int64 {
	computed := o.InputLen - o.RemPrefillTokens
	if computed < 0 {
		computed = 0
	}
	return ceilBlocks(computed, engineBlockSize)
}

// decision is the committed placement.
type decision struct {
	disaggregate bool
	decodeID     string
	prefillID    string
	score        float64

	// rolloutSubstituted carries the winning candidate's substitution count so the caller can
	// report that the published estimator was unavailable.
	rolloutSubstituted int
}

type candidateEvaluation struct {
	decision decision
	score    candidateScore
	decode   *endpointState
	prefill  *endpointState
}

// decide is the joint argmin over the FULL cross product: D local candidates plus D*P
// disaggregated candidates, all scored on one scale and compared in one argmin.
//
// This shape is load-bearing, not stylistic. Fixing the decode endpoint first and then asking a
// binary gate about it is a different algorithm with a measured cost of -0.0485 equal-cell mean
// goodput and -0.1555 worst cell.
//
// ENUMERATION ORDER AND TIE-BREAKING, both load-bearing for determinism:
//
//  1. Endpoints are sorted ascending by id, so candidate order is stable across runs. The
//     caller CANNOT rely on the framework for this: the slice handed to a picker is built by
//     ranging a Go map, whose order is randomized per iteration.
//
//  2. The router's preferred endpoint is then moved to the FRONT. This is ONLY a tie-break
//     device: enumerating it first means it wins an exact tie. It must never NARROW the
//     candidate set, which would silently become the decode-first decomposition.
//
//     THE PREFERENCE IS EMPTY IN THIS DEPLOYMENT, and that is faithful rather than unwired. It
//     exists upstream to let an EXISTING scorer's pick survive an exact tie, but the role-blind
//     profile this policy runs carries NO scorers at all -- deliberately, because every scorer
//     contribution would be clamped to [0,1] while J is signed and unbounded. With no scorer there
//     is no ranking to inherit, so there is nothing to prefer and enumeration falls back to pure id
//     order. Passing a preference is retained as a parameter because it is the documented seam, and
//     because wiring a load scorer here and letting it NARROW the set is the specific mistake that
//     turns this into the decode-first decomposition.
//
//  3. The comparison is strict-with-epsilon: a candidate replaces the incumbent only if it is
//     better by more than 1e-12, so near-ties keep the incumbent. Plain < would make the
//     outcome depend on float noise.
//
//  4. For each decode endpoint the LOCAL candidate is enumerated BEFORE that endpoint's
//     disaggregated candidates, so local wins a tie against remote on the same decoder.
//
// NO EMPTY-PROMPT OR FULLY-CACHED GUARD, DELIBERATELY. With no uncached tokens the chunk terms
// return zero and the prefill demand is zero, so a fully-cached request still gets a
// well-defined score on every candidate and the argmin still chooses which DECODE endpoint to
// use. An early return would throw that choice away and revert decode selection to whatever
// order the candidates arrived in.
func (cfg *Config) decide(
	promptTokens int, now float64,
	decodes, prefills []*endpointState,
	preferredDecode string,
	slo SLO, nHatOut float64,
) decision {
	best, _ := cfg.decideDetailed(promptTokens, now, decodes, prefills, preferredDecode, slo, nHatOut)
	return best
}

func (cfg *Config) decideDetailed(
	promptTokens int, now float64,
	decodes, prefills []*endpointState,
	preferredDecode string,
	slo SLO, nHatOut float64,
) (decision, []candidateEvaluation) {
	orderedD := scorerFirst(sortByID(decodes), preferredDecode)
	if cfg.Decomposed && len(orderedD) > 1 {
		orderedD = orderedD[:1]
	}
	// The prefill pool carries no preference: the tie-break device exists to let an EXISTING router
	// preference survive an exact tie, and no upstream scorer ranks the prefill pool here. Pure id
	// order is what keeps enumeration deterministic.
	orderedP := sortByID(prefills)

	var best *decision
	evaluations := make([]candidateEvaluation, 0, len(orderedD)*(len(orderedP)+1))
	consider := func(d decision) {
		if best == nil || d.score < best.score-scoreEpsilon {
			c := d
			best = &c
		}
	}

	for _, dec := range orderedD {
		s := cfg.scoreCandidate(promptTokens, now, dec, nil, slo, nHatOut)
		d := decision{
			disaggregate: false, decodeID: dec.id, score: s.total,
			rolloutSubstituted: s.rolloutSubstituted,
		}
		evaluations = append(evaluations, candidateEvaluation{decision: d, score: s, decode: dec})
		consider(d)
		for _, pre := range orderedP {
			// A dual-role endpoint is a legal member of both pools, so the cross product would pair
			// it with itself. That pair is not a disaggregated placement: prefilling and decoding on
			// one endpoint IS the local candidate, already enumerated above. Scoring it again as
			// remote would double-count the same endpoint's residents, price a transfer that never
			// happens, and let a spurious duplicate win the argmin.
			if pre.id == dec.id {
				continue
			}
			s = cfg.scoreCandidate(promptTokens, now, dec, pre, slo, nHatOut)
			d = decision{
				disaggregate: true, decodeID: dec.id, prefillID: pre.id, score: s.total,
				rolloutSubstituted: s.rolloutSubstituted,
			}
			evaluations = append(evaluations, candidateEvaluation{decision: d, score: s, decode: dec, prefill: pre})
			consider(d)
		}
	}

	if best == nil {
		return decision{}, evaluations
	}
	return *best, evaluations
}

// scoreEpsilon is the strict-comparison margin. Near-ties keep the incumbent, which combined
// with the preferred-endpoint ordering means an existing router preference survives a tie.
const scoreEpsilon = 1e-12

func sortByID(eps []*endpointState) []*endpointState {
	if len(eps) == 0 {
		return nil
	}
	out := make([]*endpointState, len(eps))
	copy(out, eps)
	sort.SliceStable(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// scorerFirst preserves id order except that the preferred endpoint moves to the front. A no-op
// when preferred is empty, when there are fewer than two endpoints, or when it is already first.
func scorerFirst(eps []*endpointState, preferred string) []*endpointState {
	if preferred == "" || len(eps) < 2 || eps[0].id == preferred {
		return eps
	}
	out := make([]*endpointState, 0, len(eps))
	for _, e := range eps {
		if e.id == preferred {
			out = append(out, e)
		}
	}
	for _, e := range eps {
		if e.id != preferred {
			out = append(out, e)
		}
	}
	return out
}
