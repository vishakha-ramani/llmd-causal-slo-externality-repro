package causalsloexternality

import (
	"math"
	"time"
)

const (
	causalObservationSchemaVersion = "causal-slo-externality/v1"

	eventDecisionCandidate          = "decision_candidate"
	eventDecisionResidentPrediction = "decision_resident_prediction"
	eventDecisionSelected           = "decision_selected"
	eventEPPFirstToken              = "epp_first_token"
	eventEPPComplete                = "epp_complete"
)

type candidatePrediction struct {
	PrefillCompletionUs float64
	RemoteLeadUs        float64
	ProjectedE2EUs      float64
}

func newCandidatePrediction(score candidateScore, nHatOut float64) candidatePrediction {
	prefillCompletionUs := score.prefillAdmissionUs +
		score.prefillChunks*score.prefillIteration.total + score.prefillWorkUs
	return candidatePrediction{
		PrefillCompletionUs: prefillCompletionUs,
		RemoteLeadUs:        prefillCompletionUs + score.transferUs,
		ProjectedE2EUs:      score.projectedTTFTUs + nHatOut*score.firstDecodeIteration.total,
	}
}

type externalityObservation struct {
	Population             string
	ResidentRequestID      string
	Target                 string
	Eligible               bool
	SkipReason             string
	RemainingDecodeSteps   int64
	RemainingPrefillTokens int64
	ArrivalUs              int64
	RealizedFirstTokenUs   int64
	TauTTFTUs              float64
	TauE2EUs               float64
	BaselineTargetUs       float64
	CandidateTargetUs      float64
	BaselineFirstTokenUs   float64
	CandidateFirstTokenUs  float64
	BaselineCompletionUs   float64
	CandidateCompletionUs  float64
	BaselineValue          float64
	CandidateValue         float64
	Contribution           float64
}

// externalityObservations returns the same per-resident terms summed by scoreCandidate. These
// records are produced only on the debug logging path, so normal routing does not allocate one
// record per resident.
func (cfg *Config) externalityObservations(
	promptTokens int, now float64, dec, pre *endpointState, score candidateScore,
) []externalityObservation {
	if pre == nil {
		return cfg.localExternalityObservations(promptTokens, now, dec, score)
	}
	return cfg.disaggExternalityObservations(promptTokens, now, dec, pre, score)
}

func (cfg *Config) localExternalityObservations(
	promptTokens int, now float64, ep *endpointState, score candidateScore,
) []externalityObservation {
	rt := newReTiming(ep.theta, ep.batchSize, ep.kvTokensInUse, ep.residentPrefillTokens,
		ep.residentPrefillAttentionUnits, cfg.chunkFor(ep.ap), int64(promptTokens)).
		withExactOverlap(ep.theta, ep.ap, promptTokens)
	admissionSteps := math.Ceil(score.decodeAdmissionUs / math.Max(rt.tIter0, 1))
	observations := make([]externalityObservation, 0, len(ep.runningDecode)+len(ep.runningPrefill))

	for _, resident := range ep.runningDecode {
		observation := externalityObservation{
			Population:           "decode",
			ResidentRequestID:    resident.RequestID,
			Target:               "completion",
			Eligible:             resident.Rem >= 0,
			RemainingDecodeSteps: resident.Rem,
			ArrivalUs:            resident.ArrivalUs,
			RealizedFirstTokenUs: resident.FirstTokenUs,
			TauTTFTUs:            resident.Slo.TauTTFTUs,
			TauE2EUs:             resident.Slo.TauE2EUs,
		}
		if resident.Rem < 0 {
			observation.SkipReason = "remaining_decode_steps_unknown"
			observations = append(observations, observation)
			continue
		}
		observation.BaselineTargetUs = rt.cBase(now, resident.Rem)
		observation.CandidateTargetUs = rt.cLocalAfter(now, resident.Rem, admissionSteps, score.prefillChunks)
		observation.BaselineCompletionUs = observation.BaselineTargetUs
		observation.CandidateCompletionUs = observation.CandidateTargetUs
		observation.BaselineValue = gDecodeComposite(resident, observation.BaselineTargetUs)
		observation.CandidateValue = gDecodeComposite(resident, observation.CandidateTargetUs)
		observation.Contribution = observation.BaselineValue - observation.CandidateValue
		observations = append(observations, observation)
	}

	chunk := math.Max(float64(cfg.chunkFor(ep.ap)), 1)
	for _, occupant := range ep.runningPrefill {
		observation := collocatedObservation(occupant, now, rt, chunk, admissionSteps, score.prefillChunks, true)
		observations = append(observations, observation)
	}
	return observations
}

func (cfg *Config) disaggExternalityObservations(
	promptTokens int, now float64, dec, pre *endpointState, score candidateScore,
) []externalityObservation {
	chunk := math.Max(float64(cfg.chunkFor(pre.ap)), 1)
	rt := newReTiming(dec.theta, dec.batchSize, dec.kvTokensInUse, dec.residentPrefillTokens,
		dec.residentPrefillAttentionUnits, int(chunk), int64(promptTokens)).
		withExactOverlap(dec.theta, pre.ap, promptTokens)
	arrivalSteps := math.Ceil(score.decodeJoinUs / math.Max(rt.tIter0, 1))
	observations := make([]externalityObservation, 0,
		len(dec.runningDecode)+len(dec.runningPrefill)+len(pre.runningPrefill))

	for _, resident := range dec.runningDecode {
		observation := externalityObservation{
			Population:           "decode",
			ResidentRequestID:    resident.RequestID,
			Target:               "completion",
			Eligible:             resident.Rem >= 0,
			RemainingDecodeSteps: resident.Rem,
			ArrivalUs:            resident.ArrivalUs,
			RealizedFirstTokenUs: resident.FirstTokenUs,
			TauTTFTUs:            resident.Slo.TauTTFTUs,
			TauE2EUs:             resident.Slo.TauE2EUs,
		}
		if resident.Rem < 0 {
			observation.SkipReason = "remaining_decode_steps_unknown"
			observations = append(observations, observation)
			continue
		}
		observation.BaselineTargetUs = rt.cBase(now, resident.Rem)
		observation.CandidateTargetUs = rt.cDisagg(now, resident.Rem, arrivalSteps)
		observation.BaselineCompletionUs = observation.BaselineTargetUs
		observation.CandidateCompletionUs = observation.CandidateTargetUs
		observation.BaselineValue = gDecodeComposite(resident, observation.BaselineTargetUs)
		observation.CandidateValue = gDecodeComposite(resident, observation.CandidateTargetUs)
		observation.Contribution = observation.BaselineValue - observation.CandidateValue
		observations = append(observations, observation)
	}

	for _, occupant := range dec.runningPrefill {
		observation := collocatedObservation(occupant, now, rt, chunk, arrivalSteps, 0, false)
		observations = append(observations, observation)
	}

	prefillAdmissionSteps := math.Ceil(score.prefillAdmissionUs / math.Max(score.prefillIteration.total, 1))
	arrivalChunks := math.Ceil(float64(atLeastZero(pre.ap)) / chunk)
	for _, occupant := range pre.runningPrefill {
		observation := externalityObservation{
			Population:             "prefill_pool",
			ResidentRequestID:      occupant.RequestID,
			Target:                 "first_token",
			Eligible:               occupant.RemPrefillTokens >= 0,
			RemainingDecodeSteps:   occupant.RemDecodeSteps,
			RemainingPrefillTokens: occupant.RemPrefillTokens,
			ArrivalUs:              occupant.ArrivalUs,
			TauTTFTUs:              occupant.Slo.TauTTFTUs,
			TauE2EUs:               occupant.Slo.TauE2EUs,
		}
		if occupant.RemPrefillTokens < 0 {
			observation.SkipReason = "remaining_prefill_tokens_unknown"
			observations = append(observations, observation)
			continue
		}
		remainingIterations := math.Ceil(float64(occupant.RemPrefillTokens) / chunk)
		remainingAfterAdmission := math.Max(remainingIterations-prefillAdmissionSteps, 0)
		overlap := math.Min(arrivalChunks, remainingAfterAdmission)
		observation.BaselineTargetUs = now + remainingIterations*score.prefillIteration.total
		observation.CandidateTargetUs = observation.BaselineTargetUs + prefillMarginalWork(
			pre.theta.CPf, pre.theta.CAttn, float64(pre.ap), float64(promptTokens), chunk, overlap)
		observation.BaselineFirstTokenUs = observation.BaselineTargetUs
		observation.CandidateFirstTokenUs = observation.CandidateTargetUs
		observation.BaselineValue = prefillValue(occupant, observation.BaselineTargetUs)
		observation.CandidateValue = prefillValue(occupant, observation.CandidateTargetUs)
		observation.Contribution = observation.BaselineValue - observation.CandidateValue
		observations = append(observations, observation)
	}
	return observations
}

func collocatedObservation(
	occupant PrefillOccupant, now float64, rt reTiming, chunk, admissionSteps, chunks float64, local bool,
) externalityObservation {
	observation := externalityObservation{
		Population:             "collocated_prefill",
		ResidentRequestID:      occupant.RequestID,
		Target:                 "first_token",
		Eligible:               occupant.RemPrefillTokens >= 0,
		RemainingDecodeSteps:   occupant.RemDecodeSteps,
		RemainingPrefillTokens: occupant.RemPrefillTokens,
		ArrivalUs:              occupant.ArrivalUs,
		TauTTFTUs:              occupant.Slo.TauTTFTUs,
		TauE2EUs:               occupant.Slo.TauE2EUs,
	}
	if occupant.RemPrefillTokens < 0 {
		observation.SkipReason = "remaining_prefill_tokens_unknown"
		return observation
	}
	remainingPrefillIterations := int64(math.Ceil(float64(occupant.RemPrefillTokens) / chunk))
	baselineFirstToken := rt.cBase(now, remainingPrefillIterations)
	var candidateFirstToken float64
	if local {
		candidateFirstToken = rt.cLocalAfter(now, remainingPrefillIterations, admissionSteps, chunks)
	} else {
		candidateFirstToken = rt.cDisagg(now, remainingPrefillIterations, admissionSteps)
	}
	baselineCompletion, candidateCompletion := baselineFirstToken, candidateFirstToken
	if occupant.RemDecodeSteps > 0 {
		total := remainingPrefillIterations + occupant.RemDecodeSteps
		baselineCompletion = rt.cBase(now, total)
		if local {
			candidateCompletion = rt.cLocalAfter(now, total, admissionSteps, chunks)
		} else {
			candidateCompletion = rt.cDisagg(now, total, admissionSteps)
		}
	}
	observation.BaselineTargetUs = baselineFirstToken
	observation.CandidateTargetUs = candidateFirstToken
	observation.BaselineFirstTokenUs = baselineFirstToken
	observation.CandidateFirstTokenUs = candidateFirstToken
	observation.BaselineCompletionUs = baselineCompletion
	observation.CandidateCompletionUs = candidateCompletion
	observation.BaselineValue = gCollocComposite(occupant, baselineFirstToken, baselineCompletion)
	observation.CandidateValue = gCollocComposite(occupant, candidateFirstToken, candidateCompletion)
	observation.Contribution = observation.BaselineValue - observation.CandidateValue
	return observation
}

func prefillValue(occupant PrefillOccupant, completionUs float64) float64 {
	deadline := float64(occupant.ArrivalUs) + occupant.Slo.TauTTFTUs
	scale := occupant.Slo.TauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return sigmoid((deadline - completionUs) / scale)
}

func eppObservationFields(
	event, requestID, placement, decodeEndpoint, prefillEndpoint string,
	now time.Time, elapsedUs int64,
) []any {
	return []any{
		"schemaVersion", causalObservationSchemaVersion,
		"event", event,
		"source", "epp",
		"requestID", requestID,
		"proxyRequestID", "",
		"wallTimeUnixNs", now.UnixNano(),
		"elapsedUs", elapsedUs,
		"placement", placement,
		"decodeEndpoint", decodeEndpoint,
		"prefillEndpoint", prefillEndpoint,
	}
}

func candidateObservationFields(e candidateEvaluation, nHatOut float64) []any {
	score := e.score
	prediction := newCandidatePrediction(score, nHatOut)
	prefillBatch, prefillKV, prefillSPf := 0, int64(0), int64(0)
	prefillUPf := 0.0
	prefillAP := 0
	prefillPrefixObserved := false
	if e.prefill != nil {
		prefillBatch = e.prefill.batchSize
		prefillKV = e.prefill.kvTokensInUse
		prefillSPf = e.prefill.residentPrefillTokens
		prefillUPf = e.prefill.residentPrefillAttentionUnits
		prefillAP = e.prefill.ap
		prefillPrefixObserved = e.prefill.prefixObserved
	}
	return []any{
		"decodeB", e.decode.batchSize,
		"decodeQueueDepth", e.decode.queueDepth,
		"decodeFreeKVBlocks", e.decode.freeKVBlocks,
		"decodeKV", e.decode.kvTokensInUse,
		"decodeSPf", e.decode.residentPrefillTokens,
		"decodeUPf", e.decode.residentPrefillAttentionUnits,
		"decodeUncachedPromptTokens", e.decode.ap,
		"decodePrefixObserved", e.decode.prefixObserved,
		"prefillB", prefillBatch,
		"prefillKV", prefillKV,
		"prefillSPf", prefillSPf,
		"prefillUPf", prefillUPf,
		"prefillUncachedPromptTokens", prefillAP,
		"prefillPrefixObserved", prefillPrefixObserved,
		"decodeAlphaUs", score.decodeIteration.alpha,
		"decodeReqUs", score.decodeIteration.decodeRequests,
		"decodeKVUs", score.decodeIteration.decodeKV,
		"decodePfLinearUs", score.decodeIteration.prefillLinear,
		"decodePfAttnUs", score.decodeIteration.prefillAttention,
		"decodeIterUs", score.decodeIteration.total,
		"firstDecodeAlphaUs", score.firstDecodeIteration.alpha,
		"firstDecodeReqUs", score.firstDecodeIteration.decodeRequests,
		"firstDecodeKVUs", score.firstDecodeIteration.decodeKV,
		"firstDecodePfLinearUs", score.firstDecodeIteration.prefillLinear,
		"firstDecodePfAttnUs", score.firstDecodeIteration.prefillAttention,
		"firstDecodeIterUs", score.firstDecodeIteration.total,
		"postJoinIterUs", score.firstDecodeIteration.total,
		"prefillAlphaUs", score.prefillIteration.alpha,
		"prefillReqUs", score.prefillIteration.decodeRequests,
		"prefillKVUs", score.prefillIteration.decodeKV,
		"prefillLinearUs", score.prefillIteration.prefillLinear,
		"prefillAttnUs", score.prefillIteration.prefillAttention,
		"prefillIterUs", score.prefillIteration.total,
		"prefillChunks", score.prefillChunks,
		"prefillWorkUs", score.prefillWorkUs,
		"decodeAdmissionUs", score.decodeAdmissionUs,
		"prefillAdmissionUs", score.prefillAdmissionUs,
		"prefillCompletionUs", prediction.PrefillCompletionUs,
		"transferUs", score.transferUs,
		"remoteLeadUs", prediction.RemoteLeadUs,
		"decodeJoinUs", score.decodeJoinUs,
		"projectedTTFTUs", score.projectedTTFTUs,
		"projectedE2EUs", prediction.ProjectedE2EUs,
		"externalityDecode", score.externalityBreakdown.decode,
		"externalityCollocPrefill", score.externalityBreakdown.collocPrefill,
		"externalityPrefillPool", score.externalityBreakdown.prefillPool,
		"externality", score.externality,
		"ownGood", score.ownGood,
		"netGoodCost", score.netGoodCost,
		"capacityTotal", score.capacityTotal,
		"J", score.total,
		"rolloutSubstituted", score.rolloutSubstituted,
		nHatOutKey, nHatOut,
	}
}

const nHatOutKey = "nHatOut"

func externalityObservationFields(observation externalityObservation, decisionNowUs float64, wallNow time.Time) []any {
	baselineWall := wallNow.UnixNano() + int64((observation.BaselineTargetUs-decisionNowUs)*1_000)
	candidateWall := wallNow.UnixNano() + int64((observation.CandidateTargetUs-decisionNowUs)*1_000)
	return []any{
		"externalityPopulation", observation.Population,
		"residentRequestID", observation.ResidentRequestID,
		"predictionTarget", observation.Target,
		"predictionEligible", observation.Eligible,
		"predictionSkipReason", observation.SkipReason,
		"remainingDecodeSteps", observation.RemainingDecodeSteps,
		"remainingPrefillTokens", observation.RemainingPrefillTokens,
		"residentArrivalUs", observation.ArrivalUs,
		"residentRealizedFirstTokenUs", observation.RealizedFirstTokenUs,
		"residentTauTTFTUs", observation.TauTTFTUs,
		"residentTauE2EUs", observation.TauE2EUs,
		"baselineTargetUs", observation.BaselineTargetUs,
		"candidateTargetUs", observation.CandidateTargetUs,
		"baselineTargetWallTimeUnixNs", baselineWall,
		"candidateTargetWallTimeUnixNs", candidateWall,
		"baselineFirstTokenUs", observation.BaselineFirstTokenUs,
		"candidateFirstTokenUs", observation.CandidateFirstTokenUs,
		"baselineCompletionUs", observation.BaselineCompletionUs,
		"candidateCompletionUs", observation.CandidateCompletionUs,
		"baselineValue", observation.BaselineValue,
		"candidateValue", observation.CandidateValue,
		"contribution", observation.Contribution,
	}
}
