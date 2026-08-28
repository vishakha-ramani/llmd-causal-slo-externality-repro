package causalsloexternality

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExternalityObservationsSumToCandidateBreakdown(t *testing.T) {
	cfg := testConfig()
	const now = 1_000_000.0

	decode := newTestEndpoint("ns/decode", h100, 4000)
	decode.runningDecode = []Resident{{
		RequestID: "decode-resident",
		Rem:       40, ArrivalUs: 0, FirstTokenUs: 400_000, TTFTSet: true,
		Slo: interactiveSLO, InputLen: 4000, StepsDone: 10,
	}}
	decode.runningPrefill = []PrefillOccupant{{
		RequestID:        "collocated-occupant",
		RemPrefillTokens: 2048, RemDecodeSteps: 20,
		ArrivalUs: 100_000, Slo: interactiveSLO, InputLen: 4096,
	}}
	prefill := newTestEndpoint("ns/prefill", h100, 4000)
	prefill.batchSize = 0
	prefill.runningPrefill = []PrefillOccupant{{
		RequestID:        "prefill-occupant",
		RemPrefillTokens: 4096, RemDecodeSteps: 20,
		ArrivalUs: 100_000, Slo: interactiveSLO, InputLen: 4096,
	}}

	score := cfg.scoreCandidate(4000, now, decode, prefill, interactiveSLO, 64)
	observations := cfg.externalityObservations(4000, now, decode, prefill, score)
	require.Len(t, observations, 3)

	var total float64
	seen := map[string]bool{}
	for _, observation := range observations {
		total += observation.Contribution
		seen[observation.ResidentRequestID] = true
		assert.GreaterOrEqual(t, observation.CandidateTargetUs, observation.BaselineTargetUs)
	}
	assert.InDelta(t, score.externalityBreakdown.total(), total, 1e-12)
	assert.True(t, seen["decode-resident"])
	assert.True(t, seen["collocated-occupant"])
	assert.True(t, seen["prefill-occupant"])

	localScore := cfg.scoreCandidate(4000, now, decode, nil, interactiveSLO, 64)
	localObservations := cfg.externalityObservations(4000, now, decode, nil, localScore)
	var localTotal float64
	for _, observation := range localObservations {
		localTotal += observation.Contribution
	}
	assert.InDelta(t, localScore.externalityBreakdown.total(), localTotal, 1e-12)
}

func TestCandidatePredictionExposesTimingIdentities(t *testing.T) {
	score := candidateScore{
		prefillIteration:     iterationTerms{total: 10},
		prefillAdmissionUs:   20,
		prefillChunks:        3,
		prefillWorkUs:        40,
		transferUs:           50,
		decodeJoinUs:         140,
		projectedTTFTUs:      150,
		firstDecodeIteration: iterationTerms{total: 10},
	}

	prediction := newCandidatePrediction(score, 7)
	assert.Equal(t, 90.0, prediction.PrefillCompletionUs)
	assert.Equal(t, 140.0, prediction.RemoteLeadUs)
	assert.Equal(t, 220.0, prediction.ProjectedE2EUs)
}
