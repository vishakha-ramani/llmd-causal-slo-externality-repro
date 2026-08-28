package causalsloexternality

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Coefficients from inputs/coeffs-llama70b-{h100,a100real}-tp4.json, the frozen campaign
// values. Tests assert against these rather than round numbers so a coefficient typo in config
// is caught here.
var (
	h100 = Coeffs{
		AlphaD: 16613.537554540144,
		AlphaP: 16617.85321583337,
		C0:     5.347316038602452,
		C1:     0.04761401141756073,
		CPf:    6.144687138665833,
		CAttn:  0.00010075247918809842,
	}
	a100 = Coeffs{
		AlphaD: 25563.819286862163,
		AlphaP: 25568.34953836831,
		C0:     5.945331876073271,
		C1:     0.07822809856114352,
		CPf:    9.794219053944662,
		CAttn:  0.00015977670754642687,
	}
)

// interactiveSLO is the treatment config's default triple: 1000 ms / 50 ms / 16000 ms.
var interactiveSLO = SLO{TauTTFTUs: 1_000_000, TauITLUs: 50_000, TauE2EUs: 16_000_000}

// The three workload triples from the bundle's configuration table, in microseconds.
var (
	reasoningSLO    = SLO{TauTTFTUs: 2_000_000, TauITLUs: 100_000, TauE2EUs: 802_000_000}
	deepResearchSLO = SLO{TauTTFTUs: 10_000_000, TauITLUs: 100_000, TauE2EUs: 40_000_000}
)

func testConfig() *Config {
	cfg := defaultConfig
	cfg.SLOTargets = map[string]SLO{
		"interactive":   interactiveSLO,
		"reasoning":     reasoningSLO,
		"deep_research": deepResearchSLO,
	}
	cfg.ActiveSLOTargets = "interactive"
	cfg.XferBaseUs = 50
	cfg.XferBandwidthGBps = 25
	cfg.KVBytesPerTokenPerGPU = 81920
	cfg.GPUTypeLabelKey = "pd-infocomm.io/gpu-type"
	cfg.Coefficients = map[string]Coeffs{"h100": h100, "a100": a100}
	return &cfg
}

func TestIterationTerms_SumEveryScheduledRequestMarginal(t *testing.T) {
	theta := Coeffs{
		AlphaD: 10,
		AlphaP: 10,
		C0:     2,
		C1:     3,
		CPf:    5,
		CAttn:  7,
	}

	// Two decode requests with contexts 11 and 13 contribute 2*C0 and
	// C1*(11+13). Two prefill grants contribute 5*(4+6) and
	// CAttn*(4*(20+4/2) + 6*(30+6/2)). Alpha is charged once.
	const (
		bDec = 2
		kv   = int64(24)
		sPf  = int64(10)
		uPf  = 4*(20+4.0/2.0) + 6*(30+6.0/2.0)
	)
	terms := theta.iterationTerms(theta.AlphaD, bDec, kv, sPf, uPf)

	assert.Equal(t, 10.0, terms.alpha)
	assert.Equal(t, 4.0, terms.decodeRequests)
	assert.Equal(t, 72.0, terms.decodeKV)
	assert.Equal(t, 50.0, terms.prefillLinear)
	assert.Equal(t, 7.0*uPf, terms.prefillAttention)
	assert.Equal(t, terms.alpha+terms.decodeRequests+terms.decodeKV+terms.prefillLinear+terms.prefillAttention, terms.total)
	assert.Equal(t, terms.total, theta.tIterDecode(bDec, kv, sPf, uPf))

	prefill := theta.iterationTerms(theta.AlphaP, 0, 0, sPf, uPf)
	assert.Zero(t, prefill.decodeRequests)
	assert.Zero(t, prefill.decodeKV)
	assert.Equal(t, prefill.total, theta.tIterPrefill(sPf, uPf))
}

func TestControlledIterationLawEligibility(t *testing.T) {
	cfg := testConfig()
	cfg.RequireNoResidentPrefill = true

	decode := newTestEndpoint("ns/decode", h100, 4000)
	decode.batchSize = 1
	decode.runningDecode = []Resident{{Rem: 10}}
	assert.True(t, cfg.decodeIterationLawObservable(decode))

	decode.runningPrefill = []PrefillOccupant{{RemPrefillTokens: 100}}
	assert.False(t, cfg.decodeIterationLawObservable(decode))
	decode.runningPrefill = nil
	decode.batchSize = 2
	assert.False(t, cfg.decodeIterationLawObservable(decode), "all-running is not decode-only when it disagrees with the decode shadow set")

	prefill := newTestEndpoint("ns/prefill", h100, 4000)
	prefill.batchSize = 0
	prefill.queueDepth = 0
	assert.True(t, cfg.prefillIterationLawObservable(prefill))
	prefill.batchSize = 1
	assert.False(t, cfg.prefillIterationLawObservable(prefill), "the controlled prefiller must be idle at the decision")
}

// ---------------------------------------------------------------------------
// The mechanism: causal gating
// ---------------------------------------------------------------------------

// A resident that completes before the arrival is admitted must contribute EXACTLY zero. This
// gating is what makes the term an externality rather than another load proxy, so it is asserted
// exactly rather than approximately.
func TestCausalGate_ResidentFinishingInsideAdmissionWindowContributesExactlyZero(t *testing.T) {
	rt := newReTiming(h100, 8, 16384, 0, 0, 2048, 4000).withExactOverlap(h100, 4000, 4000)
	const now = 1_000_000

	// admissionSteps well beyond the resident's remaining steps: it departs untouched.
	const rem int64 = 3
	const admissionSteps = 10.0
	const nChunks = 2.0

	cb := rt.cBase(now, rem)
	cp := rt.cLocalAfter(now, rem, admissionSteps, nChunks)
	assert.Equal(t, cb, cp, "a resident inside the admission window must be undisturbed")

	r := Resident{Rem: rem, ArrivalUs: 0, FirstTokenUs: 500_000, TTFTSet: true, Slo: interactiveSLO}
	assert.Zero(t, residentContribution(r, cb, cp), "charge must be exactly zero, not merely small")
}

// The same resident outside the admission window must be charged strictly more than zero, or the
// gate would be indistinguishable from a term that is always zero.
func TestCausalGate_ResidentOutsideWindowIsChargedStrictlyPositive(t *testing.T) {
	rt := newReTiming(h100, 8, 16384, 0, 0, 2048, 4000).withExactOverlap(h100, 4000, 4000)
	const now = 1_000_000

	const rem int64 = 40
	cb := rt.cBase(now, rem)
	cp := rt.cLocalAfter(now, rem, 0, 2)
	require.Greater(t, cp, cb, "placement must delay a resident it overlaps")

	r := Resident{Rem: rem, ArrivalUs: 0, FirstTokenUs: 500_000, TTFTSet: true, Slo: interactiveSLO}
	assert.Positive(t, residentContribution(r, cb, cp))
}

// A resident with no first token yet is skipped by the decode-side kernel. Such a request is
// priced by the collocated-occupant term instead, whose TTFT is still live.
func TestGDecodeComposite_NoFirstTokenContributesZero(t *testing.T) {
	r := Resident{Rem: 10, ArrivalUs: 0, TTFTSet: false, Slo: interactiveSLO}
	assert.Zero(t, gDecodeComposite(r, 5_000_000))
	assert.Zero(t, residentContribution(r, 1_000, 9_999_999))
}

// The asymmetry between local and disaggregated placement is the mechanism's lever: a remote
// prefill does not inflate the decode endpoint's iteration time, so it delays residents strictly
// less than the same placement made locally.
func TestDisaggDelaysResidentsLessThanLocal(t *testing.T) {
	rt := newReTiming(h100, 8, 16384, 0, 0, 2048, 4000).withExactOverlap(h100, 4000, 4000)
	const now, rem, steps = 1_000_000, int64(40), 2.0

	local := rt.cLocalAfter(now, rem, steps, 2)
	disagg := rt.cDisagg(now, rem, steps)
	assert.Less(t, disagg, local, "local placement must cost residents more than remote prefill")
}

// ---------------------------------------------------------------------------
// The flattening failure mode
// ---------------------------------------------------------------------------

// All-zero thresholds do not loosen the policy, they flatten it: the value becomes identically 1,
// every charge is zero, and the argmin degenerates to enumeration order. Config validation must
// reject that, but the kernel's behaviour is asserted here so the reason is documented in code.
func TestSLOCompositeValue_ZeroThresholdsFlattenToOne(t *testing.T) {
	zero := SLO{}
	assert.Equal(t, 1.0, sloCompositeValue(zero, 0, 0))
	assert.Equal(t, 1.0, sloCompositeValue(zero, 9e9, 9e9), "no threshold means no discrimination")

	r := Resident{Rem: 10, TTFTSet: true, Slo: zero}
	assert.Zero(t, residentContribution(r, 1_000, 9_999_999), "flat value gives a zero charge at any delay")
}

// A single disabled dimension contributes exactly 1 rather than 0, so disabling TTFT must not
// zero the whole value.
func TestSLOCompositeValue_DisabledDimensionContributesOne(t *testing.T) {
	e2eOnly := SLO{TauE2EUs: 16_000_000}
	both := SLO{TauTTFTUs: 1_000_000, TauE2EUs: 16_000_000}
	assert.Equal(t, sigmoid(1.0), sloCompositeValue(e2eOnly, 0, 0))
	assert.Less(t, sloCompositeValue(both, 0, 0), sloCompositeValue(e2eOnly, 0, 0))
}

// ---------------------------------------------------------------------------
// The score identity the ablation cohort gates on
// ---------------------------------------------------------------------------

// With capacity disabled the campaign asserts score == V*(externality - ownGood) exactly. V is a
// common positive factor that cannot reorder the argmin, but folding it away would make that
// check unreproducible, so the identity is asserted here at the configured V.
func TestScoreIdentity_TotalIsVTimesNetGoodCost(t *testing.T) {
	cfg := testConfig()
	require.Equal(t, 8.0, cfg.V, "the registered arm runs V=8")

	dec := newTestEndpoint("ns/decode-a", h100, 4000)
	dec.runningDecode = []Resident{
		{Rem: 30, ArrivalUs: 0, FirstTokenUs: 800_000, TTFTSet: true, Slo: interactiveSLO, InputLen: 4000, StepsDone: 20},
	}
	pre := newTestEndpoint("ns/prefill-a", h100, 4000)

	for _, tc := range []struct {
		name string
		pre  *endpointState
	}{{"local", nil}, {"disagg", pre}} {
		t.Run(tc.name, func(t *testing.T) {
			s := cfg.scoreCandidate(4000, 1_000_000, dec, tc.pre, interactiveSLO, 300)
			assert.Equal(t, s.externality-s.ownGood, s.netGoodCost)
			assert.Zero(t, s.capacityTotal, "capacity terms must be exactly zero in every arm")
			assert.InDelta(t, cfg.V*s.netGoodCost, s.total, 1e-12)
		})
	}
}

// J is signed and unbounded, which is why this policy cannot be expressed as a Scorer: every
// scorer contribution is clamped to [0,1], and clamping would collapse every candidate with
// J <= 0 to one score. Assert the objective really does leave that range.
func TestScore_IsSignedAndCanExceedUnitRange(t *testing.T) {
	cfg := testConfig()

	// Residents sitting just inside their E2E deadline, where an extra co-scheduled prefill
	// flips them to a miss. The externality dominates and J exceeds 1.
	high := cfg.scoreCandidate(40000, 1_000_000, residentsWithRemainingSteps(800), nil, interactiveSLO, 300)

	// The same fleet state with residents far inside their deadline: nothing recoverable is at
	// risk, the arrival's own value dominates, and J is negative.
	low := cfg.scoreCandidate(40000, 1_000_000, residentsWithRemainingSteps(100), nil, interactiveSLO, 300)

	assert.Negative(t, low.total, "a placement that destroys little must score negative, which a [0,1] clamp would erase")
	assert.Greater(t, high.total, 1.0, "a harmful placement must exceed 1, which a clamp would also erase")
	assert.Greater(t, high.total, low.total)
}

// ---------------------------------------------------------------------------
// The causal claim: the signal is a property of resident deadlines, not of load
// ---------------------------------------------------------------------------

// THE CENTRAL CLAIM. Two endpoints with IDENTICAL queue depth, batch size, KV occupancy and
// prefix residency must score differently when their residents sit at different distances from
// their deadlines. A load-shaped signal cannot see that difference, because it is a property of
// the residents' deadlines rather than of the endpoint's load. If this test fails, the port has
// become a load proxy no matter what else still passes.
func TestExternality_DiscriminatesBetweenEndpointsWithIdenticalLoad(t *testing.T) {
	cfg := testConfig()

	// Every observable a load signal reads is equal between these two; only the residents'
	// arrival times, and hence their remaining E2E slack, differ.
	atRisk := residentsAtDeadlineOffset(-9_000_000)  // about 1 ms of slack left
	doomed := residentsAtDeadlineOffset(-90_000_000) // about 82 s past deadline

	for _, field := range []struct {
		name string
		a, b any
	}{
		{"batchSize", atRisk.batchSize, doomed.batchSize},
		{"queueDepth", atRisk.queueDepth, doomed.queueDepth},
		{"kvTokensInUse", atRisk.kvTokensInUse, doomed.kvTokensInUse},
		{"freeKVBlocks", atRisk.freeKVBlocks, doomed.freeKVBlocks},
		{"residentCount", len(atRisk.runningDecode), len(doomed.runningDecode)},
		{"uncachedPromptTokens", atRisk.ap, doomed.ap},
	} {
		require.Equal(t, field.a, field.b, "%s must be identical for this test to isolate deadline proximity", field.name)
	}

	risky := cfg.externalityLocal(40000, 1_000_000, atRisk, 17436).total()
	cheap := cfg.externalityLocal(40000, 1_000_000, doomed, 17436).total()

	assert.Greater(t, risky/cheap, 10.0,
		"residents at their deadline must be priced at least an order of magnitude above residents already lost")
}

// The externality peaks at the deadline and falls away on BOTH sides. The inflicted delay is held
// identical across these cases, so the only thing varying is where each resident sits relative to
// its deadline.
//
// Note the peak is broad rather than sharp, and residents somewhat past their deadline are still
// priced highly: the charge only collapses deep in the tail, where the sigmoid has saturated.
func TestExternality_PeaksAtTheDeadlineAndFallsOffBothSides(t *testing.T) {
	cfg := testConfig()
	at := func(offset int64) float64 {
		return cfg.externalityLocal(40000, 1_000_000, residentsAtDeadlineOffset(offset), 17436).total()
	}

	// Measured at these parameters with the inflicted delay held at 1,050,245 us: 0.401 at +8 s of
	// slack, 0.423 at +2 s, 0.423 at the deadline, 0.407 at -6 s, 0.268 at -22 s, 0.0097 at -82 s.
	// THE PEAK IS BROAD, NOT SHARP. Pricing stays within a few percent of maximum until roughly six
	// seconds past the deadline and only collapses tens of seconds out, so the bundle's "already
	// doomed costs nothing recoverable" does not apply just past the deadline. Do not relax these
	// assertions into a sharp-peak story.
	comfortable := at(0)         // about 8 s of slack
	atDeadline := at(-9_000_000) // about 1 ms of slack
	wellPast := at(-30_000_000)  // about 22 s past
	hopeless := at(-90_000_000)  // about 82 s past

	assert.Greater(t, atDeadline, comfortable, "slack that can still be saved must be priced above ample slack")
	assert.Greater(t, atDeadline, wellPast, "the charge must fall off past the deadline")
	assert.Greater(t, wellPast, hopeless, "and collapse once nothing is recoverable")
	assert.Less(t, hopeless/atDeadline, 0.05, "a hopelessly late resident must be nearly free to disturb")
}

// ---------------------------------------------------------------------------
// Determinism, given that the framework hands the picker a randomly ordered slice
// ---------------------------------------------------------------------------

// The slice a picker receives is built by ranging a Go map, so its order is randomized per
// iteration. The argmin uses a strict-with-epsilon comparison that keeps the incumbent on a
// near-tie, which is only well-defined if the incumbent is deterministic. Shuffling the input
// must not change the outcome.
func TestDecide_IsDeterministicUnderShuffledInput(t *testing.T) {
	cfg := testConfig()
	build := func() ([]*endpointState, []*endpointState) {
		ds := make([]*endpointState, 0, 3)
		ps := make([]*endpointState, 0, 2)
		for _, id := range []string{"ns/d-c", "ns/d-a", "ns/d-b"} {
			ds = append(ds, newTestEndpoint(id, h100, 4000))
		}
		for _, id := range []string{"ns/p-b", "ns/p-a"} {
			p := newTestEndpoint(id, h100, 4000)
			ps = append(ps, p)
		}
		return ds, ps
	}

	ds, ps := build()
	want := cfg.decide(4000, 1_000_000, ds, ps, "", interactiveSLO, 300)

	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 32; i++ {
		ds, ps := build()
		rng.Shuffle(len(ds), func(a, b int) { ds[a], ds[b] = ds[b], ds[a] })
		rng.Shuffle(len(ps), func(a, b int) { ps[a], ps[b] = ps[b], ps[a] })
		got := cfg.decide(4000, 1_000_000, ds, ps, "", interactiveSLO, 300)
		require.Equal(t, want, got, "shuffled input changed the decision on iteration %d", i)
	}
}

// On an exact tie the local candidate must win, because it is enumerated before that decoder's
// disaggregated candidates and the comparison is strict.
func TestDecide_LocalWinsExactTieAgainstRemote(t *testing.T) {
	cfg := testConfig()
	// Zero transfer cost and an unobserved prefill pool make local and remote score alike.
	cfg.CXferSizeAware = false

	dec := newTestEndpoint("ns/d-a", h100, 0)
	pre := newTestEndpoint("ns/p-a", h100, 0)

	got := cfg.decide(0, 1_000_000, []*endpointState{dec}, []*endpointState{pre}, "", interactiveSLO, 300)
	assert.False(t, got.disaggregate, "an exact tie must keep the locally-enumerated candidate")
	assert.Equal(t, "ns/d-a", got.decodeID)
}

// The preferred endpoint is a TIE-BREAK ONLY device: moving it to the front lets it win an exact
// tie without narrowing the candidate set.
func TestDecide_PreferredEndpointWinsOnlyExactTies(t *testing.T) {
	cfg := testConfig()

	// Three identical decoders: whichever is enumerated first wins the tie.
	mk := func() []*endpointState {
		return []*endpointState{
			newTestEndpoint("ns/d-a", h100, 100),
			newTestEndpoint("ns/d-b", h100, 100),
			newTestEndpoint("ns/d-c", h100, 100),
		}
	}
	assert.Equal(t, "ns/d-a", cfg.decide(100, 1e6, mk(), nil, "", interactiveSLO, 300).decodeID)
	assert.Equal(t, "ns/d-c", cfg.decide(100, 1e6, mk(), nil, "ns/d-c", interactiveSLO, 300).decodeID,
		"the preferred endpoint should win an exact tie")

	// But a preference must not override a strictly better candidate. Make d-a strictly better
	// by giving it no residents while the preferred one carries a costly resident set.
	eps := mk()
	for i := 0; i < 20; i++ {
		eps[2].runningDecode = append(eps[2].runningDecode, Resident{
			Rem: 300, ArrivalUs: 0, FirstTokenUs: 300_000, TTFTSet: true,
			Slo: interactiveSLO, InputLen: 4000, StepsDone: 10,
		})
	}
	got := cfg.decide(100, 1e6, eps, nil, "ns/d-c", interactiveSLO, 300)
	assert.Equal(t, "ns/d-a", got.decodeID, "a preference must never beat a strictly better candidate")
}

// The joint argmin must consider the full cross product, not just the first decoder. The
// decomposed switch is what truncates it, and that is a different algorithm.
func TestDecide_DecomposedTruncatesCandidateSetToOneDecoder(t *testing.T) {
	cfg := testConfig()

	// d-a is heavily loaded; d-b is idle. The joint argmin must pick d-b.
	loaded := newTestEndpoint("ns/d-a", h100, 4000)
	loaded.batchSize = 64
	loaded.kvTokensInUse = 400_000
	for i := 0; i < 30; i++ {
		loaded.runningDecode = append(loaded.runningDecode, Resident{
			Rem: 250, ArrivalUs: 0, FirstTokenUs: 300_000, TTFTSet: true,
			Slo: interactiveSLO, InputLen: 4000, StepsDone: 20,
		})
	}
	idle := newTestEndpoint("ns/d-b", h100, 4000)

	joint := cfg.decide(4000, 15e6, []*endpointState{loaded, idle}, nil, "", interactiveSLO, 300)
	require.Equal(t, "ns/d-b", joint.decodeID, "the joint argmin must be free to move off a loaded decoder")

	cfg.Decomposed = true
	first := cfg.decide(4000, 15e6, []*endpointState{loaded, idle}, nil, "", interactiveSLO, 300)
	assert.Equal(t, "ns/d-a", first.decodeID,
		"decomposed truncates to the first enumerated decoder, which is the measured decode-first ablation")
}

// ---------------------------------------------------------------------------
// Hardware heterogeneity
// ---------------------------------------------------------------------------

// The discriminating term is the per-iteration intercept, present on every iteration regardless
// of KV state, which is why a defaulted GPU label is wrong on every decision rather than only
// under load. Values from config.md section 3.
func TestTIterDecode_HeterogeneityRidesTheIntercept(t *testing.T) {
	const bDec, kv = 8, int64(16384)
	hot := h100.tIterDecode(bDec, kv, 0, 0)
	cold := a100.tIterDecode(bDec, kv, 0, 0)

	assert.InDelta(t, 17436.0, hot, 1.0)
	assert.InDelta(t, 26893.0, cold, 1.0)

	gap := cold - hot
	interceptGap := a100.AlphaD - h100.AlphaD
	assert.InDelta(t, 1.54, cold/hot, 0.01)
	assert.Greater(t, interceptGap/gap, 0.94, "the intercept must account for nearly all of the gap")

	kvGap := (a100.C1 - h100.C1) * float64(kv)
	assert.Less(t, kvGap, 550.0, "the C1 spread contributes only a few hundred microseconds")
}

// The drain rates and the marginal-work term are uncalled by construction rather than by omission:
// they belong to the capacity account and the historical drift objective, neither of which this arm
// carries. They are tested so the stated algebra is verified rather than merely preserved, and so a
// future port that re-enables either term inherits checked code.
func TestUncalledLawTerms_AlgebraIsStillCorrect(t *testing.T) {
	// Live drain rates are mu = 1 - alpha/T_iter.
	tIterD := h100.tIterDecode(8, 16384, 0, 0)
	assert.InDelta(t, 1.0-h100.AlphaD/tIterD, h100.muDecode(8, 16384, 0, 0), 1e-12)
	tIterP := h100.tIterPrefill(2048, 0)
	assert.InDelta(t, 1.0-h100.AlphaP/tIterP, h100.muPrefill(2048, 0), 1e-12)

	// The nominal decode rate evaluates at the SLO-critical batch, where T_iter equals tau_itl.
	assert.InDelta(t, 1.0-h100.AlphaD/50_000.0, h100.muDNom(50_000), 1e-12)
	// The nominal prefill rate evaluates at the nominal operating chunk.
	assert.InDelta(t, 1.0-h100.AlphaP/(h100.AlphaP+h100.CPf*2048), h100.muPNom(2048), 1e-12)

	// Every rate is clamped so a predictor denominator cannot collapse. A tau_itl at or below the
	// intercept is the degenerate point the floor exists to protect, and config validation rejects
	// it separately.
	assert.Equal(t, minMu, h100.muDNom(h100.AlphaD))
	assert.Equal(t, minMu, h100.muDNom(h100.AlphaD/2))
	assert.Equal(t, 1e-3, minMu, "the floor is 1e-3, not 1e-6; a 1000x error here is latent")

	// The marginal decode charge at a context length.
	assert.InDelta(t, h100.C0+h100.C1*16384, h100.deltaBarDecode(16384), 1e-12)
}

// An absent or unrecognised GPU label must be rejected, never defaulted.
func TestCoeffsFor_RejectsUnknownAndMissingGPUType(t *testing.T) {
	cfg := testConfig()

	got, err := cfg.coeffsFor("h100")
	require.NoError(t, err)
	assert.Equal(t, h100, got)

	_, err = cfg.coeffsFor("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "refusing to default")

	_, err = cfg.coeffsFor("NVIDIA-L40S")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no coefficients configured")
	assert.Contains(t, err.Error(), "a100", "the error should name the configured types")
}

// ---------------------------------------------------------------------------
// Unit traps
// ---------------------------------------------------------------------------

// The KV usage metric is a FRACTION in [0,1] despite its name. Dividing by 100 would under-count
// KV by 100x and collapse the term that carries hardware heterogeneity.
func TestKvTokensInUse_TreatsUsageAsFractionNotPercent(t *testing.T) {
	// Half of 1000 blocks of 16 tokens is 8000 tokens.
	assert.Equal(t, int64(8000), kvTokensInUse(0.5, 16, 1000))
	assert.Equal(t, int64(16000), kvTokensInUse(1.0, 16, 1000))
	assert.Equal(t, int64(0), kvTokensInUse(0.0, 16, 1000))
}

// The derived free-block count is a floor, never an over-count, and never negative.
func TestFreeKVBlocks_IsAFloorAndNeverNegative(t *testing.T) {
	assert.Equal(t, int64(500), freeKVBlocks(0.5, 1000))
	assert.Equal(t, int64(0), freeKVBlocks(1.0, 1000))
	assert.Equal(t, int64(0), freeKVBlocks(1.5, 1000), "an over-unity scrape must not produce a negative count")
	// Truncation makes the result a floor rather than a rounded estimate.
	assert.Equal(t, int64(6), freeKVBlocks(0.933, 100))
}

// The chunk terms must return exactly zero work for a fully-cached prompt, which is what lets a
// fully-cached request still score on every candidate instead of short-circuiting the argmin.
func TestChunkTerms_FullyCachedPromptYieldsNoPrefillWork(t *testing.T) {
	cfg := testConfig()
	n, d := cfg.chunkTerms(h100, 0)
	assert.Zero(t, n)
	assert.Zero(t, d)

	n, d = cfg.chunkTerms(h100, -256)
	assert.Zero(t, n, "a negative uncached count means fully cached, not negative work")
	assert.Zero(t, d)

	assert.Zero(t, h100.Wp(0, 4000))
	assert.Zero(t, h100.Wd(4000, 0))
}

// The marginal-prefill charge must EXCLUDE baseline iteration time; that exclusion is what makes
// the charge marginal.
func TestPrefillMarginalWork_ExcludesBaselineAndSaturatesAtAp(t *testing.T) {
	const ap, ar, chunk = 4000.0, 4000.0, 2048.0

	assert.Zero(t, prefillMarginalWork(h100.CPf, h100.CAttn, 0, ar, chunk, 3), "no uncached tokens, no charge")
	assert.Zero(t, prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 0), "no overlapping iterations, no charge")

	one := prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 1)
	two := prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 2)
	assert.Positive(t, one)
	assert.Greater(t, two, one)

	// Beyond ceil(ap/chunk) iterations the whole prompt is processed, so the charge saturates.
	assert.Equal(t, two, prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 99))

	// At ap == ar the closed form is CPf*ap + 0.5*CAttn*ap^2.
	full := prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 99)
	assert.InDelta(t, h100.CPf*ap+0.5*h100.CAttn*ap*ap, full, 1e-6)
}

// Wp's second factor is (ar - ap/2), the trajectory integral over the prefill, not ar.
func TestWp_MatchesClosedFormAtFullPrompt(t *testing.T) {
	assert.InDelta(t, h100.CPf*4000+0.5*h100.CAttn*4000*4000, h100.Wp(4000, 4000), 1e-9)
	// A cached prefix reduces uncached tokens but raises the attention prefix.
	assert.Less(t, h100.Wp(1000, 4000), h100.Wp(4000, 4000))
}

// Wd is the exact discrete sum over decode steps, not o*(C0 + C1*ar).
func TestWd_MatchesDiscreteSum(t *testing.T) {
	const ar, o = 4000, 50
	var want float64
	for k := 0; k < o; k++ {
		want += h100.C0 + h100.C1*float64(ar+k)
	}
	assert.InDelta(t, want, h100.Wd(ar, o), 1e-9)
	assert.Zero(t, h100.Wd(ar, -1), "a non-positive output length returns exactly zero")
}

// ---------------------------------------------------------------------------
// Transfer cost
// ---------------------------------------------------------------------------

func TestCXferUs_SizeAwareAndFlatPaths(t *testing.T) {
	cfg := testConfig()

	// 4000 tokens at block size 16 is 250 blocks, 4000 block-aligned tokens,
	// 4000*81920 bytes over 25 GB/s expressed as 25000 bytes/us.
	want := 50.0 + (250*16*81920.0)/(25*1000.0)
	assert.InDelta(t, want, cfg.cXferUs(4000), 1e-9)

	// Larger prompts cost strictly more; that size awareness is what the arm selects.
	assert.Greater(t, cfg.cXferUs(40000), cfg.cXferUs(4000))

	// A non-positive bandwidth degrades to the base latency rather than dividing by zero.
	cfg.XferBandwidthGBps = 0
	assert.Equal(t, 50.0, cfg.cXferUs(4000))

	cfg = testConfig()
	cfg.CXferSizeAware = false
	assert.Zero(t, cfg.cXferUs(4000))
}

// reqKVNeed uses integer ceiling arithmetic in engine blocks.
func TestReqKVNeed_IntegerCeiling(t *testing.T) {
	cfg := testConfig()
	assert.Equal(t, int64(0), cfg.reqKVNeed(0))
	assert.Equal(t, int64(1), cfg.reqKVNeed(1))
	assert.Equal(t, int64(1), cfg.reqKVNeed(16))
	assert.Equal(t, int64(2), cfg.reqKVNeed(17))
	assert.Equal(t, int64(250), cfg.reqKVNeed(4000))

	cfg.BlockSize = 0
	assert.Equal(t, int64(0), cfg.reqKVNeed(4000))
}

// ---------------------------------------------------------------------------
// Admission estimator (the substituted one)
// ---------------------------------------------------------------------------

func TestRollforward_FastPathFloorsAtOneIteration(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize: 4, MaxBatchSize: 256, FreeKVBlocks: 1000, ReqKVNeed: 250, TIter: 17000,
	}
	// A free slot and enough free KV means no wait beyond the one-iteration floor: even with a
	// slot, the request waits for the current step to finish.
	assert.Equal(t, 17000.0, rollforwardTAdm(ctx))
}

func TestRollforward_NeedsQueueDepthPlusOneSlots(t *testing.T) {
	// Batch is full, so the fast path is closed and departures must free QueueDepth+1 slots.
	// Two requests are queued ahead, so ours needs the third departure.
	ctx := AdmissionContext{
		BatchSize: 3, MaxBatchSize: 3, FreeKVBlocks: 10_000, ReqKVNeed: 1, TIter: 1000,
		QueueDepth: 2,
		Running: []ResidentKV{
			{Rem: 5, KVBlocks: 1}, {Rem: 1, KVBlocks: 1}, {Rem: 3, KVBlocks: 1},
		},
	}
	// Sorted ascending the departures are 1, 3, 5; the third frees the slot ours takes.
	assert.Equal(t, 5.0*1000, rollforwardTAdm(ctx))

	// With nothing queued ahead, the earliest departure suffices.
	ctx.QueueDepth = 0
	assert.Equal(t, 1.0*1000, rollforwardTAdm(ctx))
}

func TestRollforward_KVShortfallDelaysBeyondSlotAvailability(t *testing.T) {
	// A slot frees immediately but the KV need is not met until later departures return blocks.
	ctx := AdmissionContext{
		BatchSize: 2, MaxBatchSize: 2, FreeKVBlocks: 0, ReqKVNeed: 100, TIter: 1000,
		QueueDepth: 0,
		Running:    []ResidentKV{{Rem: 1, KVBlocks: 10}, {Rem: 7, KVBlocks: 95}},
	}
	assert.Equal(t, 7.0*1000, rollforwardTAdm(ctx), "admission waits for KV, not just for a slot")
}

// The wave form is the branch that carries the substitution's bias: it is taken when departures
// cannot free QueueDepth+1 slots, and it UNDERSTATES deep-queue delay.
func TestRollforward_FallsBackToWaveFormOnDeepQueue(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize: 2, MaxBatchSize: 2, FreeKVBlocks: 10_000, ReqKVNeed: 1, TIter: 1000,
		QueueDepth: 9, RemainingStepsEst: 4,
		Running: []ResidentKV{{Rem: 1, KVBlocks: 1}, {Rem: 2, KVBlocks: 1}},
	}
	// ceil((9+1)/2) = 5 waves of RemainingStepsEst iterations each.
	assert.Equal(t, 5.0*4*1000, rollforwardTAdm(ctx))
}

// An unknown per-resident remaining must fall back to the aggregate estimate by TRUNCATION, not
// rounding, and never below one step.
func TestRollforward_CensoredRemainingTruncatesToEstimate(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 10_000, ReqKVNeed: 1, TIter: 1000,
		QueueDepth: 0, RemainingStepsEst: 3.9,
		Running: []ResidentKV{{Rem: -1, KVBlocks: 1}},
	}
	assert.Equal(t, 3.0*1000, rollforwardTAdm(ctx), "3.9 must truncate to 3, not round to 4")

	ctx.RemainingStepsEst = 0.4
	assert.Equal(t, 1000.0, rollforwardTAdm(ctx), "a sub-one estimate floors at one step")
}

// The departure walk must be stable so the same fleet state scores identically across runs.
func TestRollforward_IsStableUnderInputPermutation(t *testing.T) {
	base := []ResidentKV{{Rem: 3, KVBlocks: 7}, {Rem: 1, KVBlocks: 2}, {Rem: 3, KVBlocks: 1}, {Rem: 2, KVBlocks: 5}}
	mk := func(order []int) AdmissionContext {
		running := make([]ResidentKV, 0, len(order))
		for _, i := range order {
			running = append(running, base[i])
		}
		return AdmissionContext{
			BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 9, TIter: 1000,
			QueueDepth: 1, RemainingStepsEst: 2, Running: running,
		}
	}
	want := rollforwardTAdm(mk([]int{0, 1, 2, 3}))
	for _, order := range [][]int{{3, 2, 1, 0}, {1, 3, 0, 2}, {2, 0, 3, 1}} {
		assert.Equal(t, want, rollforwardTAdm(mk(order)))
	}
}

// A zero MaxBatchSize is the default that silently converts the substitution's tail bias into an
// always-on one: the fast path becomes unreachable and the free-slot count starts negative, so
// every endpoint takes the wave form. Config validation rejects it; this records why.
func TestRollforward_ZeroMaxBatchSizeForcesWaveForm(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize: 4, MaxBatchSize: 0, FreeKVBlocks: 10_000, ReqKVNeed: 1, TIter: 1000,
		QueueDepth: 0, RemainingStepsEst: 5,
		Running: []ResidentKV{{Rem: 1, KVBlocks: 1}},
	}
	assert.Equal(t, 1.0*5*1000, rollforwardTAdm(ctx), "the departure walk cannot satisfy a negative slot count")
}

// ---------------------------------------------------------------------------
// Censored remaining-steps estimates
// ---------------------------------------------------------------------------

// The estimate is censored, not a plain mean: a plain mean goes negative once residents outlive
// the class estimate.
func TestDecodeRemStepsEst_CensorsByLongestElapsed(t *testing.T) {
	assert.Equal(t, 1.0, decodeRemStepsEst(nil, 300), "an empty endpoint returns one, not zero")

	// Every resident has already produced more than the class estimate, so a plain mean would be
	// negative. Censoring floors each contribution at one.
	residents := []Resident{{StepsDone: 500}, {StepsDone: 600}}
	got := decodeRemStepsEst(residents, 300)
	assert.Positive(t, got)
	// nHatEff = max(300, 600) = 600, contributions max(600-500,1)=100 and max(600-600,1)=1.
	assert.InDelta(t, (100.0+1.0)/2.0, got, 1e-9)
}

func TestResidentRem_FloorsAtOneAndCensors(t *testing.T) {
	assert.Equal(t, int64(280), residentRem(300, 20))
	assert.Equal(t, int64(1), residentRem(300, 300), "a resident at the estimate is still decoding")
	assert.Equal(t, int64(1), residentRem(300, 900), "an outlier must not produce a negative remaining")
}

func TestOccupantRemDecodeSteps_FloorsAtOne(t *testing.T) {
	assert.Equal(t, int64(300), occupantRemDecodeSteps(300))
	assert.Equal(t, int64(1), occupantRemDecodeSteps(0))
}

// The prefilling-tokens estimate is capped per occupant and in total, because summing whole
// remaining prompts over-estimates by roughly the chunk count.
func TestResidentPrefillTokens_CapsPerOccupantAndInTotal(t *testing.T) {
	cfg := testConfig()
	require.Equal(t, 2048, cfg.ChunkTokens)

	assert.Equal(t, int64(0), cfg.residentPrefillTokens(nil))
	assert.Equal(t, int64(500), cfg.residentPrefillTokens([]PrefillOccupant{{RemPrefillTokens: 500}}))
	// One occupant with a 45k backlog is capped at a single chunk, not summed whole.
	assert.Equal(t, int64(2048), cfg.residentPrefillTokens([]PrefillOccupant{{RemPrefillTokens: 45000}}))
	// The total is capped at the engine's batched-token budget.
	assert.Equal(t, int64(2048), cfg.residentPrefillTokens([]PrefillOccupant{
		{RemPrefillTokens: 45000}, {RemPrefillTokens: 45000}, {RemPrefillTokens: 45000},
	}))
	// Non-positive backlogs are skipped rather than subtracted.
	assert.Equal(t, int64(300), cfg.residentPrefillTokens([]PrefillOccupant{
		{RemPrefillTokens: 300}, {RemPrefillTokens: 0}, {RemPrefillTokens: -5},
	}))
}

// ---------------------------------------------------------------------------
// Population sums
// ---------------------------------------------------------------------------

// Residents whose remaining steps are unknown must be skipped, not guessed at.
func TestSums_SkipUnknownResidents(t *testing.T) {
	rt := newReTiming(h100, 8, 16384, 0, 0, 2048, 4000).withExactOverlap(h100, 4000, 4000)
	residents := []Resident{
		{Rem: -1, TTFTSet: true, ArrivalUs: 0, FirstTokenUs: 100, Slo: interactiveSLO},
	}
	assert.Zero(t, decodeLocalSum(1e6, residents, rt, 2, 0))
	assert.Zero(t, decodeDisaggSum(1e6, residents, rt, 0))

	occupants := []PrefillOccupant{{RemPrefillTokens: -1, Slo: interactiveSLO}}
	assert.Zero(t, collocLocalSum(1e6, occupants, rt, 2048, 2, 0))
	assert.Zero(t, collocDisaggSum(1e6, occupants, rt, 2048, 0))
}

// The prefill pool's charge must not price an occupant for the arrival's whole prompt when it has
// only one iteration left; that over-charge is the naive form this arm replaces.
func TestPrefillPoolSum_ChargesOnlyOverlappingChunks(t *testing.T) {
	tIterP := h100.tIterPrefill(0, 0)
	occupantAlmostDone := []PrefillOccupant{{RemPrefillTokens: 1, ArrivalUs: 0, Slo: interactiveSLO}}
	occupantFarFromDone := []PrefillOccupant{{RemPrefillTokens: 40000, ArrivalUs: 0, Slo: interactiveSLO}}

	near := prefillPoolSum(1e6, occupantAlmostDone, tIterP, 2048, 0, 40000, 40000, h100)
	far := prefillPoolSum(1e6, occupantFarFromDone, tIterP, 2048, 0, 40000, 40000, h100)
	assert.Less(t, near, far, "an occupant near its first token must be charged less")
	assert.Positive(t, far)
}

// An occupant admitted after the arrival's chunks have run is not charged at all: the same causal
// gate as the decode side.
func TestPrefillPoolSum_AdmissionWindowGatesTheCharge(t *testing.T) {
	tIterP := h100.tIterPrefill(0, 0)
	occupants := []PrefillOccupant{{RemPrefillTokens: 2048, ArrivalUs: 0, Slo: interactiveSLO}}

	charged := prefillPoolSum(1e6, occupants, tIterP, 2048, 0, 40000, 40000, h100)
	gated := prefillPoolSum(1e6, occupants, tIterP, 2048, 100, 40000, 40000, h100)
	assert.Positive(t, charged)
	assert.Zero(t, gated, "an occupant whose remaining iterations fall inside the admission window is free")
}

// ---------------------------------------------------------------------------
// Per-resident KV reconstruction
// ---------------------------------------------------------------------------

func TestResidentKVBlocks_UsesEngineBlockSizeAndCeiling(t *testing.T) {
	// 4000 prompt + 20 streamed = 4020 tokens, ceil(4020/16) = 252 blocks.
	assert.Equal(t, int64(252), residentKVBlocks(Resident{InputLen: 4000, StepsDone: 20}, 16))
	assert.Equal(t, int64(0), residentKVBlocks(Resident{}, 16))

	// An occupant holds only its already-computed prompt span.
	assert.Equal(t, int64(125), occupantKVBlocks(PrefillOccupant{InputLen: 4000, RemPrefillTokens: 2000}, 16))
	assert.Equal(t, int64(0), occupantKVBlocks(PrefillOccupant{InputLen: 4000, RemPrefillTokens: 9000}, 16),
		"a negative computed span must not produce negative blocks")
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

func TestConfigValidate_AcceptsTheRegisteredArm(t *testing.T) {
	require.NoError(t, testConfig().validate())
}

func TestConfigValidate_RejectsConfigurationsThatWouldMisprice(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
		expect string
	}{
		{"zero maxBatchSize forces the wave form", func(c *Config) { c.MaxBatchSize = 0 }, "maxBatchSize must be positive"},
		{"zero chunk budget breaks every overlap window", func(c *Config) { c.ChunkTokens = 0 }, "chunkTokens must be positive"},
		{"zero block size breaks KV accounting", func(c *Config) { c.BlockSize = 0 }, "blockSize must be positive"},
		{"non-positive V", func(c *Config) { c.V = 0 }, "v must be positive"},
		{"all-zero SLO triple flattens the policy", func(c *Config) {
			c.SLOTargets = map[string]SLO{"interactive": {TauITLUs: 50_000}}
		}, "argmin would degenerate"},
		{"no SLO triples at all", func(c *Config) { c.SLOTargets = nil }, "at least one threshold triple"},
		{"selector naming nothing would resolve to a zero triple", func(c *Config) {
			c.ActiveSLOTargets = "interactve"
		}, "names no entry in sloTargets"},
		{"empty selector", func(c *Config) { c.ActiveSLOTargets = "" }, "activeSLOTargets is required"},
		{"no coefficients", func(c *Config) { c.Coefficients = nil }, "at least one GPU type"},
		{"missing GPU label key", func(c *Config) { c.GPUTypeLabelKey = "" }, "gpuTypeLabelKey is required"},
		{"missing role label key", func(c *Config) { c.RoleLabelKey = "" }, "roleLabelKey is required"},
		{"tau_itl below the per-iteration intercept is unachievable", func(c *Config) {
			active := c.SLOTargets[c.ActiveSLOTargets]
			active.TauITLUs = 20_000 // below the A100 intercept of 25,564 us
			c.SLOTargets[c.ActiveSLOTargets] = active
		}, "not achievable on GPU type"},
		{"mismatched alpha halves indicate different hardware", func(c *Config) {
			bad := h100
			bad.AlphaP = bad.AlphaD * 1.5
			c.Coefficients = map[string]Coeffs{"h100": bad}
		}, "fitted on different hardware"},
		{"negative coefficient", func(c *Config) {
			bad := h100
			bad.C1 = -1
			c.Coefficients = map[string]Coeffs{"h100": bad}
		}, "must not be negative"},
		{"size-aware transfer without bandwidth", func(c *Config) { c.XferBandwidthGBps = 0 }, "xferBandwidthGBps must be positive"},
		{"size-aware transfer without KV footprint", func(c *Config) { c.KVBytesPerTokenPerGPU = 0 }, "kvBytesPerTokenPerGPU must be positive"},
		{"decode and prefill result keys collide", func(c *Config) { c.Profiles.Prefill = c.Profiles.Decode }, "must differ"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			tc.mutate(cfg)
			err := cfg.validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.expect)
		})
	}
}

// The interactive triple must clear the A100 intercept, which config.md flags as thin headroom:
// 50 ms against 25.6 ms. If a future coefficient set raises the intercept past the target,
// validation must catch it rather than let the policy run.
func TestConfigValidate_InteractiveTripleClearsBothIntercepts(t *testing.T) {
	cfg := testConfig()
	require.NoError(t, cfg.validate())
	assert.Greater(t, cfg.activeSLO().TauITLUs, a100.AlphaD, "50 ms must exceed the A100 per-iteration intercept")
	assert.Less(t, cfg.activeSLO().TauITLUs-a100.AlphaD, 25_000.0, "and the headroom is thin, as config.md notes")
}

// The selector picks the live triple, and a request-supplied class overrides it ONLY when the class
// names an existing entry. An empty or unrecognised class must fall back to the selector: missing the
// map would resolve to an all-zero triple and flatten the policy.
func TestResolveSLO_SelectorAndClassOverride(t *testing.T) {
	cfg := testConfig()
	assert.Equal(t, interactiveSLO, cfg.resolveSLO(""), "an absent class uses the selector")
	assert.Equal(t, interactiveSLO, cfg.resolveSLO("standard"),
		"every registered cohort declares this one class, so it cannot select a workload")
	assert.Equal(t, interactiveSLO, cfg.resolveSLO("nonsense"), "an unrecognised class must not resolve to zero")
	assert.False(t, cfg.classOverridesSelector(""))
	assert.False(t, cfg.classOverridesSelector("standard"))

	// A class that does name an entry overrides, which is the forward-compatibility path.
	assert.Equal(t, reasoningSLO, cfg.resolveSLO("reasoning"))
	assert.True(t, cfg.classOverridesSelector("reasoning"))

	// Moving the selector is the one-line change that produces another workload's run.
	cfg.ActiveSLOTargets = "deep_research"
	assert.Equal(t, deepResearchSLO, cfg.resolveSLO(""))
	require.NoError(t, cfg.validate(), "every configured workload must be individually runnable")
}

// The SLO triple is what gives the mechanism its discrimination, and it scales as ~1/tau_e2e.
// This is why the treatment config defaults to the interactive workload rather than reasoning.
func TestSLODiscrimination_ScalesInverselyWithE2ETarget(t *testing.T) {
	rt := newReTiming(h100, 8, 16384, 0, 0, 2048, 4000).withExactOverlap(h100, 4000, 4000)
	const now, rem = 1e6, int64(40)
	cb := rt.cBase(now, rem)
	cp := rt.cLocalAfter(now, rem, 0, 2)

	charge := func(slo SLO) float64 {
		return residentContribution(Resident{
			Rem: rem, ArrivalUs: 0, FirstTokenUs: 500_000, TTFTSet: true, Slo: slo,
		}, cb, cp)
	}
	interactive := charge(interactiveSLO)
	reasoning := charge(SLO{TauTTFTUs: 2_000_000, TauITLUs: 100_000, TauE2EUs: 802_000_000})

	require.Positive(t, interactive)
	require.Positive(t, reasoning)
	assert.Greater(t, interactive/reasoning, 10.0,
		"the interactive target must discriminate at least an order of magnitude more than reasoning")
}

// ---------------------------------------------------------------------------
// The substituted estimator must be reported, never silent
// ---------------------------------------------------------------------------

// The published TTFT model is unreachable at this pin, so every candidate falls back. The count
// must be non-zero so the caller can record the substitution rather than letting it pass
// unremarked.
func TestScoreCandidate_ReportsRolloutSubstitution(t *testing.T) {
	cfg := testConfig()
	dec := newTestEndpoint("ns/d-a", h100, 4000)
	pre := newTestEndpoint("ns/p-a", h100, 4000)

	require.False(t, dec.schedulerObserved, "no signal populates the scheduler queue at this pin")
	assert.Equal(t, 1, cfg.scoreCandidate(4000, 1e6, dec, nil, interactiveSLO, 300).rolloutSubstituted)
	// The disaggregated candidate substitutes on both the decode and the prefill endpoint.
	assert.Equal(t, 2, cfg.scoreCandidate(4000, 1e6, dec, pre, interactiveSLO, 300).rolloutSubstituted)

	got := cfg.decide(4000, 1e6, []*endpointState{dec}, []*endpointState{pre}, "", interactiveSLO, 300)
	assert.Positive(t, got.rolloutSubstituted, "the winning candidate must carry the substitution count")
}

// Enabling the capacity term must be rejected at CONSTRUCTION, because the account it reads is not
// reconstructed. A scoring-time NaN sentinel would be strictly worse than a zero: every comparison
// involving NaN is false in Go, so a NaN score never wins the strict argmin comparison and the first
// enumerated candidate would silently take every decision.
func TestConfigValidate_RejectsEnablingTheUnimplementedCapacityTerm(t *testing.T) {
	cfg := testConfig()
	cfg.NoCapacity = false

	err := cfg.validate()
	require.Error(t, err, "enabling capacity must fail at config load, not score strangely at runtime")
	assert.Contains(t, err.Error(), "noCapacity must be true")

	// And the score of a valid configuration must be an ordinary finite number.
	valid := testConfig()
	s := valid.scoreCandidate(4000, 1e6, newTestEndpoint("ns/d-a", h100, 4000), nil, interactiveSLO, 300)
	assert.False(t, math.IsNaN(s.total))
	assert.Zero(t, s.capacityTotal)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// residentsWithRemainingSteps builds an endpoint carrying 40 residents with the given remaining
// step count, holding every load observable fixed.
func residentsWithRemainingSteps(rem int64) *endpointState {
	ep := newTestEndpoint("ns/decode-loaded", h100, 40000)
	for i := 0; i < 40; i++ {
		ep.runningDecode = append(ep.runningDecode, Resident{
			Rem: rem, ArrivalUs: 0, FirstTokenUs: 400_000, TTFTSet: true,
			Slo: interactiveSLO, InputLen: 4000, StepsDone: 50,
		})
	}
	return ep
}

// residentsAtDeadlineOffset builds an endpoint whose residents all have the same remaining step
// count, and therefore suffer the SAME inflicted delay, but arrived at the given offset so their
// remaining E2E slack differs. A more negative offset means a resident further past its deadline.
//
// The realized TTFT is held at a fixed distance from each arrival so the TTFT factor, which is a
// common positive multiplier on the charge, is identical across cases.
func residentsAtDeadlineOffset(arrivalUs int64) *endpointState {
	ep := newTestEndpoint("ns/decode-loaded", h100, 40000)
	for i := 0; i < 40; i++ {
		ep.runningDecode = append(ep.runningDecode, Resident{
			Rem: 400, ArrivalUs: arrivalUs, FirstTokenUs: arrivalUs + 400_000, TTFTSet: true,
			Slo: interactiveSLO, InputLen: 4000, StepsDone: 50,
		})
	}
	return ep
}

func newTestEndpoint(id string, theta Coeffs, ap int) *endpointState {
	return &endpointState{
		id:              id,
		theta:           theta,
		batchSize:       4,
		queueDepth:      0,
		kvTokensInUse:   16384,
		freeKVBlocks:    5000,
		engineBlockSize: 16,
		maxBatchSize:    256,
		ap:              ap,
		prefixObserved:  true,
	}
}
