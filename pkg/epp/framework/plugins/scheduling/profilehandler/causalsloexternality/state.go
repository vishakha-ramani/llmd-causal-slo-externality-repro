package causalsloexternality

// endpointState is the reconstruction of one candidate endpoint's routing view. It is
// deliberately free of framework types so the scoring algebra is testable without
// constructing endpoints, and so the observation code has exactly one place to go wrong.
type endpointState struct {
	// id is the endpoint's namespaced name. It is the shadow-table key, the candidate
	// ordering key, and the only identifier that is identical on the scheduling and the
	// request-lifecycle paths. Pod IP and port are not used: they are not stable across
	// restarts.
	id    string
	theta Coeffs

	// Scraped aggregates. batchSize is the running request count, which includes requests
	// still prefilling; see tIterDecode for why that is deliberate.
	batchSize     int
	queueDepth    int
	kvTokensInUse int64
	freeKVBlocks  int64

	// engineBlockSize is the engine's own KV block size, used to reconstruct per-resident block
	// counts. Validated equal to Config.BlockSize and rejected when absent, because the admission
	// test compares blocks accumulated in this unit against a need expressed in that one.
	engineBlockSize int64

	// maxBatchSize comes from config; no metric carries it.
	maxBatchSize int64

	// ap is the uncached prompt-token count for the arriving request ON THIS ENDPOINT.
	//
	// Evaluated per endpoint rather than once per request, because prefix residency differs
	// per endpoint and a single per-request value would erase the cache-affinity signal.
	// NOT floored at zero here: the chunk terms treat a non-positive value as "no prefill
	// work" and return zero, so the sign has to survive to that check.
	ap int

	// prefixObserved records whether the prefix producer supplied residency for this
	// endpoint. A miss means "no information", which is not the same as "nothing cached", so
	// ap falls back to the full prompt length and the candidate is over-priced rather than
	// silently treated as a cold hit.
	prefixObserved bool

	residentPrefillTokens         int64
	residentPrefillAttentionUnits float64

	runningDecode  []Resident
	runningPrefill []PrefillOccupant

	// schedulerObserved reports whether the engine's per-request scheduler queue contents are
	// available, which would permit the published rollout estimator instead of the
	// closed-form substitute.
	//
	// Nothing populates this true at this pin and no signal exists that could: the only
	// queue observable is a single WaitingQueueSize integer, and there is no route to the
	// running set's per-request prompt and computed token counts, nor to the in-flight step's
	// start time. The field exists so the substitution is explicit at the point it happens
	// rather than implicit in the absence of a branch.
	schedulerObserved bool
}

// decodeIterationLawObservable reports whether the aggregate running-request
// count can be interpreted as B_dec and aggregate KV as decode context. The
// equality against the shadow set rejects locally prefilling, transferred but
// not yet decoding, untracked, and scrape-lagged snapshots.
func (cfg *Config) decodeIterationLawObservable(ep *endpointState) bool {
	if !cfg.RequireNoResidentPrefill {
		return true
	}
	return len(ep.runningPrefill) == 0 && ep.residentPrefillTokens == 0 &&
		ep.residentPrefillAttentionUnits == 0 && ep.batchSize == len(ep.runningDecode)
}

// prefillIterationLawObservable restricts a dedicated prefiller to an idle
// decision snapshot. The arriving request's later chunks are then evaluated
// request-wise by Wp, including their attention marginals.
func (cfg *Config) prefillIterationLawObservable(ep *endpointState) bool {
	if !cfg.RequireNoResidentPrefill {
		return true
	}
	return len(ep.runningPrefill) == 0 && ep.residentPrefillTokens == 0 &&
		ep.residentPrefillAttentionUnits == 0 && ep.batchSize == 0 && ep.queueDepth == 0
}

// DELIBERATE, DECISION-NEUTRAL OMISSIONS.
//
// The upstream routing view also carries a waiting-work backlog (QWork) maintained by
// EPP-side conservation bookkeeping, an observed admission rate, and a virtual capacity
// account drained per decision. None is reconstructed here, and none can change this arm's
// decision:
//
//   - QWork and the admission rate are read only by the `waiting` and `little` admission
//     estimators. This arm selects `rollforward`, which reads neither.
//   - The capacity account feeds only the capacity term, which is disabled for this arm.
//     Maintaining it would be dead state.
//
// They are named here so their absence reads as a decision rather than an oversight. A port
// that re-enables the capacity term must add the account back, along with a monotonic clock
// and per-endpoint last-drain timestamps.
