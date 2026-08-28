package causalsloexternality

import (
	"sync"
	"time"
)

// shadowEntry is one in-flight request this EPP placed.
type shadowEntry struct {
	requestID string
	// decodeID and prefillID are the endpoints this request was routed to.
	decodeID  string
	prefillID string
	sloClass  string

	arrivalUs int64
	inputLen  int64

	// firstTokenUs is stamped when the first completion token is observed.
	//
	// It is a DEQUEUE time, not a chunk-arrival time: non-final response chunks are drained by
	// a per-request goroutine, so the recorded value is LATE. The direction of that error is
	// worth working through rather than guessing. A resident's charge is
	//
	//	g(cb) - g(cp) = u_ttft * [ sigma_e2e(cb) - sigma_e2e(cp) ]
	//
	// where u_ttft is fixed by the REALIZED first token and is therefore a common positive
	// multiplier on both terms. A late first token shrinks u_ttft, scaling the whole difference
	// DOWN. Every resident's charge shrinks, so the externality shrinks, so local placement is
	// under-priced and the policy is biased TOWARD local.
	firstTokenUs int64
	ttftSet      bool

	// stepsDone is the completion-token count observed so far. It requires per-chunk usage to be
	// enabled on the engine or the request; otherwise usage arrives only in the final chunk, this
	// stays 0 for the request's whole lifetime, and every resident's remaining-steps estimate is
	// wrong.
	stepsDone int64

	lastSeen time.Time
}

// shadowTable is the EPP-side view of the requests this replica placed. It substitutes for
// per-resident engine state, which vLLM exports only in aggregate.
//
// It is a long-lived per-endpoint index rather than per-request scratch state, so it is a
// mutex-guarded field rather than framework-provided request state.
//
// THE MUTEX IS REQUIRED. Entries are written from the async response-body goroutine and read
// from the scheduling path, which are different goroutines.
//
// THREE DEGRADATIONS ARE STRUCTURAL TO THIS DESIGN, all pointing the same way:
//
//	(a) One EPP replica only. Replicas partition this table, so a second replica sees a strict
//	    subset of residents and UNDER-counts the externality, biasing toward local. The
//	    deployment pins one replica.
//	(b) Traffic bypassing this EPP is invisible here. The aggregate running count still sees it,
//	    so the batch size is right while the resident detail is short.
//	(c) The recorded first token is late; see shadowEntry.firstTokenUs.
type shadowTable struct {
	mu sync.RWMutex
	// byRequest owns the entries; byDecode and byPrefill index into it.
	byRequest map[string]*shadowEntry
	byDecode  map[string]map[string]*shadowEntry
	byPrefill map[string]map[string]*shadowEntry
}

func newShadowTable() *shadowTable {
	return &shadowTable{
		byRequest: map[string]*shadowEntry{},
		byDecode:  map[string]map[string]*shadowEntry{},
		byPrefill: map[string]map[string]*shadowEntry{},
	}
}

// open records a placement. A repeated requestID replaces the earlier entry, so a retried request
// cannot be counted twice.
func (s *shadowTable) open(e *shadowEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if prev, ok := s.byRequest[e.requestID]; ok {
		s.unlinkLocked(prev)
	}
	e.lastSeen = time.Now()
	s.byRequest[e.requestID] = e
	if e.decodeID != "" {
		if s.byDecode[e.decodeID] == nil {
			s.byDecode[e.decodeID] = map[string]*shadowEntry{}
		}
		s.byDecode[e.decodeID][e.requestID] = e
	}
	if e.prefillID != "" {
		if s.byPrefill[e.prefillID] == nil {
			s.byPrefill[e.prefillID] = map[string]*shadowEntry{}
		}
		s.byPrefill[e.prefillID][e.requestID] = e
	}
}

// observe updates progress for one request. Traffic this replica did not place is unknown here and
// is ignored: it has no recorded arrival or input length, so it cannot be a resident.
//
// firstToken is recorded only on the transition to a non-zero completion count, so a later chunk
// cannot move it. stepsDone is monotonic: a chunk reporting fewer tokens than already seen is
// ignored rather than allowed to walk the count backwards.
func (s *shadowTable) observe(requestID string, completionTokens int64, nowUs int64) (shadowEntry, bool, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.byRequest[requestID]
	if !ok {
		return shadowEntry{}, false, false
	}
	e.lastSeen = time.Now()
	if completionTokens > e.stepsDone {
		e.stepsDone = completionTokens
	}
	firstToken := false
	if !e.ttftSet && completionTokens >= 1 {
		e.firstTokenUs = nowUs
		e.ttftSet = true
		firstToken = true
	}
	return *e, firstToken, true
}

// close deregisters a request. Called when the response stream ends.
func (s *shadowTable) close(requestID string) (*shadowEntry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.byRequest[requestID]
	if !ok {
		return nil, false
	}
	s.unlinkLocked(e)
	return e, true
}

func (s *shadowTable) unlinkLocked(e *shadowEntry) {
	delete(s.byRequest, e.requestID)
	if m := s.byDecode[e.decodeID]; m != nil {
		delete(m, e.requestID)
		if len(m) == 0 {
			delete(s.byDecode, e.decodeID)
		}
	}
	if m := s.byPrefill[e.prefillID]; m != nil {
		delete(m, e.requestID)
		if len(m) == 0 {
			delete(s.byPrefill, e.prefillID)
		}
	}
}

// residents returns one decode endpoint's view: residents that have produced a first token, and
// pre-first-token occupants collocated on it.
//
// The split is what the value kernel requires: the decode-side kernel returns zero for a resident
// with no first token, so such a request would be missed entirely if it were not moved into the
// occupant population, whose TTFT is still live.
//
// Snapshots are copied out under the lock so the scheduling path never holds it while scoring.
func (s *shadowTable) residents(
	endpointID string, cfg *Config, nHatOut func(sloClass string) float64,
) ([]Resident, []PrefillOccupant) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries := s.byDecode[endpointID]
	if len(entries) == 0 {
		return nil, nil
	}
	residents := make([]Resident, 0, len(entries))
	occupants := make([]PrefillOccupant, 0)
	for _, e := range entries {
		slo := cfg.resolveSLO(e.sloClass)
		nHat := nHatOut(e.sloClass)
		if e.ttftSet {
			residents = append(residents, Resident{
				RequestID:    e.requestID,
				Rem:          residentRem(nHat, e.stepsDone),
				ArrivalUs:    e.arrivalUs,
				FirstTokenUs: e.firstTokenUs,
				TTFTSet:      true,
				Slo:          slo,
				StepsDone:    e.stepsDone,
				InputLen:     e.inputLen,
			})
			continue
		}
		// No first token yet. A request whose prefill ran remotely is not prefilling HERE, so it
		// occupies no prefill capacity on this decode endpoint; it is simply not yet chargeable
		// and is skipped.
		if e.prefillID != "" && e.prefillID != endpointID {
			continue
		}
		occupants = append(occupants, PrefillOccupant{
			RequestID:        e.requestID,
			RemPrefillTokens: remainingPrompt(e),
			RemDecodeSteps:   occupantRemDecodeSteps(nHat),
			ArrivalUs:        e.arrivalUs,
			Slo:              slo,
			InputLen:         e.inputLen,
		})
	}
	return residents, occupants
}

// prefillOccupants returns a prefill pool endpoint's pre-first-token occupants: the requests this
// replica sent there whose prefill has not completed.
//
// Where a pool's occupants are unobserved this is empty and the pool's charge is zero, which
// UNDER-prices remote prefill and biases toward disaggregation.
func (s *shadowTable) prefillOccupants(
	endpointID string, cfg *Config, nHatOut func(sloClass string) float64,
) []PrefillOccupant {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries := s.byPrefill[endpointID]
	if len(entries) == 0 {
		return nil
	}
	occupants := make([]PrefillOccupant, 0, len(entries))
	for _, e := range entries {
		if e.ttftSet {
			continue
		}
		occupants = append(occupants, PrefillOccupant{
			RequestID:        e.requestID,
			RemPrefillTokens: remainingPrompt(e),
			RemDecodeSteps:   occupantRemDecodeSteps(nHatOut(e.sloClass)),
			ArrivalUs:        e.arrivalUs,
			Slo:              cfg.resolveSLO(e.sloClass),
			InputLen:         e.inputLen,
		})
	}
	return occupants
}

// remainingPrompt is an occupant's outstanding prompt tokens. Before the first token nothing has
// been streamed, so the whole prompt is outstanding; the caller caps this per chunk when it forms
// the prefilling-token estimate.
func remainingPrompt(e *shadowEntry) int64 {
	if e.inputLen < 0 {
		return 0
	}
	return e.inputLen
}

// reap drops entries older than ttl. Requests can terminate without a final chunk (client
// disconnect, upstream error), and the response hooks carry no termination state, so without a
// sweep those entries would be charged as residents forever and permanently inflate every
// externality on their endpoint.
func (s *shadowTable) reap(ttl time.Duration, now time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	var dropped int
	for _, e := range s.byRequest {
		if now.Sub(e.lastSeen) > ttl {
			s.unlinkLocked(e)
			dropped++
		}
	}
	return dropped
}

// size reports the number of tracked requests.
func (s *shadowTable) size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.byRequest)
}

// classMeans tracks the per-class running mean of realized output lengths.
//
// Updated ONLY on requests that actually completed. A request reaching a terminal state without
// completing carries no realized output length, and folding its truncated count in would drag the
// estimate down.
type classMeans struct {
	mu sync.RWMutex
	m  map[string]*runningMean
}

func newClassMeans() *classMeans {
	return &classMeans{m: map[string]*runningMean{}}
}

func (c *classMeans) update(sloClass string, outputTokens float64) {
	if outputTokens <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	rm, ok := c.m[sloClass]
	if !ok {
		rm = &runningMean{}
		c.m[sloClass] = rm
	}
	rm.update(outputTokens)
}

// mean returns the class mean, seeded at 1 for a class with no completions yet. A zero seed would
// make the decode demand vanish and collapse every remaining-steps floor.
func (c *classMeans) mean(sloClass string) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if rm, ok := c.m[sloClass]; ok {
		return rm.mean()
	}
	return 1
}
