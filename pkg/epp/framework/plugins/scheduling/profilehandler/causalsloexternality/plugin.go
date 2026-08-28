// Package causalsloexternality implements joint causal-SLO-externality P/D placement.
//
// The decision rule is a joint argmin over (decode endpoint d, prefill placement p) of
//
//	J(d, p) = V * ( externality(d, p) - ownGood(d, p) )
//
// Each candidate is priced by the smooth SLO value it destroys among the residents already on the
// endpoint, evaluated CAUSALLY: every completion model is gated on an admission window, so a
// resident that finishes before the arrival is admitted contributes exactly zero. That gating is
// what makes the term an externality rather than another load proxy, and it is why the signal
// survives on a fleet whose load signals are already balanced. Two endpoints with identical queue
// depth and identical KV occupancy are not interchangeable: one may hold residents just inside
// their end-to-end deadline, where an extra co-scheduled prefill flips them to a miss, and the
// other residents with comfortable slack, where the same prefill costs nothing recoverable.
//
// WHY THIS IS A PICKER AND A PROFILE HANDLER RATHER THAN A SCORER. Two independent reasons, both
// structural. A Scorer returns one scalar per endpoint, so no (decode, prefill) PAIR can be named.
// And every scorer contribution is passed through a [0,1] clamp, while J is signed and unbounded,
// so clamping would collapse every candidate with J <= 0 to a single score and destroy the ranking
// rather than degrade it.
//
// WHY THE ARGMIN IS JOINT. Fixing the decode endpoint first and then asking a binary gate whether
// to disaggregate is a DIFFERENT ALGORITHM with a measured cost: -0.0485 equal-cell mean goodput
// and -0.1555 worst cell. Such a port still runs and still reports goodput, which is what makes
// the mistake dangerous. The single role-blind profile and the cross-product enumeration are
// load-bearing, not stylistic.
//
// THE TWO-PLUGIN SPLIT IS FORCED BY THE LANGUAGE, NOT CHOSEN.
//
//	ProfileHandler.Pick(ctx, request, profiles, profileResults) map[string]SchedulerProfile
//	                                    framework/interface/scheduling/plugins.go:47-48
//	Picker.Pick(ctx, scoredPods []*ScoredEndpoint) *ProfileRunResult
//	                                    framework/interface/scheduling/plugins.go:77
//
// Same method name, incompatible signatures, so no Go type can satisfy both interfaces. The
// specification describes this placement as a custom Picker PLUS a custom ProfileHandler as though
// it were one extension point; at the language level it cannot be.
//
// The stock disaggregation handler is not a counterexample. It asserts ProfileHandler and PreRequest
// on one struct, which works precisely because PreRequest contributes no Pick method.
//
// So the policy is one shared core behind two registered types: Handler owns the state and the
// request lifecycle, Picker performs the argmin. Picker resolves its Handler by name at construction,
// the same mechanism the stock handler uses for its decider, which requires the Handler to be
// declared FIRST because factories run in declaration order. There is no cross-plugin state channel:
// both types hold the same core pointer.
//
// DECLARED DEVIATIONS FROM THE SPECIFICATION, recorded here because they change what the measured
// numbers mean:
//
//   - The published TTFT model is a scheduler rollout over the engine's ordered wait queue. Its
//     per-request inputs do not exist on this target, so the closed-form rollforward estimator is
//     substituted and every substitution is counted. Upstream substitutes silently.
//
//   - The comparison baseline routes on APPROXIMATE rather than precise prefix residency. The
//     stock prefix-threshold decider reads the bare prefix data key, whose identity includes its
//     producer name, so it can only ever see the approximate producer's attribute. A precise-only
//     baseline would miss that key on every request and never disaggregate: loud in the logs, and
//     silent in the results. The precise scorer is separately deprecated at this pin, and precise
//     routing would additionally require engine-side KV-event publication the deployment does not
//     configure.
//
//   - The baseline's disaggregation threshold is PORT-INTRODUCED and has no campaign provenance.
//     The simulation arm was tuned per workload and records no value; the target tree offers four
//     unrelated ones. It is not the campaign's tuned threshold and must not be reported as such.
//
//   - Per-resident state comes from an EPP-side shadow table. See shadowTable for the three
//     degradations that follow, all pointing the same way.
//
//   - The deployment configures a token producer with the vLLM render backend, so the input length
//     is a real tokenization. This ADDS something the simulation never had: an uncached
//     tokenization round trip on the request path, which is inside the measured time to first
//     token. It is common-mode across both arms, so the ordering claim is unaffected, but it
//     inflates absolute TTFT in both arms against a 1,000 ms interactive target. Its magnitude is
//     unquantified until the argmin-duration histogram and the framework's producer timings are
//     read from a run.
//
//     Omitting that producer would be worse than imprecise. The auto-created fallback packs request
//     bytes at four bytes per pseudo-token, and under it two quantities become MIXED-UNIT: the
//     per-resident KV reconstruction sums a pseudo-token prompt length with a real-token completion
//     count, and the KV block need divides pseudo-tokens by a real-token block size. Those are
//     correctness faults rather than accuracy ones, and the configured backend removes both.
//
//     This is also LINKED to the approximate-prefix entry above rather than independent of it: with
//     real token IDs the approximate index hashes the same token stream the engine caches, so the
//     uncached prompt length becomes engine-correlated instead of merely self-consistent. Not
//     exact, since the prefix block size still differs from the engine's.
//
// ONE QUALIFICATION OF THE BUNDLE'S PROSE, not a degradation of this port. The bundle describes
// residents that are already past their deadline as costing nothing recoverable. Measured here with
// the inflicted delay and the realized first token held fixed, the charge peaks at the deadline and
// falls away on both sides, but the peak is BROAD: residents up to roughly six seconds past their
// deadline are still priced near the maximum, and the charge only collapses tens of seconds out.
// The kernel is reproduced faithfully; it is the phrase that is loose.
package causalsloexternality

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// shadowEntryTTL bounds how long an entry survives without response activity. Requests can
	// terminate without a final chunk and the response hooks carry no termination state, so entries
	// must expire or they are charged as residents forever.
	shadowEntryTTL = 10 * time.Minute

	// shadowReapInterval is how often the staleness sweep runs.
	shadowReapInterval = time.Minute
)

// compile-time assertions.
var (
	_ fwksched.ProfileHandler     = &Handler{}
	_ fwkrc.PreRequest            = &Handler{}
	_ fwkrc.ResponseBodyProcessor = &Handler{}
	_ fwkplugin.ConsumerPlugin    = &Handler{}

	_ fwksched.Filter          = &Picker{}
	_ fwksched.Picker          = &Picker{}
	_ fwkplugin.ConsumerPlugin = &Picker{}
)

// core is the state both registered types share: the configuration, the shadow table that supplies
// per-resident estimates, and the per-class output-length means.
type core struct {
	cfg    *Config
	shadow *shadowTable
	means  *classMeans

	// start anchors the decision clock. Elapsed microseconds from one monotonic origin are used
	// rather than wall-clock timestamps, so the scheduling path and the response path agree and no
	// comparison can be perturbed by a clock adjustment.
	start time.Time

	// prefixKey is the producer-qualified prefix data key. A DataKey's identity includes its
	// producer name, so reading the bare key would bind to whichever producer owns the default
	// name rather than the configured one.
	prefixKey fwkplugin.DataKey
}

func (c *core) nowUs() int64 { return time.Since(c.start).Microseconds() }

// consumes is the dependency declaration both types share.
//
// The declared types are VALUE types even though the endpoint attribute is stored as a pointer: the
// producers declare value types, and the dependency graph matches produced against consumed by
// exact reflected type, so declaring a pointer here fails startup with a type mismatch.
func (c *core) consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			c.prefixKey:                          attrprefix.PrefixCacheMatchInfo{},
			tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{},
		},
	}
}

// ---------------------------------------------------------------------------
// Handler: state owner, profile orchestration, request lifecycle
// ---------------------------------------------------------------------------

// Handler owns the policy state, drives the single role-blind profile, splits the chosen pair into
// decode and prefill results, and maintains the shadow table across the request lifecycle.
type Handler struct {
	typedName fwkplugin.TypedName
	*core
}

// HandlerFactory builds the Handler from the policy's parameters block.
func HandlerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}

	cfg := defaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid '%s' configuration: %w", PluginType, err)
	}

	h := &Handler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		core: &core{
			cfg:       &cfg,
			shadow:    newShadowTable(),
			means:     newClassMeans(),
			start:     time.Now(),
			prefixKey: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(cfg.PrefixMatchInfoProducerName),
		},
	}
	go h.reapLoop(handle.Context())

	log.FromContext(handle.Context()).Info("Configured joint causal-SLO-externality placement",
		"plugin", h.typedName, "v", cfg.V,
		"activeSLOTargets", cfg.ActiveSLOTargets,
		"tauTTFTUs", cfg.activeSLO().TauTTFTUs, "tauE2EUs", cfg.activeSLO().TauE2EUs,
		"configuredWorkloads", sortedKeys(cfg.SLOTargets),
		"chunkTokens", cfg.ChunkTokens, "blockSize", cfg.BlockSize, "maxBatchSize", cfg.MaxBatchSize,
		"gpuTypes", sortedKeys(cfg.Coefficients),
		"prefixProducer", cfg.PrefixMatchInfoProducerName,
		"ttftEstimator", "rollforward substitute; the published scheduler rollout is unobservable at this target")
	return h, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *Handler) TypedName() fwkplugin.TypedName { return h.typedName }

// WithName sets the instance name.
func (h *Handler) WithName(name string) *Handler {
	h.typedName.Name = name
	return h
}

// Consumes declares the inputs the policy requires.
func (h *Handler) Consumes() fwkplugin.DataDependencies { return h.consumes() }

// reapLoop drops shadow entries whose requests stopped reporting.
func (h *Handler) reapLoop(ctx context.Context) {
	ticker := time.NewTicker(shadowReapInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if n := h.shadow.reap(shadowEntryTTL, time.Now()); n > 0 {
				shadowTableReapedCount.WithLabelValues(h.typedName.Name, h.typedName.Type).Add(float64(n))
				log.FromContext(ctx).V(logutil.DEBUG).Info("Reaped stale shadow-table entries",
					"plugin", h.typedName, "dropped", n, "remaining", h.shadow.size())
			}
			shadowTableSize.WithLabelValues(h.typedName.Name, h.typedName.Type).Set(float64(h.shadow.size()))
		}
	}
}

// Pick selects the single role-blind profile to run, then reports completion.
//
// There is deliberately no decode stage followed by a disaggregation gate: that decomposition is
// the measured ablation, and running the joint profile once is what keeps the cross product intact.
func (h *Handler) Pick(ctx context.Context, request *fwksched.InferenceRequest,
	profiles map[string]fwksched.SchedulerProfile, profileResults map[string]*fwksched.ProfileRunResult,
) map[string]fwksched.SchedulerProfile {
	none := map[string]fwksched.SchedulerProfile{}
	if request == nil {
		return none
	}
	if _, executed := profileResults[h.cfg.Profiles.Joint]; executed {
		return none
	}
	profile, ok := profiles[h.cfg.Profiles.Joint]
	if !ok {
		log.FromContext(ctx).Error(nil, "Configured joint profile is absent; a scheduling profile must be named to match profiles.joint",
			"plugin", h.typedName, "expected", h.cfg.Profiles.Joint)
		return none
	}
	return map[string]fwksched.SchedulerProfile{h.cfg.Profiles.Joint: profile}
}

// ProcessResults splits the picker's (decode, prefill) pair into the two result entries that put
// the prefill endpoint on the wire.
//
// The result keys need not name configured profiles: they are the keys PreRequest and any
// downstream consumer read.
func (h *Handler) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult,
) (*fwksched.SchedulingResult, error) {
	if request == nil {
		return nil, errors.New("request is nil")
	}
	joint := profileResults[h.cfg.Profiles.Joint]
	if joint == nil || len(joint.TargetEndpoints) == 0 {
		return nil, errors.New("joint causal-SLO-externality profile selected no endpoint")
	}

	// INVARIANT: Profiles.Decode must key an entry whenever this returns a result, and it is also
	// the PrimaryProfileName below. The director dereferences
	// ProfileResults[PrimaryProfileName].TargetEndpoints with no nil check, so a missing key panics
	// in the request path rather than failing here. The decode entry is therefore unconditional
	// while the prefill entry below is conditional; the two look symmetric and are not.
	results := map[string]*fwksched.ProfileRunResult{
		h.cfg.Profiles.Decode: {TargetEndpoints: joint.TargetEndpoints[:1]},
	}
	if len(joint.TargetEndpoints) > 1 {
		results[h.cfg.Profiles.Prefill] = &fwksched.ProfileRunResult{TargetEndpoints: joint.TargetEndpoints[1:2]}
	}
	return &fwksched.SchedulingResult{
		PrimaryProfileName: h.cfg.Profiles.Decode,
		ProfileResults:     results,
	}, nil
}

// PreRequest puts the prefill endpoint on the wire and opens this request's shadow entry.
//
// The recorded arrival is the routing instant rather than the client's arrival: this hook fires
// after scheduling, and no arrival timestamp is reachable from a scheduling plugin. Resident TTFTs
// are therefore measured from routing, which understates them by the pre-scheduling latency.
func (h *Handler) PreRequest(_ context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	if request == nil || schedulingResult == nil {
		return
	}

	delete(request.Headers, routing.PrefillEndpointHeader)
	var prefillID string
	if res := schedulingResult.ProfileResults[h.cfg.Profiles.Prefill]; res != nil && len(res.TargetEndpoints) > 0 {
		if meta := res.TargetEndpoints[0].GetMetadata(); meta != nil {
			if request.Headers == nil {
				request.Headers = map[string]string{}
			}
			request.Headers[routing.PrefillEndpointHeader] = net.JoinHostPort(meta.Address, meta.Port)
			prefillID = meta.GetNamespacedName().String()
		}
	}

	decodeID := endpointIDFrom(schedulingResult, h.cfg.Profiles.Decode)
	if decodeID == "" {
		// Without a decode endpoint there is nothing to key a resident on.
		return
	}

	h.shadow.open(&shadowEntry{
		requestID: request.RequestID,
		decodeID:  decodeID,
		prefillID: prefillID,
		sloClass:  sloClassOf(request),
		arrivalUs: h.nowUs(),
		inputLen:  int64(promptTokensOf(request)),
	})
}

// ResponseBody records first-token and progress, and deregisters the request at end of stream.
//
// First token is detected by the completion count first reaching one. The response carries no body
// bytes, so the router's own "first non-empty chunk" rule is unavailable here; the usage count is
// exact and is the same signal the progress estimate already needs.
//
// PER-CHUNK USAGE IS A PRECONDITION. Without it the count arrives only in the final chunk, so the
// progress estimate stays zero for a request's whole lifetime and every resident's remaining-steps
// estimate is wrong.
func (h *Handler) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest,
	response *fwkrc.Response, _ *fwkdl.EndpointMetadata,
) {
	if request == nil || response == nil {
		return
	}
	completion := int64(response.Usage.CompletionTokens)
	nowUs := h.nowUs()
	observed, firstToken, known := h.shadow.observe(request.RequestID, completion, nowUs)
	if known && firstToken {
		placement := decisionLocal
		if observed.prefillID != "" {
			placement = decisionDisagg
		}
		fields := eppObservationFields(eventEPPFirstToken, request.RequestID, placement,
			observed.decodeID, observed.prefillID, time.Now(), nowUs-observed.arrivalUs)
		fields = append(fields,
			"observedTTFTUs", nowUs-observed.arrivalUs,
			"completionTokens", completion,
			"observationClass", "epp_response_callback")
		log.FromContext(ctx).Info("Causal SLO observation", fields...)
	}

	if !response.EndOfStream {
		return
	}

	entry, ok := h.shadow.close(request.RequestID)
	if !ok {
		return
	}
	// Only realized output counts toward the class estimate. The final chunk's usage is
	// authoritative, but a stream whose last chunk omitted usage would report less than already
	// observed, so take the larger of the two.
	realized := completion
	if entry.stepsDone > realized {
		realized = entry.stepsDone
	}
	h.means.update(entry.sloClass, float64(realized))
	placement := decisionLocal
	if entry.prefillID != "" {
		placement = decisionDisagg
	}
	observedTTFTUs := int64(-1)
	if entry.ttftSet {
		observedTTFTUs = entry.firstTokenUs - entry.arrivalUs
	}
	fields := eppObservationFields(eventEPPComplete, request.RequestID, placement,
		entry.decodeID, entry.prefillID, time.Now(), nowUs-entry.arrivalUs)
	fields = append(fields,
		"observedTTFTUs", observedTTFTUs,
		"observedE2EUs", nowUs-entry.arrivalUs,
		"outputTokens", realized,
		"classMean", h.means.mean(entry.sloClass),
		"observationClass", "epp_response_callback")
	log.FromContext(ctx).Info("Causal SLO observation", fields...)

	if v := log.FromContext(ctx).V(logutil.TRACE); v.Enabled() {
		v.Info("Closed shadow-table entry", "plugin", h.typedName, "requestID", request.RequestID,
			"decode", entry.decodeID, "prefill", entry.prefillID,
			"outputTokens", realized, "classMean", h.means.mean(entry.sloClass))
	}
}

func endpointIDFrom(result *fwksched.SchedulingResult, profile string) string {
	res := result.ProfileResults[profile]
	if res == nil || len(res.TargetEndpoints) == 0 {
		return ""
	}
	meta := res.TargetEndpoints[0].GetMetadata()
	if meta == nil {
		return ""
	}
	return meta.GetNamespacedName().String()
}

// ---------------------------------------------------------------------------
// Picker: the joint argmin
// ---------------------------------------------------------------------------

// pickerParameters names the Handler instance whose state this picker scores against.
type pickerParameters struct {
	PolicyPluginName string `json:"policyPluginName"`
}

// Picker computes J over the cross product and returns the argmin. It also implements Filter, which
// is the only route by which the arriving request reaches the argmin.
type Picker struct {
	typedName fwkplugin.TypedName
	*core
}

// PickerFactory builds the Picker and binds it to its Handler.
func PickerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}

	params := pickerParameters{PolicyPluginName: PluginType}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PickerType, err)
		}
	}
	if params.PolicyPluginName == "" {
		return nil, fmt.Errorf("%s requires policyPluginName naming a %s instance", PickerType, PluginType)
	}

	target := handle.Plugin(params.PolicyPluginName)
	if target == nil {
		return nil, fmt.Errorf("%s: policyPluginName %q not found; the %s plugin must be declared BEFORE this one, because factories run in declaration order",
			PickerType, params.PolicyPluginName, PluginType)
	}
	handler, ok := target.(*Handler)
	if !ok {
		return nil, fmt.Errorf("%s: plugin %q is a %s, not a %s", PickerType, params.PolicyPluginName, target.TypedName().Type, PluginType)
	}

	return &Picker{
		typedName: fwkplugin.TypedName{Type: PickerType, Name: name},
		core:      handler.core,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Picker) TypedName() fwkplugin.TypedName { return p.typedName }

// WithName sets the instance name.
func (p *Picker) WithName(name string) *Picker {
	p.typedName.Name = name
	return p
}

// Consumes declares the inputs the argmin reads.
func (p *Picker) Consumes() fwkplugin.DataDependencies { return p.consumes() }

// candidateEndpoint carries the request alongside an endpoint.
//
// A Picker receives only (context, scored endpoints), and the request is reachable from neither:
// nothing places it in the context. But J needs the arriving request's input length INSIDE the
// argmin. A Filter does receive the request, and endpoint identity is preserved verbatim from
// filters through scorers to the picker, so wrapping here is the supported way to carry it.
//
// It MUST be a pointer type. The scoring stage keys a map by the Endpoint interface and Go hashes
// the dynamic type, so a struct value carrying a map, slice or func field would panic on insertion.
//
// The wrapper never escapes: Pick unwraps before building its result, so no downstream consumer,
// and in particular no endpoint comparison that type-asserts the concrete endpoint, observes it.
type candidateEndpoint struct {
	fwksched.Endpoint
	request *fwksched.InferenceRequest
}

// Filter attaches the request to every candidate. It performs NO role filtering: the argmin needs
// both pools in one candidate set, and a role split here would reduce this to the decode-first
// decomposition.
func (p *Picker) Filter(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	out := make([]fwksched.Endpoint, len(endpoints))
	for i, ep := range endpoints {
		out[i] = &candidateEndpoint{Endpoint: ep, request: request}
	}
	return out
}

// Pick computes J for every (decode, placement) candidate and returns the argmin.
//
// The returned TargetEndpoints carry the decode endpoint first and, for a disaggregated placement,
// the prefill endpoint second. The Handler's ProcessResults splits that pair.
//
// The inherited Score field is IGNORED: it is the clamped upstream sum, and this objective is
// computed here from the endpoint values instead.
func (p *Picker) Pick(ctx context.Context, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	logger := log.FromContext(ctx)
	if len(scoredEndpoints) == 0 {
		return &fwksched.ProfileRunResult{}
	}
	started := time.Now()
	defer func() {
		argminDurationSeconds.WithLabelValues(p.typedName.Name, p.typedName.Type).
			Observe(time.Since(started).Seconds())
	}()

	request := requestFrom(scoredEndpoints)
	if request == nil {
		// Without the request there is no input length, so J cannot be formed. Refusing is
		// correct: scoring against a default prompt length would price every candidate wrongly
		// while still returning a plausible answer.
		logger.Error(nil, "No request reached the picker; the profile must list this plugin so its Filter runs",
			"plugin", p.typedName)
		return &fwksched.ProfileRunResult{}
	}

	// TOKENIZATION FAILURE MUST NOT BE SCORED. Data-producer errors are swallowed upstream, so a
	// renderer that is down or slow leaves the tokenized prompt absent and the request proceeds.
	// Refusing here is deliberate: substituting a byte estimate would silently reprice every
	// candidate, and scoring with a zero input length would collapse the prefill terms and hand the
	// decision to enumeration order. Neither failure would be visible in the results.
	//
	// A genuinely empty prompt is a different case and MUST still be scored: the argmin still
	// chooses which decode endpoint to use. See tokenizationFailed for why the tokenized prompt
	// alone cannot tell the two apart.
	if tokenizationFailed(request) {
		tokenizationFailureCount.WithLabelValues(p.typedName.Name, p.typedName.Type).Inc()
		logger.Error(nil, "Refusing to score: the request carries no tokenized prompt, so the input length that every term of the objective depends on is unavailable. Check the token-producer and its render backend",
			"plugin", p.typedName, "requestID", request.RequestID)
		return &fwksched.ProfileRunResult{}
	}

	promptTokens := promptTokensOf(request)
	class := sloClassOf(request)
	if class != "" && !p.cfg.classOverridesSelector(class) {
		logger.V(logutil.DEBUG).Info("Request SLO class names no configured threshold entry; using the configured selector",
			"plugin", p.typedName, "class", class, "activeSLOTargets", p.cfg.ActiveSLOTargets)
	}
	slo := p.cfg.resolveSLO(class)
	nHatOut := p.means.mean(class)
	now := float64(p.nowUs())

	decodes, prefills, handles := p.buildCandidates(ctx, scoredEndpoints, promptTokens)
	if len(decodes) == 0 {
		logger.Error(nil, "No usable decode candidate remains; check the GPU-type and role labels on every endpoint",
			"plugin", p.typedName, "candidates", len(scoredEndpoints))
		return &fwksched.ProfileRunResult{}
	}

	chosen, evaluations := p.cfg.decideDetailed(promptTokens, now, decodes, prefills, "", slo, nHatOut)
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		for candidateIndex, e := range evaluations {
			placement := decisionLocal
			if e.decision.disaggregate {
				placement = decisionDisagg
			}
			selected := e.decision.disaggregate == chosen.disaggregate &&
				e.decision.decodeID == chosen.decodeID && e.decision.prefillID == chosen.prefillID
			eventTime := time.Now()
			fields := eppObservationFields(eventDecisionCandidate, request.RequestID, placement,
				e.decision.decodeID, e.decision.prefillID, eventTime, time.Since(started).Microseconds())
			fields = append(fields,
				"candidateIndex", candidateIndex,
				"selected", selected,
				"promptTokens", promptTokens,
				"V", p.cfg.V,
				"noExternality", p.cfg.NoExternality,
				"noOwnGood", p.cfg.NoOwnGood,
				"noCapacity", p.cfg.NoCapacity,
				"outputTokenProcessingUs", p.cfg.OutputTokenProcessingUs,
				"tauTTFTUs", slo.TauTTFTUs,
				"tauITLUs", slo.TauITLUs,
				"tauE2EUs", slo.TauE2EUs,
				"observationClass", "prediction")
			fields = append(fields, candidateObservationFields(e, nHatOut)...)
			v.Info("Causal SLO candidate", fields...)

			predictionWallAnchor := p.start.Add(time.Duration(now) * time.Microsecond)
			for _, observation := range p.cfg.externalityObservations(promptTokens, now, e.decode, e.prefill, e.score) {
				residentFields := eppObservationFields(eventDecisionResidentPrediction, request.RequestID, placement,
					e.decision.decodeID, e.decision.prefillID, time.Now(), time.Since(started).Microseconds())
				residentFields = append(residentFields,
					"candidateIndex", candidateIndex,
					"selected", selected,
					"observationClass", "counterfactual_prediction")
				residentFields = append(residentFields,
					externalityObservationFields(observation, now, predictionWallAnchor)...)
				v.Info("Causal SLO resident prediction", residentFields...)
			}
		}
	}

	if chosen.rolloutSubstituted > 0 {
		ttftEstimatorSubstitutionCount.WithLabelValues(p.typedName.Name, p.typedName.Type).
			Add(float64(chosen.rolloutSubstituted))
	}
	placement := decisionLocal
	if chosen.disaggregate {
		placement = decisionDisagg
	}
	placementDecisionCount.WithLabelValues(p.typedName.Name, p.typedName.Type, request.TargetModel, placement).Inc()

	decodeEP, ok := handles[chosen.decodeID]
	if !ok {
		logger.Error(nil, "Chosen decode endpoint is absent from the candidate handles",
			"plugin", p.typedName, "decode", chosen.decodeID)
		return &fwksched.ProfileRunResult{}
	}
	targets := []fwksched.Endpoint{decodeEP}
	if chosen.disaggregate {
		prefillEP, ok := handles[chosen.prefillID]
		if !ok {
			logger.Error(nil, "Chosen prefill endpoint is absent from the candidate handles; placing locally instead",
				"plugin", p.typedName, "prefill", chosen.prefillID)
		} else {
			targets = append(targets, prefillEP)
		}
	}

	for _, e := range evaluations {
		if e.decision.disaggregate != chosen.disaggregate || e.decision.decodeID != chosen.decodeID ||
			e.decision.prefillID != chosen.prefillID {
			continue
		}
		actualPlacement := decisionLocal
		if len(targets) > 1 {
			actualPlacement = decisionDisagg
		}
		fields := eppObservationFields(eventDecisionSelected, request.RequestID, actualPlacement,
			chosen.decodeID, chosen.prefillID, time.Now(), time.Since(started).Microseconds())
		fields = append(fields,
			"plugin", p.typedName,
			"requestedPlacement", placement,
			"routingFallback", actualPlacement != placement,
			"decodeCandidates", len(decodes),
			"prefillCandidates", len(prefills),
			"promptTokens", promptTokens,
			"V", p.cfg.V,
			"noExternality", p.cfg.NoExternality,
			"noOwnGood", p.cfg.NoOwnGood,
			"noCapacity", p.cfg.NoCapacity,
			"outputTokenProcessingUs", p.cfg.OutputTokenProcessingUs,
			"tauTTFTUs", slo.TauTTFTUs,
			"tauITLUs", slo.TauITLUs,
			"tauE2EUs", slo.TauE2EUs,
			"ttftEstimator", "rollforward-substitute",
			"observationClass", "prediction")
		fields = append(fields, candidateObservationFields(e, nHatOut)...)
		logger.Info("Selected placement by joint causal-SLO-externality argmin", fields...)
		break
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: targets}
}

// requestFrom recovers the request the Filter attached. Endpoint identity is preserved through the
// scoring stage, so any candidate carries it.
func requestFrom(scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.InferenceRequest {
	for _, se := range scoredEndpoints {
		if ce, ok := se.Endpoint.(*candidateEndpoint); ok && ce.request != nil {
			return ce.request
		}
	}
	return nil
}

// promptTokensOf is the arriving request's input length in tokens.
func promptTokensOf(request *fwksched.InferenceRequest) int {
	if request == nil || request.Body == nil {
		return 0
	}
	return request.Body.TokenizedPrompt.TokenCount()
}

// tokenizationFailed distinguishes a FAILED tokenization from a legitimately empty prompt.
//
// The tokenized prompt is nil in both cases and neither surfaces an error to the scheduling path:
// the producer returns early on a render failure, and it also declines to assign when the result
// has zero tokens. So the field alone cannot separate them, and the two demand opposite handling.
// An empty prompt MUST still be scored, because the argmin still chooses a decode endpoint and an
// early return would throw that choice away. A failed tokenization must NOT be scored, because
// every term of the objective depends on the input length.
//
// The raw request body is the discriminator: if the caller sent prompt content but no tokens came
// back, tokenization failed.
func tokenizationFailed(request *fwksched.InferenceRequest) bool {
	if request == nil || request.Body == nil {
		return false
	}
	if request.Body.TokenizedPrompt != nil {
		return false
	}
	return rawPromptNonEmpty(request)
}

// rawPromptNonEmpty reports whether the caller supplied any prompt content, independent of
// tokenization. The body size is the cross-protocol fallback for shapes not inspected directly.
func rawPromptNonEmpty(request *fwksched.InferenceRequest) bool {
	body := request.Body
	if c := body.Completions; c != nil && !c.Prompt.IsEmpty() {
		return true
	}
	if cc := body.ChatCompletions; cc != nil {
		for i := range cc.Messages {
			if cc.Messages[i].Content.PlainText() != "" {
				return true
			}
		}
	}
	return request.RequestSizeBytes > 0
}

// sloClassOf reads a request's SLO class from the inference-objective header.
//
// FORWARD COMPATIBILITY ONLY. This does not address the per-workload threshold problem: every
// registered cohort declares the same single SLO class, so the header carries one constant string
// across all three workloads and cannot distinguish them. Thresholds varied per campaign
// invocation, not per class.
//
// The two-value read is mandatory. The accessor returns ("", false) when the header is absent, and
// letting an empty string fall through into a map lookup would miss every entry and resolve to an
// all-zero triple, which flattens the policy rather than loosening it.
//
// Read-only by design: the request's header map is the same reference that reaches the outgoing
// proxy path, so writing to it here would alter the request on the wire.
func sloClassOf(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	class, ok := metadata.GetLowerCaseHeaderValue(request.Headers, metadata.ObjectiveKey)
	if !ok {
		return ""
	}
	return class
}

// buildCandidates reconstructs each endpoint's routing view and partitions candidates by role.
//
// An endpoint whose GPU type is absent or unrecognised is EXCLUDED rather than scored under
// defaulted coefficients, and every exclusion is counted: heterogeneity rides the per-iteration
// intercept, so defaulting would be wrong on every decision rather than only under load. Without
// the counter that removal would be invisible and would look like a routing preference.
func (p *Picker) buildCandidates(
	ctx context.Context, scoredEndpoints []*fwksched.ScoredEndpoint, promptTokens int,
) (decodes, prefills []*endpointState, handles map[string]fwksched.Endpoint) {
	logger := log.FromContext(ctx)
	handles = make(map[string]fwksched.Endpoint, len(scoredEndpoints))
	reject := func(reason string) {
		candidateRejectionCount.WithLabelValues(p.typedName.Name, p.typedName.Type, reason).Inc()
	}

	for _, se := range scoredEndpoints {
		ep := unwrap(se.Endpoint)
		meta := ep.GetMetadata()
		if meta == nil {
			reject(rejectMissingMetrics)
			continue
		}
		id := meta.GetNamespacedName().String()

		metrics := ep.GetMetrics()
		if metrics == nil {
			reject(rejectMissingMetrics)
			logger.V(logutil.DEBUG).Info("Endpoint has no scraped metrics", "plugin", p.typedName, "endpoint", id)
			continue
		}

		gpuType := meta.Labels[p.cfg.GPUTypeLabelKey]
		theta, err := p.cfg.coeffsFor(gpuType)
		if err != nil {
			if gpuType == "" {
				reject(rejectMissingGPULabel)
			} else {
				reject(rejectUnknownGPUType)
			}
			logger.Error(err, "Excluding endpoint rather than defaulting its GPU coefficients",
				"plugin", p.typedName, "endpoint", id, "label", p.cfg.GPUTypeLabelKey, "value", gpuType)
			continue
		}

		role := meta.Labels[p.cfg.RoleLabelKey]
		isDecode, isPrefill := poolMembership(role)
		if !isDecode && !isPrefill {
			reject(rejectUnknownRole)
			logger.V(logutil.DEBUG).Info("Endpoint carries no recognised role label",
				"plugin", p.typedName, "endpoint", id, "label", p.cfg.RoleLabelKey, "value", role)
			continue
		}
		// An unlabelled or dual-role endpoint is admitted to match the stock filters, but it means
		// the fleet is not the clean single-role shape the deployment assumes, so record it.
		if role == "" || (isDecode && isPrefill) {
			logger.Info("Endpoint is not single-role; pool membership follows the stock role filters",
				"plugin", p.typedName, "endpoint", id, "label", p.cfg.RoleLabelKey, "value", role,
				"decodeCandidate", isDecode, "prefillCandidate", isPrefill)
		}

		// KV geometry absent. The scrape resets the block size on every cycle and repopulates it
		// only when the engine's cache-config info metric is present and parses, so a zero is
		// reachable through a renamed label, a scrape landing before that metric appears, or a
		// non-vLLM mapping.
		//
		// A zero must not pass through. The derived resident-token count would be 0, removing the
		// C1*KV term from both the baseline and the post-join iteration time: at these coefficients
		// and a 400,000-token occupancy that is about 19,000 us against a roughly 17,000 us
		// baseline iteration, so it more than doubles the iteration time and re-prices every
		// completion projection on that endpoint, which the argmin then systematically prefers. A
		// zero block count additionally zeroes the free-block estimate, closing the estimator's
		// fast path on that endpoint alone.
		//
		// Excluded for the same reason an unknown GPU type is: this policy does not score an
		// endpoint whose physics it cannot derive.
		if metrics.CacheBlockSize <= 0 || metrics.CacheNumBlocks <= 0 {
			reject(rejectMissingKVGeometry)
			logger.Error(nil, "Endpoint reports no KV block geometry; the C1*KV term cannot be derived and would read zero",
				"plugin", p.typedName, "endpoint", id,
				"cacheBlockSize", metrics.CacheBlockSize, "cacheNumBlocks", metrics.CacheNumBlocks)
			continue
		}

		// The configured block size and the engine's must agree: the admission test compares
		// resident blocks accumulated in the scraped unit against a need expressed in the
		// configured one, so a mismatch is wrong by their ratio. Excluded rather than reconciled,
		// because a port with two quantities that merely convert has a latent unit bug.
		if metrics.CacheBlockSize != p.cfg.BlockSize {
			reject(rejectBlockSizeMismatch)
			blockSizeMismatchCount.WithLabelValues(p.typedName.Name, p.typedName.Type).Inc()
			logger.Error(nil, "Endpoint KV block size disagrees with the configured blockSize; excluding it rather than converting between two units",
				"plugin", p.typedName, "endpoint", id, "configured", p.cfg.BlockSize, "scraped", metrics.CacheBlockSize)
			continue
		}

		state := &endpointState{
			id:              id,
			theta:           theta,
			batchSize:       metrics.RunningRequestsSize,
			queueDepth:      metrics.WaitingQueueSize,
			kvTokensInUse:   kvTokensInUse(metrics.KVCacheUsagePercent, metrics.CacheBlockSize, metrics.CacheNumBlocks),
			freeKVBlocks:    freeKVBlocks(metrics.KVCacheUsagePercent, metrics.CacheNumBlocks),
			engineBlockSize: int64(metrics.CacheBlockSize),
			maxBatchSize:    p.cfg.MaxBatchSize,
		}
		state.ap, state.prefixObserved = p.uncachedPromptTokens(ep, promptTokens)
		if !state.prefixObserved {
			prefixUnobservedCount.WithLabelValues(p.typedName.Name, p.typedName.Type).Inc()
			logger.V(logutil.DEBUG).Info("No prefix residency for endpoint; the uncached prompt length falls back to the full prompt, which over-prices its prefill",
				"plugin", p.typedName, "endpoint", id, "dataKey", p.prefixKey.String())
		}

		if isDecode {
			state.runningDecode, state.runningPrefill = p.shadow.residents(id, p.cfg, p.means.mean)
		}
		if isPrefill {
			poolOccupants := p.shadow.prefillOccupants(id, p.cfg, p.means.mean)
			if isDecode {
				state.runningPrefill = append(state.runningPrefill, poolOccupants...)
			} else {
				state.runningPrefill = poolOccupants
			}
		}
		state.residentPrefillTokens = p.cfg.residentPrefillTokens(state.runningPrefill)
		// Actual scheduler grants and their pre-chunk contexts are not exported by
		// this target, so U_pf cannot be reconstructed for a resident mixed batch.
		// The controlled experiment rejects those snapshots below.
		state.residentPrefillAttentionUnits = 0

		decodeEligible := isDecode && p.cfg.decodeIterationLawObservable(state)
		prefillEligible := isPrefill && p.cfg.prefillIterationLawObservable(state)
		if isDecode && !decodeEligible {
			reject(rejectDecodeLawUnobservable)
			logger.V(logutil.DEBUG).Info("Excluding decode endpoint whose aggregates are not decode-only",
				"plugin", p.typedName, "endpoint", id, "runningMetric", state.batchSize,
				"decodeShadow", len(state.runningDecode), "prefillShadow", len(state.runningPrefill))
		}
		if isPrefill && !prefillEligible {
			reject(rejectPrefillLawUnobservable)
			logger.V(logutil.DEBUG).Info("Excluding prefiller that is not idle at the controlled decision",
				"plugin", p.typedName, "endpoint", id, "runningMetric", state.batchSize,
				"waitingMetric", state.queueDepth, "prefillShadow", len(state.runningPrefill))
		}
		if !decodeEligible && !prefillEligible {
			continue
		}

		handles[id] = ep
		if decodeEligible {
			decodes = append(decodes, state)
		}
		if prefillEligible {
			prefills = append(prefills, state)
		}
	}
	return decodes, prefills, handles
}

// unwrap recovers the underlying endpoint so the wrapper never escapes this plugin.
func unwrap(ep fwksched.Endpoint) fwksched.Endpoint {
	if ce, ok := ep.(*candidateEndpoint); ok {
		return ce.Endpoint
	}
	return ep
}

// uncachedPromptTokens is the arriving request's uncached prompt-token count ON THIS ENDPOINT.
//
// Evaluated per endpoint because prefix residency differs per endpoint; a single per-request value
// would erase the cache-affinity signal. NOT floored at zero: a non-positive value means "fully
// cached", which the chunk terms treat as no prefill work, so the sign must survive to that check.
//
// The block size comes from the match info, NOT from the engine's KV block size. The prefix
// producer carries its own independently configurable block size, clamped up to a floor well above
// the engine's, so using the engine's would overstate cached tokens by their ratio.
//
// A miss means "no information", which is not "nothing cached", so the estimate falls back to the
// full prompt. That over-prices prefill here rather than asserting a cold cache as fact.
func (p *Picker) uncachedPromptTokens(ep fwksched.Endpoint, promptTokens int) (int, bool) {
	if p.cfg.PrefixCachingDisabled {
		return promptTokens, true
	}
	raw, ok := ep.Get(p.prefixKey.String())
	if !ok || raw == nil {
		return promptTokens, false
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil {
		return promptTokens, false
	}
	// CachedBlockCount is the literal cached-block count. MatchBlocks is a device-tier-weighted
	// ranking score at most equal to it, so using MatchBlocks would overstate cached tokens and
	// under-price prefill work.
	return promptTokens - info.CachedBlockCount()*info.BlockSizeTokens(), true
}
