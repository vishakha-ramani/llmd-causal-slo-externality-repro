package causalsloexternality

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Coeffs are the E3 latency-law coefficients for one GPU type. Units: AlphaD and
// AlphaP microseconds per iteration, C0 us/request, C1 and CPf us/token, CAttn
// us/token^2.
type Coeffs struct {
	AlphaD float64 `json:"alphaDUs"`
	AlphaP float64 `json:"alphaPUs"`
	C0     float64 `json:"c0UsPerReq"`
	C1     float64 `json:"c1UsPerToken"`
	CPf    float64 `json:"cPfUsPerToken"`
	CAttn  float64 `json:"cAttnUsPerUnit"`
}

// SLO holds the thresholds the value kernel evaluates against, in microseconds. A
// non-positive threshold disables that dimension's factor, contributing exactly 1.
type SLO struct {
	TauTTFTUs float64 `json:"tauTTFTUs"`
	TauITLUs  float64 `json:"tauITLUs"`
	TauE2EUs  float64 `json:"tauE2EUs"`
}

// profileNames carries the role-blind profile this policy runs plus the two keys
// ProcessResults splits the chosen pair into. Decode and Prefill are result keys in
// SchedulingResult.ProfileResults, not necessarily configured profiles.
type profileNames struct {
	Joint   string `json:"joint"`
	Decode  string `json:"decode"`
	Prefill string `json:"prefill"`
}

// Config is the plugin's parameter set. Every field the campaign treats as a knob is
// here rather than a Go literal, so an arm can be reproduced from config alone.
type Config struct {
	// V is the penalty multiplier. With NoCapacity it is a common positive factor
	// that cannot reorder the argmin, but the ablation cohort's validity gate asserts
	// score == V*(externality-ownGood) exactly, so folding it away would make that
	// check unreproducible.
	V float64 `json:"v"`

	// SLOTargets holds every workload's threshold triple, keyed by workload name, and is the
	// single source for both the arriving request's SLO and every resident's. There is
	// deliberately no second flat triple alongside it, because two sources could drift.
	//
	// An all-zero triple does not loosen the policy, it FLATTENS it. Each factor of the value
	// kernel is guarded on its threshold being positive, so an all-zero triple makes the value
	// identically 1, every charge 1-1 = 0, the externality 0 on every candidate, and the argmin
	// a tie broken by enumeration order. An unresolvable key would produce exactly that, which
	// is why resolution is validated at construction instead of falling back to a zero triple.
	SLOTargets map[string]SLO `json:"sloTargets"`

	// ActiveSLOTargets names the SLOTargets entry this deployment runs under.
	//
	// It exists because the thresholds are per WORKLOAD while the deployed configuration is per
	// PACKAGE: run assembly writes one scenario per package and emits pipeline runs per
	// (workload, package) with no per-workload override hook. So exactly one triple is live per
	// assemble, and producing another workload's run means changing this one value and
	// re-assembling into a separate run.
	//
	// A request-supplied class key overrides this selector only when it names an existing
	// entry. See resolveSLO.
	ActiveSLOTargets string `json:"activeSLOTargets"`

	// ChunkTokens must equal the engine's --max-num-batched-tokens; it sets every
	// chunk count and every overlap window.
	ChunkTokens int `json:"chunkTokens"`

	// BlockSize must equal the engine's --block-size. It is validated against the
	// scraped Metrics.CacheBlockSize, because the admission test compares accumulated
	// resident blocks against a need expressed in this unit and a mismatch is wrong by
	// their ratio in whichever direction it runs.
	BlockSize int `json:"blockSize"`

	// MaxBatchSize must equal the engine's --max-num-seqs. No metric carries it.
	// A zero disables the admission estimator's zero-wait fast path and makes its
	// free-slot count start negative, forcing the wave form on every decision.
	MaxBatchSize int64 `json:"maxBatchSize"`

	// RequireNoResidentPrefill restricts the experiment to endpoint snapshots for
	// which the exported aggregate running-request and KV signals are decode-only.
	// The target does not export actual per-iteration prefill grants or their
	// attention basis, so mixed resident-prefill snapshots cannot be evaluated from
	// the INFOCOM sufficient statistics.
	RequireNoResidentPrefill bool `json:"requireNoResidentPrefill"`

	// PrefixCachingDisabled makes every prompt token uncached. It must match an
	// engine launched without prefix caching; otherwise the approximate prefix
	// producer can report reuse that the engine will not perform.
	PrefixCachingDisabled bool `json:"prefixCachingDisabled"`

	// OutputTokenProcessingUs is per-output-token post-processing outside the
	// calibrated GPU-step coefficients, added to every TTFT projection. It does not
	// cancel between a local and a remote candidate, so a zero shifts the boundary.
	OutputTokenProcessingUs float64 `json:"outputTokenProcessingUs"`

	// Size-aware KV transfer. XferBaseUs and XferBandwidthGBps are the simulator's
	// executor settings, not target measurements: they must be measured on the actual
	// interconnect or every disaggregated candidate is mispriced in one direction.
	CXferSizeAware        bool    `json:"cXferSizeAware"`
	XferBaseUs            float64 `json:"xferBaseUs"`
	XferBandwidthGBps     float64 `json:"xferBandwidthGBps"`
	KVBytesPerTokenPerGPU float64 `json:"kvBytesPerTokenPerGPU"`

	// GPUTypeLabelKey names the pod label carrying the GPU type, and Coefficients is
	// keyed by that label's value. An endpoint whose label is absent or unrecognised is
	// rejected, never defaulted: heterogeneity rides the per-iteration intercept, so a
	// wrong coefficient set is wrong on every iteration regardless of KV state.
	GPUTypeLabelKey string            `json:"gpuTypeLabelKey"`
	Coefficients    map[string]Coeffs `json:"coefficients"`

	// RoleLabelKey names the pod label used to partition the role-blind candidate slice
	// into the decode and prefill pools.
	RoleLabelKey string `json:"roleLabelKey"`

	// PrefixMatchInfoProducerName binds the uncached-prompt-length input to a specific
	// prefix producer. A DataKey's identity includes its producer name, so this string
	// determines which producer's attribute is visible.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`

	Profiles profileNames `json:"profiles"`

	// Ablation switches. The focal arm is NoCapacity only.
	//
	// Decomposed truncates the decode candidate list to its first element, which is
	// the stock decode-first decomposition. It is a different algorithm with a measured
	// cost, not a tuning knob.
	NoCapacity    bool `json:"noCapacity"`
	NoExternality bool `json:"noExternality"`
	NoOwnGood     bool `json:"noOwnGood"`
	Decomposed    bool `json:"decomposed"`
}

// defaultConfig seeds the decoder so unset keys keep sane values. Coefficients is
// deliberately nil: json.Decode merges into a pre-populated map rather than replacing
// it, so a seeded coefficient set could not be overridden wholesale.
var defaultConfig = Config{
	V:                           8.0,
	ChunkTokens:                 2048,
	BlockSize:                   16,
	MaxBatchSize:                256,
	OutputTokenProcessingUs:     45,
	CXferSizeAware:              true,
	RoleLabelKey:                defaultRoleLabelKey,
	PrefixMatchInfoProducerName: defaultPrefixProducerName,
	Profiles: profileNames{
		Joint:   defaultJointProfile,
		Decode:  defaultDecodeProfile,
		Prefill: defaultPrefillProfile,
	},
	// SLOTargets is deliberately nil: json.Decode MERGES into a pre-populated map rather than
	// replacing it, so a seeded triple set could not be overridden wholesale.
	SLOTargets:   nil,
	NoCapacity:   true,
	Coefficients: nil,
}

// resolveSLO is the single SLO resolution point. It is called with a RESIDENT's class
// for a resident charge and with the arriving request's class for the own-good term:
// resident thresholds come from the resident's own class, because a co-resident
// meeting or missing its own SLO is what defines the goodput destroyed.
//
// The class parameter is retained rather than dropped even though every registered
// cell is single-class. Dropping it would make a second SLO class silently resolve
// every resident against the wrong targets.
// A request-supplied class overrides the configured selector ONLY when it names an existing
// entry. An absent, empty or unrecognised class falls back to the selector rather than missing
// the map, because a miss would resolve to a zero triple and flatten the policy.
//
// This override is forward compatibility, not a fix for the per-workload threshold problem: every
// registered cohort declares the same single SLO class, so the class key cannot distinguish the
// three workloads. Thresholds varied per campaign invocation, not per class.
func (cfg *Config) resolveSLO(class string) SLO {
	if class != "" {
		if slo, ok := cfg.SLOTargets[class]; ok {
			return slo
		}
	}
	return cfg.SLOTargets[cfg.ActiveSLOTargets]
}

// classOverridesSelector reports whether a class key would resolve to its own entry rather than
// falling back to the selector. Callers use it to log the fallback once per request instead of
// silently discarding an unrecognised class.
func (cfg *Config) classOverridesSelector(class string) bool {
	if class == "" {
		return false
	}
	_, ok := cfg.SLOTargets[class]
	return ok
}

// activeSLO is the triple this deployment runs under, ignoring any per-request class.
func (cfg *Config) activeSLO() SLO { return cfg.SLOTargets[cfg.ActiveSLOTargets] }

// validate rejects a configuration that would run but misprice. Every check here
// corresponds to a documented failure mode whose symptom is a plausible-looking score
// rather than an error.
func (cfg *Config) validate() error {
	if cfg.V <= 0 {
		return fmt.Errorf("v must be positive, got %v", cfg.V)
	}
	if cfg.ChunkTokens <= 0 {
		return fmt.Errorf("chunkTokens must be positive (engine --max-num-batched-tokens), got %d", cfg.ChunkTokens)
	}
	if cfg.BlockSize <= 0 {
		return fmt.Errorf("blockSize must be positive (engine --block-size), got %d", cfg.BlockSize)
	}
	// A zero MaxBatchSize is the one default that silently converts a tail-behaviour
	// degradation into an always-on one, by making the estimator's fast path
	// unreachable and its free-slot count start negative.
	if cfg.MaxBatchSize <= 0 {
		return fmt.Errorf("maxBatchSize must be positive (engine --max-num-seqs); a zero forces the wave-form admission estimate on every decision, got %d", cfg.MaxBatchSize)
	}
	if cfg.GPUTypeLabelKey == "" {
		return errors.New("gpuTypeLabelKey is required: coefficients are selected per endpoint by GPU type and must never be defaulted")
	}
	if cfg.RoleLabelKey == "" {
		return errors.New("roleLabelKey is required to partition candidates into decode and prefill pools")
	}
	if len(cfg.Coefficients) == 0 {
		return fmt.Errorf("coefficients must declare at least one GPU type keyed by the value of label %q", cfg.GPUTypeLabelKey)
	}
	// Strict decoding rejects unknown struct fields but NOT unknown map keys, so a
	// typo'd GPU label would decode cleanly and then never match an endpoint. Validate
	// each entry's contents instead.
	for _, label := range sortedKeys(cfg.Coefficients) {
		if err := cfg.Coefficients[label].validate(label); err != nil {
			return err
		}
	}
	if len(cfg.SLOTargets) == 0 {
		return errors.New("sloTargets must declare at least one threshold triple keyed by workload name")
	}
	for _, key := range sortedKeys(cfg.SLOTargets) {
		slo := cfg.SLOTargets[key]
		if err := slo.validate(fmt.Sprintf("sloTargets[%q]", key)); err != nil {
			return err
		}
	}
	// FAIL LOUD on a selector that names nothing. Resolving to the zero value instead would make
	// the value kernel return 1 for every resident, every charge 1-1 = 0, and the argmin a tie
	// broken by enumeration order: the policy would keep running and keep reporting goodput while
	// having stopped discriminating entirely.
	if cfg.ActiveSLOTargets == "" {
		return fmt.Errorf("activeSLOTargets is required and must name one of sloTargets [%s]",
			strings.Join(sortedKeys(cfg.SLOTargets), ", "))
	}
	if _, ok := cfg.SLOTargets[cfg.ActiveSLOTargets]; !ok {
		return fmt.Errorf("activeSLOTargets %q names no entry in sloTargets [%s]; a missing entry would resolve to an all-zero triple and flatten the policy rather than loosen it",
			cfg.ActiveSLOTargets, strings.Join(sortedKeys(cfg.SLOTargets), ", "))
	}
	if cfg.CXferSizeAware {
		if cfg.XferBandwidthGBps <= 0 {
			return fmt.Errorf("xferBandwidthGBps must be positive when cXferSizeAware is set, got %v", cfg.XferBandwidthGBps)
		}
		if cfg.KVBytesPerTokenPerGPU <= 0 {
			return fmt.Errorf("kvBytesPerTokenPerGPU must be positive when cXferSizeAware is set, got %v", cfg.KVBytesPerTokenPerGPU)
		}
	}
	// The capacity account is not reconstructed, so the term cannot be computed. Rejecting at
	// construction makes the EPP fail at configuration load, which is the loudest failure
	// available. A scoring-time sentinel would be worse than useless here: every comparison
	// involving NaN is false in Go, so a NaN score never wins the strict argmin comparison and the
	// first enumerated candidate would silently take every decision.
	if !cfg.NoCapacity {
		return errors.New("noCapacity must be true: the capacity account this term reads is not reconstructed on this target, so the capacity contribution cannot be computed")
	}
	if cfg.Profiles.Joint == "" || cfg.Profiles.Decode == "" || cfg.Profiles.Prefill == "" {
		return errors.New("profiles.joint, profiles.decode and profiles.prefill must all be non-empty")
	}
	if cfg.Profiles.Decode == cfg.Profiles.Prefill {
		return fmt.Errorf("profiles.decode and profiles.prefill must differ, both are %q", cfg.Profiles.Decode)
	}
	return cfg.validateSLOAgainstCoeffs()
}

// validateSLOAgainstCoeffs enforces tau_itl > AlphaD on every configured GPU type. An
// ITL target at or below the per-iteration intercept is physically unachievable on
// that hardware, so the pairing indicates either a mis-set target or coefficients
// fitted on different hardware than the fleet actually runs.
//
// The focal arm's score does not read tau_itl, so this cannot change a decision. It
// is validated anyway because it is the cheapest check that the coefficient set and
// the SLO triple describe the same deployment.
func (cfg *Config) validateSLOAgainstCoeffs() error {
	check := func(where string, slo SLO) error {
		if slo.TauITLUs <= 0 {
			return nil
		}
		for _, label := range sortedKeys(cfg.Coefficients) {
			if alphaD := cfg.Coefficients[label].AlphaD; slo.TauITLUs <= alphaD {
				return fmt.Errorf("%s tauITLUs %v is not achievable on GPU type %q whose per-iteration intercept alphaDUs is %v",
					where, slo.TauITLUs, label, alphaD)
			}
		}
		return nil
	}
	// Only the ACTIVE triple is checked. The other workloads' triples are carried in the same
	// config for the operator's convenience, and reasoning and deep research both set a 100 ms ITL
	// target that an A100's 25.6 ms intercept clears comfortably; but a coefficient set steep
	// enough to fail an inactive workload must not block a run of the active one.
	return check(fmt.Sprintf("sloTargets[%q]", cfg.ActiveSLOTargets), cfg.activeSLO())
}

// validate rejects an SLO triple that is entirely disabled. All-zero thresholds make
// the value kernel return 1 for every resident at every projected completion, so every
// charge is zero, the externality is zero on every candidate, and the policy silently
// stops discriminating.
func (s SLO) validate(where string) error {
	if s.TauTTFTUs < 0 || s.TauITLUs < 0 || s.TauE2EUs < 0 {
		return fmt.Errorf("%s thresholds must not be negative, got %+v", where, s)
	}
	if s.TauTTFTUs <= 0 && s.TauE2EUs <= 0 {
		return fmt.Errorf("%s has neither tauTTFTUs nor tauE2EUs set; the value kernel would return 1 for every candidate and the argmin would degenerate to enumeration order", where)
	}
	return nil
}

// validate rejects a coefficient set that cannot have been fitted on one device.
func (c Coeffs) validate(label string) error {
	if c.AlphaD <= 0 || c.AlphaP <= 0 {
		return fmt.Errorf("coefficients[%q]: alphaDUs and alphaPUs must be positive, got %v and %v", label, c.AlphaD, c.AlphaP)
	}
	if c.C0 < 0 || c.C1 < 0 || c.CPf < 0 || c.CAttn < 0 {
		return fmt.Errorf("coefficients[%q]: c0UsPerReq, c1UsPerToken, cPfUsPerToken and cAttnUsPerUnit must not be negative, got %+v", label, c)
	}
	// Prefill and decode share the per-iteration intercept. More than 10 percent
	// divergence means the two halves were fitted on mismatched hardware.
	if rel := math.Abs(c.AlphaD-c.AlphaP) / c.AlphaD; rel > maxAlphaDivergence {
		return fmt.Errorf("coefficients[%q]: |alphaDUs-alphaPUs|/alphaDUs is %.4f, above %.2f, which indicates the decode and prefill halves were fitted on different hardware",
			label, rel, maxAlphaDivergence)
	}
	return nil
}

// coeffsFor resolves an endpoint's coefficient set by GPU type. It returns an error on
// an absent or unrecognised label instead of falling back, and names the configured
// types so the mismatch is actionable.
func (cfg *Config) coeffsFor(gpuType string) (Coeffs, error) {
	if gpuType == "" {
		return Coeffs{}, fmt.Errorf("endpoint carries no %q label; refusing to default GPU coefficients", cfg.GPUTypeLabelKey)
	}
	theta, ok := cfg.Coefficients[gpuType]
	if !ok {
		return Coeffs{}, fmt.Errorf("no coefficients configured for GPU type %q (label %q); configured types are [%s]",
			gpuType, cfg.GPUTypeLabelKey, strings.Join(sortedKeys(cfg.Coefficients), ", "))
	}
	return theta, nil
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
