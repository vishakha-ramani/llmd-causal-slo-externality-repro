package causalsloexternality

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// decisionLocal is a placement whose prefill runs on the chosen decode endpoint.
	decisionLocal = "local"
	// decisionDisagg is a placement whose prefill runs on a separate pool endpoint.
	decisionDisagg = "disaggregated"
)

var (
	// ttftEstimatorSubstitutionCount records that the published TTFT estimator could not run and
	// the closed-form substitute was used instead.
	//
	// THIS COUNTER IS A REQUIREMENT, NOT INSTRUMENTATION. Every goodput margin this policy is
	// measured against was produced with a scheduler-rollout TTFT model whose per-request
	// wait-queue inputs do not exist on this target. Upstream falls back to the closed form
	// SILENTLY. A port that also stays silent is running a different estimator than the one that
	// produced the published numbers with nothing recording the difference. The fallback is
	// acceptable; the silence is the defect.
	ttftEstimatorSubstitutionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_ttft_estimator_substitution_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total candidate scorings that substituted the closed-form rollforward admission estimator "+
					"because the engine scheduler queue contents required by the published rollout estimator are unobservable",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// tokenizationFailureCount records requests refused because no tokenized prompt was available.
	//
	// The input length is the single most load-bearing scalar in the objective: it enters the
	// prefill demand quadratically and also drives the post-join iteration time, the KV need and
	// the transfer cost. Data-producer errors are swallowed upstream, so without this counter a
	// renderer that is down would show up only as an absence of placements.
	tokenizationFailureCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_tokenization_unavailable_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total requests the joint argmin refused to score because the tokenized prompt was absent",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// prefixUnobservedCount records candidates scored without prefix-residency information.
	//
	// Nothing is rejected: a miss means "no information", which is not "nothing cached", so the
	// uncached length falls back to the full prompt. On a well-cached endpoint that is the
	// difference between no prefill work and the entire prompt, so it re-prices the candidate and
	// moves the local/remote boundary while leaving it in the argmin. Counted rather than silent,
	// for the same reason the rejections are.
	prefixUnobservedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_prefix_unobserved_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total candidate endpoints scored with no prefix-cache residency information, so the uncached prompt length fell back to the full prompt",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// placementDecisionCount records the joint argmin's outcome.
	placementDecisionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_placement_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total placements chosen by the joint causal-SLO-externality argmin, by placement kind",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "model_name", "placement"},
	)

	// candidateRejectionCount records candidates dropped before scoring, by reason. The GPU-type
	// reasons matter most: coefficients are never defaulted, so a mislabelled endpoint leaves the
	// candidate set entirely rather than being scored under the wrong physics. Without this
	// counter that removal would be invisible and would look like a routing preference.
	candidateRejectionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_candidate_rejected_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total candidate endpoints excluded before scoring, by reason",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "reason"},
	)

	// blockSizeMismatchCount records a scrape whose engine KV block size disagrees with the
	// configured one. The two are compared rather than reconciled: upstream has a single block
	// size, and silently converting between two would leave a latent unit bug in the admission
	// test.
	blockSizeMismatchCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_block_size_mismatch_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total endpoints whose scraped KV block size disagreed with the configured blockSize",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// argminDurationSeconds records how long the cross-product argmin itself takes.
	//
	// The policy adds EPP-side work inside the request path, and the request path is inside the
	// TTFT this experiment measures. That contribution must be quantifiable after a run rather
	// than argued about. It covers the argmin only; the framework separately records per-plugin
	// extension-point latency, and the tokenizer's own round trip is the producer's to report.
	argminDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_argmin_duration_seconds",
			Help: metricsutil.HelpMsgWithStability(
				"Duration of the joint causal-SLO-externality argmin, including endpoint-state reconstruction",
				compbasemetrics.ALPHA),
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// shadowTableSize reports how many in-flight requests the resident estimates are drawn from.
	// A value that sits near zero under load means the resident populations are empty and every
	// externality is zero, which is the policy silently degrading to its own tie-break.
	shadowTableSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_shadow_table_requests",
			Help: metricsutil.HelpMsgWithStability(
				"In-flight requests tracked in the EPP-side shadow table that supplies per-resident estimates",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	// shadowTableReapedCount records entries dropped by the staleness sweep. Requests can
	// terminate without a final chunk, and the response hooks carry no termination state, so
	// without the sweep those entries would be charged as residents forever.
	shadowTableReapedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_shadow_table_reaped_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total shadow-table entries dropped because no response activity was observed within the staleness window",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)
)

// rejection reasons.
const (
	rejectMissingGPULabel        = "missing_gpu_label"
	rejectUnknownGPUType         = "unknown_gpu_type"
	rejectMissingMetrics         = "missing_metrics"
	rejectUnknownRole            = "unknown_role"
	rejectBlockSizeMismatch      = "block_size_mismatch"
	rejectMissingKVGeometry      = "missing_kv_geometry"
	rejectDecodeLawUnobservable  = "decode_iteration_law_unobservable"
	rejectPrefillLawUnobservable = "prefill_iteration_law_unobservable"
)

// registerMetrics registers this plugin's collectors. A nil registerer is tolerated, and
// re-registering the same collector is not an error, so several plugin instances may coexist.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{
		ttftEstimatorSubstitutionCount,
		tokenizationFailureCount,
		prefixUnobservedCount,
		argminDurationSeconds,
		placementDecisionCount,
		candidateRejectionCount,
		blockSizeMismatchCount,
		shadowTableSize,
		shadowTableReapedCount,
	} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register causal-slo-externality metric: %w", err)
		}
	}
	return nil
}
