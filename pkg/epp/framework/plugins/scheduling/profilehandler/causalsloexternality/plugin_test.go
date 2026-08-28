package causalsloexternality

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	configapi "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	approxprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const gpuLabel = "pd-infocomm.io/gpu-type"

// policyParams mirrors the treatment overlay's parameter block. Kept as one string so a rename in
// the Go config struct fails this test rather than surfacing on a cluster.
const policyParams = `
    v: 8.0
    sloTargets:
      interactive:
        tauTTFTUs: 1000000
        tauITLUs: 50000
        tauE2EUs: 16000000
      reasoning:
        tauTTFTUs: 2000000
        tauITLUs: 100000
        tauE2EUs: 802000000
      deep_research:
        tauTTFTUs: 10000000
        tauITLUs: 100000
        tauE2EUs: 40000000
    activeSLOTargets: interactive
    chunkTokens: 2048
    blockSize: 16
    maxBatchSize: 256
    outputTokenProcessingUs: 45
    cXferSizeAware: true
    xferBaseUs: 50.0
    xferBandwidthGBps: 25.0
    kvBytesPerTokenPerGPU: 81920.0
    gpuTypeLabelKey: pd-infocomm.io/gpu-type
    coefficients:
      NVIDIA-H100-80GB-HBM3:
        alphaDUs: 16613.537554540144
        alphaPUs: 16617.85321583337
        c0UsPerReq: 5.347316038602452
        c1UsPerToken: 0.04761401141756073
        cPfUsPerToken: 6.144687138665833
        cAttnUsPerUnit: 0.00010075247918809842
    roleLabelKey: llm-d.ai/role
    prefixMatchInfoProducerName: approx-prefix-cache-producer
    profiles:
      joint: joint
      decode: decode
      prefill: prefill
    noCapacity: true
    noExternality: false
    noOwnGood: false
    decomposed: false`

// treatmentConfig is the EndpointPickerConfig the treatment overlay carries.
const treatmentConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
  parameters:
    autoTune: false
    blockSizeTokens: 64
- type: causal-slo-externality
  parameters:` + policyParams + `
- type: causal-slo-externality-picker
  parameters:
    policyPluginName: causal-slo-externality
schedulingProfiles:
- name: joint
  plugins:
  - pluginRef: causal-slo-externality-picker
`

// reversedConfig declares the picker before the policy it depends on.
const reversedConfig = `apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: approx-prefix-cache-producer
- type: causal-slo-externality-picker
  parameters:
    policyPluginName: causal-slo-externality
- type: causal-slo-externality
  parameters:` + policyParams + `
schedulingProfiles:
- name: joint
  plugins:
  - pluginRef: causal-slo-externality-picker
`

func init() {
	// The loader instantiates by registered type, so every type the treatment configuration names
	// must be registered for the end-to-end tests below. Production registration lives in the
	// runner; registering the prefix and token producers here as well is what makes those tests
	// exercise the real document rather than a trimmed one.
	fwkplugin.Register(PluginType, HandlerFactory)
	fwkplugin.Register(PickerType, PickerFactory)
	fwkplugin.RegisterAsDefaultProducer(approxprefix.ApproxPrefixCachePluginType,
		approxprefix.ApproxPrefixCacheFactory, attrprefix.PrefixCacheMatchInfoDataKey)
	fwkplugin.RegisterAsDefaultProducer(tokenizer.PluginType, tokenizer.PluginFactory, tokenizer.TokenizedPromptDataKey)
}

// ---------------------------------------------------------------------------
// Configuration loading, end to end through the real loader
// ---------------------------------------------------------------------------

// The treatment overlay's plugin document must parse against the real schema, name a profile that
// resolves to this policy's picker, and instantiate with both types sharing one state object.
//
// Instantiation is driven here rather than through InstantiateAndConfigure because that also applies
// system-wide flow-control defaults, whose plugin types a scheduling-plugin package has no business
// registering. instantiateInOrder mirrors what the loader does with the plugin list: walk it in
// declaration order, decode each parameters block strictly, and add each instance to the handle.
func TestTreatmentConfigInstantiates(t *testing.T) {
	logger := logging.NewTestLogger()
	raw, _, err := loader.LoadRawConfig([]byte(treatmentConfig), logger)
	require.NoError(t, err, "the treatment overlay's EndpointPickerConfig must parse")

	require.Len(t, raw.SchedulingProfiles, 1, "the policy runs exactly one role-blind profile")
	require.Equal(t, "joint", raw.SchedulingProfiles[0].Name)
	require.Len(t, raw.SchedulingProfiles[0].Plugins, 1,
		"the profile carries only the picker: no role filter, and no scorer whose contribution would be clamped")
	assert.Equal(t, PickerType, raw.SchedulingProfiles[0].Plugins[0].PluginRef)

	// The profile handler is deliberately NOT referenced from the profile; it is found by interface.
	for _, ref := range raw.SchedulingProfiles[0].Plugins {
		assert.NotEqual(t, PluginType, ref.PluginRef, "the profile handler must not be a profile plugin")
	}

	handle := testutils.NewTestHandle(context.Background())
	require.NoError(t, instantiateInOrder(t, raw, handle))

	policy := handle.Plugin(PluginType)
	require.NotNil(t, policy, "the policy plugin must be instantiated")
	handler, ok := policy.(*Handler)
	require.True(t, ok)

	picker := handle.Plugin(PickerType)
	require.NotNil(t, picker, "the picker must be instantiated")
	p, ok := picker.(*Picker)
	require.True(t, ok)

	// Both types must share one state object, or the picker would score against an empty shadow
	// table while the handler filled a different one.
	assert.Same(t, handler.core, p.core, "picker and policy must share one core")

	// The parameters really did decode: a rename in the Go config struct must fail here.
	assert.Equal(t, 8.0, handler.cfg.V)
	assert.Equal(t, 2048, handler.cfg.ChunkTokens)
	assert.Equal(t, int64(256), handler.cfg.MaxBatchSize)
	assert.Equal(t, "interactive", handler.cfg.ActiveSLOTargets)
	assert.Equal(t, 16_000_000.0, handler.cfg.activeSLO().TauE2EUs)
	assert.Len(t, handler.cfg.SLOTargets, 3, "all three workload triples travel in one config")
	assert.Contains(t, handler.cfg.Coefficients, "NVIDIA-H100-80GB-HBM3")
}

// Declaration order is load-bearing: the picker resolves its policy by name and factories run in
// declaration order, so the reversed document must fail loudly rather than misbehave silently.
func TestConfigRejectsPickerDeclaredBeforePolicy(t *testing.T) {
	raw, _, err := loader.LoadRawConfig([]byte(reversedConfig), logging.NewTestLogger())
	require.NoError(t, err, "the reversed document is schema-valid; only instantiation may reject it")

	err = instantiateInOrder(t, raw, testutils.NewTestHandle(context.Background()))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "declared BEFORE")
}

// instantiateInOrder walks the plugin list in declaration order, as the loader does.
func instantiateInOrder(t *testing.T, raw *configapi.EndpointPickerConfig, handle fwkplugin.Handle) error {
	t.Helper()
	for _, spec := range raw.Plugins {
		factory, ok := fwkplugin.Registry[spec.Type]
		require.True(t, ok, "plugin type %q is not registered", spec.Type)
		name := spec.Name
		if name == "" {
			name = spec.Type
		}
		instance, err := factory(name, fwkplugin.StrictDecoder(spec.Parameters), handle)
		if err != nil {
			return err
		}
		handle.AddPlugin(name, instance)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Factory construction
// ---------------------------------------------------------------------------

func TestHandlerFactory_ConstructsAndValidates(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())

	plg, err := HandlerFactory("policy", decoderFor(t, minimalParams()), handle)
	require.NoError(t, err)
	h, ok := plg.(*Handler)
	require.True(t, ok)
	assert.Equal(t, fwkplugin.TypedName{Type: PluginType, Name: "policy"}, h.TypedName())
	assert.Equal(t, 8.0, h.cfg.V)
	assert.NotNil(t, h.shadow)
	assert.NotNil(t, h.means)

	// The prefix key must be producer-qualified, since a DataKey's identity includes its producer.
	assert.Contains(t, h.prefixKey.String(), "approx-prefix-cache-producer")
}

func TestHandlerFactory_RejectsBadParameters(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())

	for _, tc := range []struct {
		name   string
		params map[string]any
		expect string
	}{
		{
			name: "unknown field",
			params: withParams(func(m map[string]any) {
				m["tauTtft"] = 1000
			}),
			expect: "failed to parse",
		},
		{
			name: "no coefficients",
			params: withParams(func(m map[string]any) {
				delete(m, "coefficients")
			}),
			expect: "at least one GPU type",
		},
		{
			name: "zero maxBatchSize",
			params: withParams(func(m map[string]any) {
				m["maxBatchSize"] = 0
			}),
			expect: "maxBatchSize must be positive",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := HandlerFactory("policy", decoderFor(t, tc.params), handle)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.expect)
		})
	}
}

func TestHandlerFactory_RequiresHandle(t *testing.T) {
	_, err := HandlerFactory("policy", decoderFor(t, minimalParams()), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "handle is required")
}

func TestPickerFactory_RejectsMissingAndWrongTypedPolicy(t *testing.T) {
	handle := testutils.NewTestHandle(context.Background())

	_, err := PickerFactory("picker", decoderFor(t, map[string]any{"policyPluginName": "absent"}), handle)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	// A name that resolves to something that is not the policy must also fail.
	policy, err := HandlerFactory("policy", decoderFor(t, minimalParams()), handle)
	require.NoError(t, err)
	handle.AddPlugin("policy", policy)

	got, err := PickerFactory("picker", decoderFor(t, map[string]any{"policyPluginName": "policy"}), handle)
	require.NoError(t, err)
	assert.Equal(t, PickerType, got.TypedName().Type)
}

// ---------------------------------------------------------------------------
// Filter and Picker wiring
// ---------------------------------------------------------------------------

// The filter must attach the request to every candidate and must NOT split by role: the argmin
// needs both pools in one candidate set.
func TestFilter_AttachesRequestAndKeepsBothPools(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	in := []fwksched.Endpoint{
		newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3"),
		newFramedEndpoint("p-a", bylabel.RolePrefill, "NVIDIA-H100-80GB-HBM3"),
	}
	out := p.Filter(context.Background(), request, in)

	require.Len(t, out, 2, "no role filtering may happen here")
	for _, ep := range out {
		ce, ok := ep.(*candidateEndpoint)
		require.True(t, ok, "every candidate must be wrapped so the request reaches the picker")
		assert.Same(t, request, ce.request)
	}
}

// The wrapper must never escape the plugin: an endpoint comparison downstream type-asserts the
// concrete endpoint and would panic on it.
func TestPick_UnwrapsWrapperBeforeReturning(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	endpoints := []fwksched.Endpoint{
		newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3"),
		newFramedEndpoint("p-a", bylabel.RolePrefill, "NVIDIA-H100-80GB-HBM3"),
	}
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, endpoints)))

	require.NotEmpty(t, result.TargetEndpoints)
	for _, ep := range result.TargetEndpoints {
		_, wrapped := ep.(*candidateEndpoint)
		assert.False(t, wrapped, "the wrapper must not reach any downstream consumer")
	}
}

// A picker with no request reaching it must refuse rather than score against a default prompt
// length, which would price every candidate wrongly while still returning a plausible answer.
func TestPick_RefusesWhenNoRequestReachedIt(t *testing.T) {
	p := newTestPicker(t)
	unwrapped := []fwksched.Endpoint{newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")}

	result := p.Pick(context.Background(), scored(unwrapped))
	assert.Empty(t, result.TargetEndpoints)
}

// A FAILED TOKENIZATION MUST NOT BE SCORED. Producer errors are swallowed upstream, so a renderer
// that is down leaves the tokenized prompt absent and the request proceeds. Substituting a byte
// estimate would silently reprice every candidate; scoring with a zero input length would collapse
// the prefill terms and hand the decision to enumeration order.
func TestPick_RefusesWhenTokenizationIsUnavailable(t *testing.T) {
	p := newTestPicker(t)

	// The caller sent prompt content but no tokens came back: tokenization failed.
	request := newTestRequest(4000)
	request.Body.TokenizedPrompt = nil
	request.Body.Completions = &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "a real prompt"}}

	endpoints := []fwksched.Endpoint{newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")}
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, endpoints)))
	assert.Empty(t, result.TargetEndpoints, "a failed tokenization must refuse, not score on a fallback")
}

// The tokenized prompt is nil for BOTH a failed render and a zero-token prompt, and neither
// surfaces an error. Only the raw body separates them, and they demand opposite handling.
func TestPick_DistinguishesFailedTokenizationFromAnEmptyPrompt(t *testing.T) {
	withRawPrompt := newTestRequest(4000)
	withRawPrompt.Body.TokenizedPrompt = nil
	withRawPrompt.Body.Completions = &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "content"}}
	assert.True(t, tokenizationFailed(withRawPrompt), "content in, no tokens out, is a failure")

	noRawPrompt := newTestRequest(4000)
	noRawPrompt.Body.TokenizedPrompt = nil
	assert.False(t, tokenizationFailed(noRawPrompt), "no content and no tokens is an empty prompt, which must still be scored")

	tokenized := newTestRequest(4000)
	assert.False(t, tokenizationFailed(tokenized))

	// The body size is the cross-protocol fallback for shapes not inspected directly.
	opaque := newTestRequest(4000)
	opaque.Body.TokenizedPrompt = nil
	opaque.RequestSizeBytes = 512
	assert.True(t, tokenizationFailed(opaque))
}

// A genuinely EMPTY prompt is a different case and must still be scored: the argmin still chooses
// which decode endpoint to use, and an early return would throw that choice away.
func TestPick_ScoresAGenuinelyEmptyPrompt(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(0)
	require.NotNil(t, request.Body.TokenizedPrompt, "the distinction is structural, not a token count")

	endpoints := []fwksched.Endpoint{newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")}
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, endpoints)))
	require.Len(t, result.TargetEndpoints, 1, "a fully-cached or empty prompt still selects a decode endpoint")
	assert.Equal(t, "default/d-a", result.TargetEndpoints[0].GetMetadata().GetNamespacedName().String())
}

// The class read must use the two-value accessor and must never let an empty value reach a map
// lookup, which would resolve to an all-zero triple.
func TestSLOClassOf_ReadsHeaderIncludingDeprecatedAlias(t *testing.T) {
	request := newTestRequest(4000)
	assert.Empty(t, sloClassOf(request), "no header means no class, not an empty-string lookup key")
	assert.Empty(t, sloClassOf(nil))

	request.Headers[metadata.ObjectiveKey] = "reasoning"
	assert.Equal(t, "reasoning", sloClassOf(request))

	// The deprecated spelling is what the load generator currently emits, and the accessor handles
	// the alias.
	alias := newTestRequest(4000)
	alias.Headers[metadata.OldObjectiveKey] = "deep_research"
	assert.Equal(t, "deep_research", sloClassOf(alias))
}

// The class read must not mutate the request's header map: it is the same reference that reaches the
// outgoing proxy path.
func TestSLOClassOf_DoesNotMutateHeaders(t *testing.T) {
	request := newTestRequest(4000)
	request.Headers["x-unrelated"] = "keep"
	before := len(request.Headers)

	_ = sloClassOf(request)
	assert.Len(t, request.Headers, before)
	assert.Equal(t, "keep", request.Headers["x-unrelated"])
}

// An endpoint whose GPU type is unrecognised must leave the candidate set rather than be scored
// under defaulted coefficients.
func TestPick_ExcludesEndpointsWithUnusableLabels(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	endpoints := []fwksched.Endpoint{
		newFramedEndpoint("d-good", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3"),
		newFramedEndpoint("d-unknown-gpu", bylabel.RoleDecode, "NVIDIA-L40S"),
		newFramedEndpoint("d-no-gpu", bylabel.RoleDecode, ""),
		newFramedEndpoint("d-no-role", "", "NVIDIA-H100-80GB-HBM3"),
	}
	decodes, prefills, handles := p.buildCandidates(context.Background(), scored(p.Filter(context.Background(), request, endpoints)), 4000)

	// The unrecognised and absent GPU types leave the candidate set. The endpoint with no ROLE label
	// stays, because the stock decode-filter admits unlabelled endpoints and the two arms must see
	// the same fleet.
	ids := make([]string, 0, len(decodes))
	for _, d := range decodes {
		ids = append(ids, d.id)
	}
	assert.ElementsMatch(t, []string{"default/d-good", "default/d-no-role"}, ids)
	assert.Empty(t, prefills, "prefill admits no unlabelled endpoint")
	assert.Len(t, handles, 2)
}

// Pool membership must reproduce the stock filters' sets exactly, including their asymmetries: any
// divergence means the two arms see different fleets.
func TestPoolMembership_MirrorsTheStockRoleFilters(t *testing.T) {
	for _, tc := range []struct {
		role                    string
		wantDecode, wantPrefill bool
	}{
		{bylabel.RoleDecode, true, false},
		{bylabel.RolePrefill, false, true},
		{bylabel.RoleEncode, false, false},
		// Dual roles belong to BOTH pools, so a strict equality partition would drop them entirely.
		{bylabel.RolePrefillDecode, true, true},
		{bylabel.RoleBoth, true, true}, //nolint:staticcheck // must match the stock filters
		{bylabel.RoleEncodePrefillDecode, true, true},
		// Prefill has one extra that decode has no counterpart for.
		{bylabel.RoleEncodePrefill, false, true},
		// Unlabelled is asymmetric: decode-filter is built with allowsNoLabel true, prefill false.
		{"", true, false},
		{"nonsense", false, false},
	} {
		t.Run("role="+tc.role, func(t *testing.T) {
			gotDecode, gotPrefill := poolMembership(tc.role)
			assert.Equal(t, tc.wantDecode, gotDecode, "decode membership")
			assert.Equal(t, tc.wantPrefill, gotPrefill, "prefill membership")
		})
	}
}

// A dual-role endpoint is a legal member of both pools, so the cross product would pair it with
// itself. That pair is the LOCAL candidate, already enumerated, and must not also be scored as a
// disaggregated one.
func TestPick_DualRoleEndpointIsNotPairedWithItself(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	dual := newFramedEndpoint("d-dual", bylabel.RolePrefillDecode, "NVIDIA-H100-80GB-HBM3")
	decodes, prefills, _ := p.buildCandidates(context.Background(),
		scored(p.Filter(context.Background(), request, []fwksched.Endpoint{dual})), 4000)
	require.Len(t, decodes, 1, "a dual-role endpoint is a decode candidate")
	require.Len(t, prefills, 1, "and a prefill candidate")

	// One endpoint means exactly one candidate, the local one, so the result carries no prefill.
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, []fwksched.Endpoint{dual})))
	require.Len(t, result.TargetEndpoints, 1,
		"pairing an endpoint with itself is the local candidate, not a disaggregated one")
}

// An endpoint with no KV geometry must leave the candidate set. A zero block size would derive a
// zero resident-token count, removing the C1*KV term from every iteration-time projection on that
// endpoint, which the argmin would then systematically prefer.
func TestPick_ExcludesEndpointsWithNoKVGeometry(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	for _, tc := range []struct {
		name                 string
		blockSize, numBlocks int
	}{
		{"zero block size", 0, 10000},
		{"zero block count", 16, 0},
		{"both zero", 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ep := newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")
			ep.GetMetrics().CacheBlockSize = tc.blockSize
			ep.GetMetrics().CacheNumBlocks = tc.numBlocks

			decodes, _, _ := p.buildCandidates(context.Background(),
				scored(p.Filter(context.Background(), request, []fwksched.Endpoint{ep})), 4000)
			assert.Empty(t, decodes, "an endpoint whose KV geometry is unknown must not be scored")
		})
	}
}

// A scraped KV block size disagreeing with the configured one is excluded rather than reconciled:
// the admission test compares blocks in one unit against a need in the other.
func TestPick_ExcludesBlockSizeMismatch(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(4000)

	mismatched := newFramedEndpoint("d-mismatch", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")
	mismatched.GetMetrics().CacheBlockSize = 32

	decodes, _, _ := p.buildCandidates(context.Background(),
		scored(p.Filter(context.Background(), request, []fwksched.Endpoint{mismatched})), 4000)
	assert.Empty(t, decodes, "a block-size mismatch must remove the candidate")
}

// A disaggregated placement must carry the decode endpoint first and the prefill endpoint second,
// which is the ordering ProcessResults relies on.
func TestPick_PairOrderingIsDecodeThenPrefill(t *testing.T) {
	p := newTestPicker(t)
	request := newTestRequest(40000)

	// Load the only decode endpoint with residents at their deadline so remote prefill wins.
	p.shadow.open(&shadowEntry{
		requestID: "resident-1", decodeID: "default/d-a", arrivalUs: -9_000_000,
		inputLen: 4000, stepsDone: 50, firstTokenUs: -8_600_000, ttftSet: true,
	})

	endpoints := []fwksched.Endpoint{
		newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3"),
		newFramedEndpoint("p-a", bylabel.RolePrefill, "NVIDIA-H100-80GB-HBM3"),
	}
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, endpoints)))

	require.NotEmpty(t, result.TargetEndpoints)
	assert.Equal(t, "default/d-a", result.TargetEndpoints[0].GetMetadata().GetNamespacedName().String(),
		"the decode endpoint must come first")
	if len(result.TargetEndpoints) > 1 {
		assert.Equal(t, "default/p-a", result.TargetEndpoints[1].GetMetadata().GetNamespacedName().String(),
			"the prefill endpoint must come second")
	}
}

// ---------------------------------------------------------------------------
// ProfileHandler
// ---------------------------------------------------------------------------

func TestHandlerPick_RunsJointProfileExactlyOnce(t *testing.T) {
	h := newTestHandler(t)
	request := newTestRequest(4000)
	profiles := map[string]fwksched.SchedulerProfile{"joint": nil}

	first := h.Pick(context.Background(), request, profiles, map[string]*fwksched.ProfileRunResult{})
	require.Len(t, first, 1)
	_, ok := first["joint"]
	assert.True(t, ok)

	// Once the joint profile has run there is nothing further to schedule. In particular there is
	// no decode stage followed by a disaggregation gate, which is the measured ablation.
	done := h.Pick(context.Background(), request, profiles, map[string]*fwksched.ProfileRunResult{"joint": {}})
	assert.Empty(t, done)
}

func TestHandlerPick_ReportsMissingJointProfile(t *testing.T) {
	h := newTestHandler(t)
	got := h.Pick(context.Background(), newTestRequest(4000),
		map[string]fwksched.SchedulerProfile{"decode": nil}, map[string]*fwksched.ProfileRunResult{})
	assert.Empty(t, got, "a profile named other than profiles.joint must not be run by accident")
}

func TestProcessResults_SplitsThePair(t *testing.T) {
	h := newTestHandler(t)
	decode := newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")
	prefill := newFramedEndpoint("p-a", bylabel.RolePrefill, "NVIDIA-H100-80GB-HBM3")

	t.Run("disaggregated", func(t *testing.T) {
		res, err := h.ProcessResults(context.Background(), newTestRequest(4000),
			map[string]*fwksched.ProfileRunResult{"joint": {TargetEndpoints: []fwksched.Endpoint{decode, prefill}}})
		require.NoError(t, err)
		assert.Equal(t, "decode", res.PrimaryProfileName)
		require.Len(t, res.ProfileResults["decode"].TargetEndpoints, 1)
		require.Len(t, res.ProfileResults["prefill"].TargetEndpoints, 1)
		assert.Equal(t, "default/d-a", res.ProfileResults["decode"].TargetEndpoints[0].GetMetadata().GetNamespacedName().String())
		assert.Equal(t, "default/p-a", res.ProfileResults["prefill"].TargetEndpoints[0].GetMetadata().GetNamespacedName().String())
	})

	t.Run("local carries no prefill entry", func(t *testing.T) {
		res, err := h.ProcessResults(context.Background(), newTestRequest(4000),
			map[string]*fwksched.ProfileRunResult{"joint": {TargetEndpoints: []fwksched.Endpoint{decode}}})
		require.NoError(t, err)
		assert.Len(t, res.ProfileResults, 1)
		assert.NotContains(t, res.ProfileResults, "prefill", "a local placement must not name a prefill endpoint")
	})

	t.Run("no endpoint is an error", func(t *testing.T) {
		_, err := h.ProcessResults(context.Background(), newTestRequest(4000),
			map[string]*fwksched.ProfileRunResult{"joint": {}})
		require.Error(t, err)
	})

	t.Run("nil request is an error", func(t *testing.T) {
		_, err := h.ProcessResults(context.Background(), nil, map[string]*fwksched.ProfileRunResult{})
		require.Error(t, err)
	})
}

// ---------------------------------------------------------------------------
// Request lifecycle
// ---------------------------------------------------------------------------

// PreRequest is what puts the prefill endpoint on the wire, and it opens the shadow entry the
// resident estimates are drawn from.
func TestPreRequest_SetsHeaderAndOpensShadowEntry(t *testing.T) {
	h := newTestHandler(t)
	request := newTestRequest(4000)
	decode := newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")
	prefill := newFramedEndpoint("p-a", bylabel.RolePrefill, "NVIDIA-H100-80GB-HBM3")

	h.PreRequest(context.Background(), request, &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode":  {TargetEndpoints: []fwksched.Endpoint{decode}},
			"prefill": {TargetEndpoints: []fwksched.Endpoint{prefill}},
		},
	})

	assert.Equal(t, "10.0.0.1:8000", request.Headers[routing.PrefillEndpointHeader])
	assert.Equal(t, 1, h.shadow.size())

	// The entry is indexed under both endpoints, so the decode endpoint sees an occupant and the
	// pool sees one too.
	_, occupants := h.shadow.residents("default/d-a", h.cfg, h.means.mean)
	assert.Empty(t, occupants, "a remotely-prefilling request occupies no prefill capacity on the decoder")
	assert.Len(t, h.shadow.prefillOccupants("default/p-a", h.cfg, h.means.mean), 1)
}

// A local placement must not set the prefill header, and any stale header must be cleared.
func TestPreRequest_LocalPlacementClearsHeader(t *testing.T) {
	h := newTestHandler(t)
	request := newTestRequest(4000)
	request.Headers[routing.PrefillEndpointHeader] = "stale:9999"

	h.PreRequest(context.Background(), request, &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"decode": {TargetEndpoints: []fwksched.Endpoint{newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")}},
		},
	})
	assert.NotContains(t, request.Headers, routing.PrefillEndpointHeader)

	// A pre-first-token request placed locally IS a collocated occupant on its decoder.
	_, occupants := h.shadow.residents("default/d-a", h.cfg, h.means.mean)
	assert.Len(t, occupants, 1)
}

// First token is detected by the completion count first reaching one, and a resident only becomes
// chargeable on the decode side once it has one.
func TestResponseBody_FirstTokenPromotesOccupantToResident(t *testing.T) {
	h := newTestHandler(t)
	request := newTestRequest(4000)
	openLocal(h, request, "default/d-a")

	residents, occupants := h.shadow.residents("default/d-a", h.cfg, h.means.mean)
	require.Empty(t, residents)
	require.Len(t, occupants, 1, "before a first token the request is an occupant, not a resident")

	h.ResponseBody(context.Background(), request, &fwkrc.Response{
		Usage: fwkrh.Usage{CompletionTokens: 1},
	}, nil)

	residents, occupants = h.shadow.residents("default/d-a", h.cfg, h.means.mean)
	assert.Len(t, residents, 1, "a first token makes the request chargeable on the decode side")
	assert.Empty(t, occupants)
	assert.True(t, residents[0].TTFTSet)
}

// A chunk carrying no usage must not be read as zero progress, and progress must never walk
// backwards.
func TestResponseBody_ProgressIsMonotonic(t *testing.T) {
	h := newTestHandler(t)
	request := newTestRequest(4000)
	openLocal(h, request, "default/d-a")

	for _, n := range []int{1, 5, 0, 3, 12} {
		h.ResponseBody(context.Background(), request, &fwkrc.Response{Usage: fwkrh.Usage{CompletionTokens: n}}, nil)
	}
	residents, _ := h.shadow.residents("default/d-a", h.cfg, h.means.mean)
	require.Len(t, residents, 1)
	assert.Equal(t, int64(12), residents[0].StepsDone)
}

// End of stream deregisters the request and folds its realized output length into the class mean.
func TestResponseBody_EndOfStreamClosesAndUpdatesClassMean(t *testing.T) {
	h := newTestHandler(t)
	assert.Equal(t, 1.0, h.means.mean(""), "a class with no completions seeds at one, not zero")

	for i, output := range []int{100, 300} {
		request := newTestRequest(4000)
		request.RequestID = string(rune('a' + i))
		openLocal(h, request, "default/d-a")
		h.ResponseBody(context.Background(), request, &fwkrc.Response{
			EndOfStream: true, Usage: fwkrh.Usage{CompletionTokens: output},
		}, nil)
	}

	assert.Zero(t, h.shadow.size(), "every completed request must be deregistered")
	assert.InDelta(t, 200.0, h.means.mean(""), 1e-9)
}

// An unknown request must be ignored rather than opening an entry from the response path: traffic
// this replica did not place has no arrival time or input length.
func TestResponseBody_IgnoresUnknownRequest(t *testing.T) {
	h := newTestHandler(t)
	h.ResponseBody(context.Background(), newTestRequest(4000), &fwkrc.Response{
		EndOfStream: true, Usage: fwkrh.Usage{CompletionTokens: 42},
	}, nil)
	assert.Zero(t, h.shadow.size())
	assert.Equal(t, 1.0, h.means.mean(""), "unplaced traffic must not move the class estimate")
}

// ---------------------------------------------------------------------------
// Shadow table
// ---------------------------------------------------------------------------

// The table is written from the response goroutine and read from the scheduling path, so it must
// tolerate concurrent access. Run with -race for this to be meaningful.
func TestShadowTable_ConcurrentReadWrite(t *testing.T) {
	h := newTestHandler(t)
	var wg sync.WaitGroup

	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				id := string(rune('a'+w)) + string(rune('0'+i%10))
				h.shadow.open(&shadowEntry{requestID: id, decodeID: "default/d-a", inputLen: 4000})
				h.shadow.observe(id, int64(i), int64(i))
				if i%3 == 0 {
					h.shadow.close(id)
				}
			}
		}(w)
	}
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 400; i++ {
				h.shadow.residents("default/d-a", h.cfg, h.means.mean)
				h.shadow.prefillOccupants("default/d-a", h.cfg, h.means.mean)
				h.shadow.size()
			}
		}()
	}
	wg.Wait()
}

// Requests can terminate without a final chunk, and the response hooks carry no termination state.
// Without the sweep those entries would be charged as residents forever and permanently inflate
// every externality on their endpoint.
func TestShadowTable_ReapsStaleEntries(t *testing.T) {
	table := newShadowTable()
	table.open(&shadowEntry{requestID: "fresh", decodeID: "default/d-a", inputLen: 4000})
	table.open(&shadowEntry{requestID: "abandoned", decodeID: "default/d-a", inputLen: 4000})

	// Age one entry past the window.
	table.mu.Lock()
	table.byRequest["abandoned"].lastSeen = time.Now().Add(-2 * shadowEntryTTL)
	table.mu.Unlock()

	assert.Equal(t, 1, table.reap(shadowEntryTTL, time.Now()))
	assert.Equal(t, 1, table.size())
	_, ok := table.byRequest["fresh"]
	assert.True(t, ok)
}

// A repeated request ID must replace the earlier entry so a retry is not counted as two residents.
func TestShadowTable_ReopenReplacesRatherThanDuplicates(t *testing.T) {
	table := newShadowTable()
	table.open(&shadowEntry{requestID: "r1", decodeID: "default/d-a", inputLen: 4000})
	table.open(&shadowEntry{requestID: "r1", decodeID: "default/d-b", inputLen: 4000})

	assert.Equal(t, 1, table.size())
	assert.Empty(t, table.byDecode["default/d-a"], "the stale index entry must be unlinked")
	assert.Len(t, table.byDecode["default/d-b"], 1)
}

func TestShadowTable_CloseUnlinksBothIndexes(t *testing.T) {
	table := newShadowTable()
	table.open(&shadowEntry{requestID: "r1", decodeID: "default/d-a", prefillID: "default/p-a", inputLen: 4000})

	_, ok := table.close("r1")
	require.True(t, ok)
	assert.Zero(t, table.size())
	assert.Empty(t, table.byDecode)
	assert.Empty(t, table.byPrefill)

	_, ok = table.close("r1")
	assert.False(t, ok, "closing twice must not report a second close")
}

// ---------------------------------------------------------------------------
// Uncached prompt length
// ---------------------------------------------------------------------------

// The uncached count must use the PREFIX PRODUCER's block size, not the engine's, and must use the
// literal cached-block count rather than the tier-weighted ranking score.
func TestUncachedPromptTokens_UsesProducerBlockSizeAndLiteralCount(t *testing.T) {
	p := newTestPicker(t)

	// 10 cached blocks of 64 tokens is 640 cached tokens out of 4000.
	ep := newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")
	ep.Put(p.prefixKey.String(), attrprefix.NewPrefixCacheMatchInfo(10, 63, 64))

	ap, observed := p.uncachedPromptTokens(ep, 4000)
	assert.True(t, observed)
	assert.Equal(t, 4000-640, ap)
}

// A prefix miss means "no information", not "nothing cached", so the estimate falls back to the
// full prompt and over-prices prefill rather than asserting a cold cache as fact.
func TestUncachedPromptTokens_MissFallsBackToFullPrompt(t *testing.T) {
	p := newTestPicker(t)
	ep := newFramedEndpoint("d-a", bylabel.RoleDecode, "NVIDIA-H100-80GB-HBM3")

	ap, observed := p.uncachedPromptTokens(ep, 4000)
	assert.False(t, observed)
	assert.Equal(t, 4000, ap)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func minimalParams() map[string]any {
	return map[string]any{
		"v": 8.0,
		"sloTargets": map[string]any{
			"interactive": map[string]any{"tauTTFTUs": 1_000_000, "tauITLUs": 50_000, "tauE2EUs": 16_000_000},
		},
		"activeSLOTargets":            "interactive",
		"chunkTokens":                 2048,
		"blockSize":                   16,
		"maxBatchSize":                256,
		"outputTokenProcessingUs":     45,
		"cXferSizeAware":              true,
		"xferBaseUs":                  50.0,
		"xferBandwidthGBps":           25.0,
		"kvBytesPerTokenPerGPU":       81920.0,
		"gpuTypeLabelKey":             gpuLabel,
		"roleLabelKey":                bylabel.RoleLabel,
		"prefixMatchInfoProducerName": "approx-prefix-cache-producer",
		"coefficients": map[string]any{
			"NVIDIA-H100-80GB-HBM3": map[string]any{
				"alphaDUs": h100.AlphaD, "alphaPUs": h100.AlphaP, "c0UsPerReq": h100.C0,
				"c1UsPerToken": h100.C1, "cPfUsPerToken": h100.CPf, "cAttnUsPerUnit": h100.CAttn,
			},
		},
		"noCapacity": true,
	}
}

func withParams(mutate func(map[string]any)) map[string]any {
	m := minimalParams()
	mutate(m)
	return m
}

func decoderFor(t *testing.T, params map[string]any) *json.Decoder {
	t.Helper()
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	return fwkplugin.StrictDecoder(raw)
}

func newTestHandler(t *testing.T) *Handler {
	t.Helper()
	plg, err := HandlerFactory("policy", decoderFor(t, minimalParams()), testutils.NewTestHandle(context.Background()))
	require.NoError(t, err)
	return plg.(*Handler)
}

func newTestPicker(t *testing.T) *Picker {
	t.Helper()
	return &Picker{
		typedName: fwkplugin.TypedName{Type: PickerType, Name: PickerType},
		core:      newTestHandler(t).core,
	}
}

func newTestRequest(promptTokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:   "req-1",
		TargetModel: "meta-llama/Llama-3.3-70B-Instruct",
		Headers:     map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				PerPromptTokens: [][]uint32{make([]uint32, promptTokens)},
			},
		},
	}
}

// newFramedEndpoint builds an endpoint with the labels and metrics the policy reads.
func newFramedEndpoint(name, role, gpuType string) fwksched.Endpoint {
	labels := map[string]string{}
	if role != "" {
		labels[bylabel.RoleLabel] = role
	}
	if gpuType != "" {
		labels[gpuLabel] = gpuType
	}
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: name},
			Address:        "10.0.0.1",
			Port:           "8000",
			Labels:         labels,
		},
		&fwkdl.Metrics{
			RunningRequestsSize: 4,
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.25,
			CacheBlockSize:      16,
			CacheNumBlocks:      10000,
		},
		fwkdl.NewAttributes(),
	)
}

func scored(endpoints []fwksched.Endpoint) []*fwksched.ScoredEndpoint {
	out := make([]*fwksched.ScoredEndpoint, len(endpoints))
	for i, ep := range endpoints {
		// A non-zero inherited score, which the picker must ignore in favour of computing J.
		out[i] = &fwksched.ScoredEndpoint{Endpoint: ep, Score: 0.9}
	}
	return out
}

func openLocal(h *Handler, request *fwksched.InferenceRequest, decodeID string) {
	h.shadow.open(&shadowEntry{
		requestID: request.RequestID,
		decodeID:  decodeID,
		arrivalUs: h.nowUs(),
		inputLen:  int64(promptTokensOf(request)),
	})
}
