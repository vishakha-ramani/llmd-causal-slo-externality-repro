package causalsloexternality

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// The token producer supplies the arriving request's input length, which every term of the objective
// depends on. The deployment configures its vLLM render backend, so that backend is the component
// most likely to fail on a cluster and it must not be covered by shape checks alone.
//
// The backend's own seams are unexported and in-package, but its endpoint URL is a public parameter
// and its render paths are fixed, so a local server driven through the public factory exercises the
// real plugin: construction, the negotiated timeout, the produced data key, a successful
// tokenization, and the failure path this policy refuses to score on.

const (
	completionsRenderPath = "/v1/completions/render"
	chatRenderPath        = "/v1/chat/completions/render"
)

// renderServer stands in for the vLLM render sidecar. status other than 200 exercises the failure
// path.
func renderServer(t *testing.T, tokenIDs []uint32, status int) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	// The completions path returns an ARRAY of results, one per prompt; the chat path returns a
	// single object.
	mux.HandleFunc(completionsRenderPath, func(w http.ResponseWriter, _ *http.Request) {
		if status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		_ = json.NewEncoder(w).Encode([]map[string]any{{"token_ids": tokenIDs}})
	})
	mux.HandleFunc(chatRenderPath, func(w http.ResponseWriter, _ *http.Request) {
		if status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": tokenIDs})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// newRenderTokenProducer builds the real token producer against a local render server, using the
// same public parameter shape both overlays carry.
func newRenderTokenProducer(t *testing.T, srv *httptest.Server) fwkplugin.Plugin {
	t.Helper()
	params := map[string]any{
		"modelName": "meta-llama/Llama-3.3-70B-Instruct",
		"vllm": map[string]any{
			"url": srv.URL,
			// Distinct from the defaults so the assertion below proves the plumbing rather than a
			// constant.
			"timeout":   "2s",
			"mmTimeout": "5s",
		},
	}
	// The name is omitted in both overlays so the produced key keeps the default producer name; the
	// factory is called with that same name here.
	plg, err := tokenproducer.PluginFactory(tokenproducer.PluginType,
		decoderFor(t, params), testutils.NewTestHandle(context.Background()))
	require.NoError(t, err, "the overlays' token-producer parameters must construct")
	return plg
}

// Construction is local: the only network contact is a background warmup, so the plugin builds
// against a live local server without blocking.
func TestTokenProducer_ConstructsFromTheOverlayParameters(t *testing.T) {
	plg := newRenderTokenProducer(t, renderServer(t, []uint32{1, 2, 3}, http.StatusOK))
	assert.Equal(t, tokenproducer.PluginType, plg.TypedName().Type)
	assert.Equal(t, tokenproducer.PluginType, plg.TypedName().Name,
		"the name must default to the type, or the produced key changes")
}

// THE INVARIANT THE WHOLE TOKENIZER DECISION RESTS ON.
//
// The produced key must equal the bare key constant. It is `DataType/ProducerName`, so a non-default
// name publishes a key no consumer asks for. The consequence is deterministic rather than a race: the
// approximate prefix producer requires the bare key, so the framework would auto-create an
// estimate-backed producer to supply it, and the explicitly configured render backend would warm up,
// do its per-request round trip, and be ignored. That pays the full latency cost of a real tokenizer
// while getting byte-estimate semantics, and it is one YAML character away.
func TestTokenProducer_PublishesTheBareDataKey(t *testing.T) {
	plg := newRenderTokenProducer(t, renderServer(t, []uint32{1, 2, 3}, http.StatusOK))

	producer, ok := plg.(fwkplugin.ProducerPlugin)
	require.True(t, ok, "the token producer must be a data producer")

	keys := make([]string, 0, 1)
	for k := range producer.Produces() {
		keys = append(keys, k.String())
	}
	require.Len(t, keys, 1)
	assert.Equal(t, tokenproducer.TokenizedPromptDataKey.String(), keys[0],
		"a produced key other than the bare constant silently falls back to the byte estimate")
	assert.Equal(t, "TokenizedPrompt/token-producer", keys[0])
}

// The backend negotiates a produce timeout far above the framework default, and it is the worst case
// of the two configured values because it assumes image downloads. Asserting the configured value
// rather than the default proves the parameters reach the renderer.
func TestTokenProducer_NegotiatesItsProduceTimeout(t *testing.T) {
	plg := newRenderTokenProducer(t, renderServer(t, []uint32{1, 2, 3}, http.StatusOK))

	aware, ok := plg.(fwkrc.TimeoutAwareProducer)
	require.True(t, ok, "the render backend must declare its own timeout or the director cuts it short")
	assert.Equal(t, 5*time.Second, aware.ProduceTimeout(),
		"the declared budget is the worst case of timeout and mmTimeout")
}

// A successful render populates the tokenized prompt with real token IDs, and the policy then reads
// its input length from them.
func TestTokenProducer_SuccessfulRenderPopulatesRealTokens(t *testing.T) {
	tokens := []uint32{11, 22, 33, 44, 55, 66, 77}
	plg := newRenderTokenProducer(t, renderServer(t, tokens, http.StatusOK))
	producer := plg.(fwkplugin.ProducerPlugin).(interface {
		Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error
	})

	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Headers:   map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "some prompt text"}},
		},
	}
	require.NoError(t, producer.Produce(context.Background(), request, nil))

	require.NotNil(t, request.Body.TokenizedPrompt)
	assert.Equal(t, len(tokens), request.Body.TokenizedPrompt.TokenCount())
	assert.Equal(t, len(tokens), promptTokensOf(request), "the policy reads the same length")
	assert.False(t, tokenizationFailed(request))
}

// THE FAILURE PATH, END TO END. A render outage leaves the tokenized prompt unset and the error is
// swallowed upstream, so the request reaches this policy with no input length. The policy must refuse
// to score rather than substitute a byte estimate or a zero length.
func TestTokenProducer_RenderOutageMakesThePolicyRefuseToScore(t *testing.T) {
	plg := newRenderTokenProducer(t, renderServer(t, nil, http.StatusInternalServerError))
	producer := plg.(fwkplugin.ProducerPlugin).(interface {
		Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error
	})

	request := &fwksched.InferenceRequest{
		RequestID: "req-1",
		Headers:   map[string]string{},
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "some prompt text"}},
		},
	}
	err := producer.Produce(context.Background(), request, nil)
	require.Error(t, err, "a failing renderer must surface an error to the director")
	require.Nil(t, request.Body.TokenizedPrompt, "and must leave the tokenized prompt unset")

	// The director swallows that error, so the policy sees only the absent prompt. Content in with no
	// tokens out is the discriminator.
	require.True(t, tokenizationFailed(request))

	p := newTestPicker(t)
	endpoints := []fwksched.Endpoint{newFramedEndpoint("d-a", "decode", "NVIDIA-H100-80GB-HBM3")}
	result := p.Pick(context.Background(), scored(p.Filter(context.Background(), request, endpoints)))
	assert.Empty(t, result.TargetEndpoints,
		"the argmin must refuse rather than price every candidate against a fabricated input length")
}
