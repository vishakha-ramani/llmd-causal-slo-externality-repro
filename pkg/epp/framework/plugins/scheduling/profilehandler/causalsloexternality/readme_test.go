package causalsloexternality

import (
	"encoding/json"
	"os"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/yaml"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// The README's configuration example is the first thing a reader copies, and documentation drifts
// from code silently: a parameter rename leaves the example describing a configuration that fails
// validation at startup, and nothing else notices.
//
// So the example is checked here rather than by inspection. This decodes the fenced YAML block out of
// README.md and drives it through the same strict decoder and the same validate() the factory uses,
// which means a rename, a removed field or a newly-rejected value breaks this test instead of
// surviving into the docs.

// readmeConfigBlock extracts the first fenced YAML block following the Configuration heading.
func readmeConfigBlock(t *testing.T) []byte {
	t.Helper()
	// Go tests run with the package directory as their working directory.
	raw, err := os.ReadFile("README.md")
	require.NoError(t, err)

	re := regexp.MustCompile("(?s)## Configuration\\s*\n+```yaml\n(.*?)```")
	m := re.FindSubmatch(raw)
	require.NotNil(t, m, "README.md must carry a fenced yaml block under '## Configuration'")
	return m[1]
}

// pluginDoc is the slice of the EndpointPickerConfig shape this test needs.
type pluginDoc struct {
	Plugins []struct {
		Type       string          `json:"type"`
		Name       string          `json:"name"`
		Parameters json.RawMessage `json:"parameters"`
	} `json:"plugins"`
	SchedulingProfiles []struct {
		Name    string `json:"name"`
		Plugins []struct {
			PluginRef string `json:"pluginRef"`
		} `json:"plugins"`
	} `json:"schedulingProfiles"`
}

// The documented parameters must decode strictly and validate. This is the check that would have
// caught the example describing a flat SLO triple after the code moved to a keyed map.
func TestREADMEExample_DecodesAndValidates(t *testing.T) {
	var doc pluginDoc
	require.NoError(t, yaml.Unmarshal(readmeConfigBlock(t), &doc))

	var policy json.RawMessage
	for _, p := range doc.Plugins {
		if p.Type == PluginType {
			policy = p.Parameters
		}
	}
	require.NotNil(t, policy, "the example must configure the %s plugin", PluginType)

	cfg := defaultConfig
	require.NoError(t, fwkplugin.StrictDecoder(policy).Decode(&cfg),
		"the documented parameters must decode strictly against the current Config struct")
	require.NoError(t, cfg.validate(),
		"the documented parameters must be a configuration this plugin accepts")

	// Spot-check the values a reader most relies on, so a silent value edit is caught too.
	assert.Equal(t, 8.0, cfg.V)
	assert.Equal(t, 2048, cfg.ChunkTokens)
	assert.Equal(t, 16, cfg.BlockSize)
	assert.Equal(t, int64(256), cfg.MaxBatchSize)
	assert.Equal(t, "interactive", cfg.ActiveSLOTargets)
	assert.ElementsMatch(t, []string{"interactive", "reasoning", "deep_research"}, sortedKeys(cfg.SLOTargets))
}

// Declaration order is load-bearing, so the example must show it. The picker resolves its policy
// through the plugin handle and factories run in declaration order.
func TestREADMEExample_ShowsLoadBearingDeclarationOrder(t *testing.T) {
	var doc pluginDoc
	require.NoError(t, yaml.Unmarshal(readmeConfigBlock(t), &doc))

	order := map[string]int{}
	for i, p := range doc.Plugins {
		order[p.Type] = i
		if p.Type == "token-producer" {
			assert.Empty(t, p.Name,
				"the token producer's name must stay unset, or its produced key changes and the byte estimate wins")
		}
	}

	require.Contains(t, order, PluginType)
	require.Contains(t, order, PickerType)
	assert.Less(t, order[PluginType], order[PickerType],
		"the policy must be declared before the picker that resolves it")

	// The producers supply inputs the policy declares as dependencies.
	require.Contains(t, order, "token-producer")
	require.Contains(t, order, "approx-prefix-cache-producer")
}

// The profile must carry the picker and nothing else: a role filter would split the pools and reduce
// the joint argmin to the decode-first decomposition, and a scorer's contribution would be clamped.
func TestREADMEExample_ProfileCarriesOnlyThePicker(t *testing.T) {
	var doc pluginDoc
	require.NoError(t, yaml.Unmarshal(readmeConfigBlock(t), &doc))

	require.Len(t, doc.SchedulingProfiles, 1, "the policy runs exactly one role-blind profile")
	refs := make([]string, 0, len(doc.SchedulingProfiles[0].Plugins))
	for _, r := range doc.SchedulingProfiles[0].Plugins {
		refs = append(refs, r.PluginRef)
	}
	assert.Equal(t, []string{PickerType}, refs,
		"no role filter and no scorer may appear in the joint profile")

	// The profile handler is found by interface scan, not by reference.
	assert.NotContains(t, refs, PluginType)
}
