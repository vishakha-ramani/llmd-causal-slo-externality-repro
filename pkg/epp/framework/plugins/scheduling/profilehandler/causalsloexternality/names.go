package causalsloexternality

import "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"

const (
	// PluginType is the registered type of the policy plugin: it owns the shadow table and the
	// request lifecycle, and drives the role-blind profile.
	PluginType = "causal-slo-externality"

	// PickerType is the registered type of the argmin. It is a separate type because the
	// scheduling Picker and ProfileHandler interfaces both declare a method named Pick, so no
	// single Go type can satisfy both. It must be declared AFTER PluginType in the config, since
	// factories run in declaration order and it resolves its policy plugin by name.
	PickerType = "causal-slo-externality-picker"

	defaultJointProfile   = "joint"
	defaultDecodeProfile  = "decode"
	defaultPrefillProfile = "prefill"

	// defaultRoleLabelKey tracks the label the stock role filters key on, so this
	// plugin's pool partition and those filters cannot drift apart.
	defaultRoleLabelKey = bylabel.RoleLabel

	// defaultPrefixProducerName matches the producer the uncached-prompt-length input is
	// read from. A DataKey's identity includes its producer name, so binding both arms to
	// the same name is what keeps their prefix-residency signals comparable.
	defaultPrefixProducerName = "approx-prefix-cache-producer"
)

// POOL MEMBERSHIP MIRRORS THE STOCK ROLE FILTERS EXACTLY, INCLUDING THEIR ASYMMETRY.
//
// The comparison baseline routes through decode-filter and prefill-filter, so this policy must
// admit the same endpoints those filters would. A divergence means the two arms see different
// fleets, which a comparison cannot absorb: it is invisible on a well-labelled fleet and silent on
// a badly-labelled one.
//
// The sets are taken from the stock constructors using the in-tree constants rather than string
// literals, so an upstream role addition is a compile-time surface rather than silent drift. Three
// properties are deliberately reproduced:
//
//  1. THE SETS OVERLAP. prefill-decode, the deprecated `both`, and encode-prefill-decode all belong
//     to BOTH pools, so a dual-role endpoint is a legal decode candidate AND a legal prefill
//     candidate. A strict equality partition would drop such an endpoint from the whole cross
//     product.
//  2. THE EXTRAS ARE ASYMMETRIC. encode-prefill is prefill-only; decode has no counterpart.
//  3. THE UNLABELLED CASE IS ASYMMETRIC, and is carried by decodeAllowsNoLabel below rather than
//     by these maps.
//
// Encode-only endpoints belong to neither pool, which both stock filters also enforce.
var (
	decodeRoles = map[string]bool{
		bylabel.RoleDecode:              true,
		bylabel.RolePrefillDecode:       true,
		bylabel.RoleBoth:                true, //nolint:staticcheck // the stock filters still accept it, and this must match them
		bylabel.RoleEncodePrefillDecode: true,
	}
	prefillRoles = map[string]bool{
		bylabel.RolePrefill:             true,
		bylabel.RoleEncodePrefill:       true,
		bylabel.RolePrefillDecode:       true,
		bylabel.RoleBoth:                true, //nolint:staticcheck // the stock filters still accept it, and this must match them
		bylabel.RoleEncodePrefillDecode: true,
	}
)

// decodeAllowsNoLabel and prefillAllowsNoLabel carry the stock filters' allowsNoLabel arguments:
// decode-filter is constructed with true and prefill-filter with false.
//
// So an endpoint carrying no role label IS a decode candidate here, matching the baseline, rather
// than being dropped. That is deliberate even though it admits an endpoint whose role is unknown.
// Being stricter would be defensible in isolation, but not in one arm of a comparison: the role
// label is a documented runtime precondition, and enforcing it belongs in a deployment preflight
// check rather than in the policy under test. Its presence is logged per decision so a run stays
// auditable.
const (
	decodeAllowsNoLabel  = true
	prefillAllowsNoLabel = false
)

// poolMembership reports which pools an endpoint's role label admits it to.
func poolMembership(role string) (isDecode, isPrefill bool) {
	if role == "" {
		return decodeAllowsNoLabel, prefillAllowsNoLabel
	}
	return decodeRoles[role], prefillRoles[role]
}
