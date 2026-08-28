#!/usr/bin/env ruby

require "yaml"

abort "usage: #{$PROGRAM_NAME} causal|stock RUN_NAME OUTPUT.yaml" unless ARGV.length == 3

mode, run_name, output = ARGV
abort "mode must be causal or stock" unless %w[causal stock].include?(mode)

rate = Float(ENV.fetch("RATE", "0.5"))
requests = Integer(ENV.fetch("REQUESTS", "6"))
input_tokens = Integer(ENV.fetch("INPUT_TOKENS", "15000"))
output_tokens = Integer(ENV.fetch("OUTPUT_TOKENS", "512"))
seed = Integer(ENV.fetch("SEED", "4242"))
namespace = ENV.fetch("NAMESPACE", "vramani-perfcal")
pipeline_name = ENV.fetch("PIPELINE_NAME", "sim2real")
service_account = ENV.fetch("SERVICE_ACCOUNT", "helm-installer")
data_pvc = ENV.fetch("DATA_PVC", "data-pvc")
source_pvc = ENV.fetch("SOURCE_PVC", "source-pvc")
epp_registry = ENV.fetch("EPP_REGISTRY", "quay.io/vishakharamani")
epp_repository = ENV.fetch("EPP_REPOSITORY", "llm-d-router")
epp_tag = ENV.fetch("EPP_TAG", "causal-qwen-observability-v2-20260827")
sidecar_repository = ENV.fetch(
  "SIDECAR_REPOSITORY",
  "quay.io/vishakharamani/llm-d-router-disagg-sidecar",
)
sidecar_tag = ENV.fetch("SIDECAR_TAG", "causal-qwen-observability-v2-20260827")
hf_secret = ENV.fetch("HF_SECRET", "hf-secret")
model_pvc_size = ENV.fetch("MODEL_PVC_SIZE", "100Gi")
benchmark_repo = ENV.fetch("BENCHMARK_REPO", "https://github.com/llm-d/llm-d-benchmark.git")
benchmark_commit = ENV.fetch("BENCHMARK_COMMIT", "76473d05a0f546a6bc0c4662ed7e34104da73891")
blis_repo = ENV.fetch("BLIS_REPO", "https://github.com/kalantar/inference-sim.git")
blis_commit = ENV.fetch("BLIS_COMMIT", "cc8846cc41e71488cc32004778dde83dc747a2f7")

root = File.expand_path(__dir__)
template_path = ENV.fetch(
  "PIPELINERUN_TEMPLATE",
  File.join(root, "causal-qwen-overlap-20260827-151229.yaml"),
)
doc = YAML.safe_load(File.read(template_path), aliases: true)
params = doc.fetch("spec").fetch("params").to_h { |entry| [entry.fetch("name"), entry] }

doc.fetch("metadata")["name"] = run_name
doc.fetch("metadata")["namespace"] = namespace
params.fetch("experimentId")["value"] = run_name
params.fetch("runName")["value"] = run_name
params.fetch("namespace")["value"] = namespace
params.fetch("phase")["value"] = "#{mode}-qwen-controlled-iterlaw"

scenario_doc = YAML.safe_load(params.fetch("scenarioContent").fetch("value"), aliases: true)
scenario = scenario_doc.fetch("scenario").first
scenario["name"] = "pdinfocomm2-qwen-controlled-iterlaw"
scenario.fetch("huggingface")["secretName"] = hf_secret
scenario.fetch("images").fetch("routingSidecar")["repository"] = sidecar_repository
scenario.fetch("images").fetch("routingSidecar")["tag"] = sidecar_tag
scenario.fetch("storage").fetch("modelPvc")["size"] = model_pvc_size
scenario.fetch("routing").fetch("proxy")["zapLogLevel"] = 4

%w[decode prefill].each do |role|
  flags = scenario.fetch(role).fetch("vllm").fetch("additionalFlags")
  flags.delete("--enable-prefix-caching")
  flags.delete("--no-enable-prefix-caching")
  flags << "--no-enable-prefix-caching"
end

epp = scenario.fetch("router").fetch("epp")
epp["image"] = {
  "pullPolicy" => "Always",
  "registry" => epp_registry,
  "repository" => epp_repository,
  "tag" => epp_tag,
}

scenario.fetch("router").fetch("tokenizer").fetch("env").each do |entry|
  secret_ref = entry.dig("valueFrom", "secretKeyRef")
  secret_ref["name"] = hf_secret if secret_ref
end

if mode == "causal"
  plugin_doc = YAML.safe_load(epp.fetch("pluginsCustomConfig").fetch("custom-plugins.yaml"), aliases: true)
  policy = plugin_doc.fetch("plugins").find { |plugin| plugin["type"] == "causal-slo-externality" }
  abort "causal policy missing from template" unless policy

  policy_params = policy.fetch("parameters")
  policy_params["requireNoResidentPrefill"] = true
  policy_params["prefixCachingDisabled"] = true
  policy_params["xferBaseUs"] = 595.162
  policy_params["xferBandwidthGBps"] = 12.213979
  policy_params["kvBytesPerTokenPerGPU"] = 163_840.0
  policy_params["coefficients"] = {
    "NVIDIA-H100-80GB-HBM3" => {
      "alphaDUs" => 10_884.573061102548,
      "alphaPUs" => 10_884.573061102548,
      "c0UsPerReq" => 10.419827361034679,
      "c1UsPerToken" => 0.05267373344102902,
      "cPfUsPerToken" => 39.806418327287684,
      "cAttnUsPerUnit" => 0.0014236690050459293,
    },
  }
  epp.fetch("pluginsCustomConfig")["custom-plugins.yaml"] = YAML.dump(plugin_doc)
else
  baseline_path = ENV["BASELINE_CONFIG"]
  abort "stock mode requires BASELINE_CONFIG=/path/to/baseline_config.yaml" if baseline_path.nil? || baseline_path.empty?
  baseline = YAML.safe_load(File.read(baseline_path), aliases: true)
  stock_epp = baseline.fetch("scenario").first.fetch("router").fetch("epp")
  stock_epp["image"] = epp.fetch("image")
  plugin_text = stock_epp.fetch("pluginsCustomConfig").fetch("custom-plugins.yaml")
  stock_plugins = YAML.safe_load(plugin_text, aliases: true)
  tokenizer = stock_plugins.fetch("plugins").find { |plugin| plugin["type"] == "token-producer" }
  tokenizer.fetch("parameters")["modelName"] = "Qwen/Qwen3-14B"
  stock_epp.fetch("pluginsCustomConfig")["custom-plugins.yaml"] = YAML.dump(stock_plugins)
  scenario.fetch("router")["epp"] = stock_epp
end

params.fetch("scenarioContent")["value"] = YAML.dump(scenario_doc)
params.fetch("workloadSpec")["value"] = YAML.dump(
  "version" => "2",
  "seed" => seed,
  "category" => "language",
  "aggregate_rate" => rate,
  "num_requests" => requests,
  "cohorts" => [{
    "id" => "interactive-controlled",
    "population" => 1,
    "tenant_id" => "interactive-users",
    "slo_class" => "standard",
    "rate_fraction" => 1.0,
    "streaming" => true,
    "arrival" => {"process" => "constant"},
    "input_distribution" => {"type" => "constant", "params" => {"value" => input_tokens}},
    "output_distribution" => {"type" => "constant", "params" => {"value" => output_tokens}},
  }],
)
params.fetch("maxConcurrency")["value"] = requests.to_s
params.fetch("warmupRequests")["value"] = "0"
params.fetch("prewarmDuration")["value"] = "0s"
params.fetch("benchmarkGitRepoUrl")["value"] = benchmark_repo
params.fetch("benchmarkGitCommit")["value"] = benchmark_commit
params.fetch("blisGitRepoUrl")["value"] = blis_repo
params.fetch("blisGitCommit")["value"] = blis_commit
doc.fetch("spec").fetch("pipelineRef")["name"] = pipeline_name
doc.fetch("spec").fetch("taskRunTemplate")["serviceAccountName"] = service_account
workspaces = doc.fetch("spec").fetch("workspaces").to_h { |entry| [entry.fetch("name"), entry] }
workspaces.fetch("data-storage").fetch("persistentVolumeClaim")["claimName"] = data_pvc
workspaces.fetch("source").fetch("persistentVolumeClaim")["claimName"] = source_pvc

File.write(output, YAML.dump(doc))
