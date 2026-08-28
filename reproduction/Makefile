SHELL := /bin/sh

PYTHON ?= python3
RUBY ?= ruby
OUT_DIR ?= out/r5
RUN_NAME ?= qwen-causal-reproduction

EVENTS := data/r5/events.jsonl.gz
TRACE := data/r5/trace_data.csv
TRANSFERS := data/r5/nixl-transfer-windows.csv
EXPECTED := results/r5

.PHONY: test replay verify manifest verify-manifest clean

test:
	$(PYTHON) scripts/test-analyze-causal-observability.py

replay:
	mkdir -p $(OUT_DIR)
	$(PYTHON) scripts/analyze-causal-observability.py \
		$(EVENTS) \
		--client-trace $(TRACE) \
		--transfer-windows $(TRANSFERS) \
		--report-out $(OUT_DIR)/prediction-validation.json \
		--rows-out $(OUT_DIR)/prediction-residuals.csv \
		--externality-rows-out $(OUT_DIR)/resident-prediction-residuals.csv \
		--strict > $(OUT_DIR)/analyzer.stdout

verify: test replay
	cmp $(EXPECTED)/prediction-validation.json $(OUT_DIR)/prediction-validation.json
	cmp $(EXPECTED)/prediction-residuals.csv $(OUT_DIR)/prediction-residuals.csv
	cmp $(EXPECTED)/resident-prediction-residuals.csv $(OUT_DIR)/resident-prediction-residuals.csv
	@printf '%s\n' 'R5 strict joins and byte-for-byte result replay: PASS'

manifest:
	mkdir -p out
	PIPELINERUN_TEMPLATE=$(CURDIR)/scripts/causal-qwen-overlap-20260827-151229.yaml \
		$(RUBY) scripts/generate-qwen-iterlaw-run.rb causal $(RUN_NAME) out/$(RUN_NAME).yaml

verify-manifest: manifest
	$(RUBY) -ryaml -e 'doc = YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true); abort "wrong run name" unless doc.dig("metadata", "name") == ARGV.fetch(1); params = doc.dig("spec", "params").to_h { |p| [p.fetch("name"), p.fetch("value")] }; abort "wrong workload" unless YAML.safe_load(params.fetch("workloadSpec")).dig("cohorts", 0, "output_distribution", "params", "value") == 512; puts "portable manifest generation: PASS"' out/$(RUN_NAME).yaml $(RUN_NAME)

clean:
	rm -rf out
