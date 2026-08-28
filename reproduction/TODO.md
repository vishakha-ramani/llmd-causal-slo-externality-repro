# Validation TODO

## Qwen coefficient reliability

The current methodology is structurally correct but the evidence is uneven.
Decode-only and prefill-only measurements fit the coefficients; mixed batches
are held out and test whether their marginal contributions add. Current data:

- decode-only: 6,048 iterations across 24 `(batch, context)` cells, 0.405%
  in-grid MAPE after cell-median fitting;
- prefill-only: 27 observations over nine lengths and three repeats, 16.36%
  MAPE;
- held-out mixed: 72 rows, 6.96% MAPE; these rows were not used for fitting;
- R5: 76 clean decision snapshots and no live mixed resident-prefill rows.

Before calling the vector high confidence:

1. Repeat every prefill-only point enough times for bootstrap confidence
   intervals.
2. Extend decode contexts to 8k, 16k, and the configured 32k limit.
3. Test held-out mixed batches at decode sizes 1, 2, 4, 6, 8, 16, and 32.
4. Repeat on fresh H100 pods/nodes and on a second day.
5. Report coefficient confidence intervals and leave-one-cell-out stability.
6. Measure the actual 1P2D operating points and compare predicted versus
   observed iteration time.

## Claim-bearing policy experiment

1. Warm the output-length estimator until `nHatOut=512` before measurement.
2. Run matched causal and stock arms with the same traces, seeds, replicas,
   model revision, hardware placement, and transport settings.
3. Create controlled overlap so both local and disaggregated decisions occur.
4. Add isolated one-transfer windows and recalibrate loaded transfer cost.
5. Use fixed-snapshot replay or paired intervention to observe the resident
   no-arrival baseline; a single execution cannot reveal the counterfactual.
6. Report TTFT, ITL, E2E, SLO goodput, placement counts, component residuals,
   and confidence intervals over repeated trials.

Here, **stock** is the upstream/default llm-d routing-policy comparison arm.
It is not a claim that vLLM or the entire serving stack is unmodified.
