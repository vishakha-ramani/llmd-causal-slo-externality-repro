#!/usr/bin/env python3

"""Join causal-SLO predictions to client, EPP, sidecar, and NIXL observations."""

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_VERSION = "causal-slo-externality/v1"


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def number(record, key, default=None):
    value = record.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(record, key, default=None):
    value = number(record, key, default)
    return default if value is None else int(value)


def decode_json_record(line):
    line = line.strip()
    if not line:
        return None
    candidates = [line]
    brace = line.find("{")
    if brace > 0:
        candidates.append(line[brace:])
    for candidate in candidates:
        try:
            record = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                record, _ = json.JSONDecoder().raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if not isinstance(record, dict):
            continue
        if "event" not in record and isinstance(record.get("log"), str):
            nested = decode_json_record(record["log"])
            if nested is not None:
                return nested
        return record
    return None


def expand_paths(paths):
    expanded = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            expanded.append(path)
    return expanded


def load_events(paths):
    events = []
    rejected = 0
    for path in expand_paths(paths):
        if path.suffix == ".gz":
            stream = gzip.open(path, mode="rt", errors="replace")
        else:
            stream = path.open(errors="replace")
        with stream:
            for line in stream:
                record = decode_json_record(line)
                if record is None:
                    rejected += 1
                    continue
                if record.get("schemaVersion") == SCHEMA_VERSION and record.get("event"):
                    record["_input"] = str(path)
                    events.append(record)
    return events, rejected


def first(records, event):
    matches = [row for row in records if row.get("event") == event]
    if not matches:
        return None
    return min(matches, key=lambda row: integer(row, "wallTimeUnixNs", 0))


def last(records, event):
    matches = [row for row in records if row.get("event") == event]
    if not matches:
        return None
    return max(matches, key=lambda row: integer(row, "wallTimeUnixNs", 0))


def normalize_client_rows(rows):
    normalized = []
    for row in rows:
        request_id_aliases = []
        for name in ("requestID", "upstream_request_id", "request_id", "x_request_id"):
            value = row.get(name)
            if value and value not in request_id_aliases:
                request_id_aliases.append(value)
        if not request_id_aliases:
            continue
        status = row.get("status")
        if status not in (None, "", "ok", "success"):
            continue
        ttft_us = number(row, "ttftUs")
        e2e_us = number(row, "e2eUs")
        if ttft_us is None:
            sent = number(row, "send_time_us")
            first_chunk = number(row, "first_chunk_time_us")
            if sent is not None and first_chunk is not None:
                ttft_us = first_chunk - sent
        if e2e_us is None:
            sent = number(row, "send_time_us")
            last_chunk = number(row, "last_chunk_time_us")
            if sent is not None and last_chunk is not None:
                e2e_us = last_chunk - sent
        normalized.append({
            "requestID": request_id_aliases[0],
            "requestIDAliases": request_id_aliases,
            "ttftUs": ttft_us,
            "e2eUs": e2e_us,
        })
    return normalized


def load_csv(path):
    if path is None:
        return []
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def normalize_transfer_rows(rows):
    aliases = {
        "requestID": ("requestID", "request_id", "upstream_request_id"),
        "successfulTransfersBefore": ("successfulTransfersBefore", "successful_transfers_before"),
        "successfulTransfersAfter": ("successfulTransfersAfter", "successful_transfers_after"),
        "transferTimeSecondsBefore": ("transferTimeSecondsBefore", "transfer_time_seconds_before"),
        "transferTimeSecondsAfter": ("transferTimeSecondsAfter", "transfer_time_seconds_after"),
    }
    normalized = []
    for row in rows:
        item = {}
        for target, names in aliases.items():
            item[target] = next((row[name] for name in names if row.get(name) not in (None, "")), None)
        if item["requestID"]:
            normalized.append(item)
    return normalized


def component_row(request_id, selected, component, predicted, observed, observation_class, **extra):
    residual = observed - predicted
    row = {
        "requestID": request_id,
        "proxyRequestID": selected.get("proxyRequestID", ""),
        "placement": selected.get("placement", ""),
        "decodeEndpoint": selected.get("decodeEndpoint", ""),
        "prefillEndpoint": selected.get("prefillEndpoint", ""),
        "promptTokens": number(selected, "promptTokens"),
        "nHatOut": number(selected, "nHatOut"),
        "component": component,
        "predictedUs": predicted,
        "observedUs": observed,
        "residualUs": residual,
        "absErrorUs": abs(residual),
        "relativeError": residual / predicted if predicted else None,
        "observationClass": observation_class,
    }
    row.update(extra)
    return row


def decode_iteration_estimate(events):
    token_events = sorted(
        (row for row in events if row.get("event") == "decode_token_event"),
        key=lambda row: (integer(row, "wallTimeUnixNs", 0), integer(row, "eventIndex", 0)),
    )
    samples = []
    for event in token_events:
        gap = number(event, "eventGapUs")
        delta = integer(event, "tokenDelta", -1)
        if gap is None or gap <= 0:
            continue
        if delta is not None and delta > 0:
            samples.append((gap / delta, True))
        elif event.get("tokenBearing") is True:
            samples.append((gap, False))
    if not samples:
        return None
    early_samples = samples[:16]
    early_values = [value for value, _ in early_samples]
    lifetime_values = [value for value, _ in samples]
    early_usage_samples = sum(1 for _, usage_attributed in early_samples if usage_attributed)
    lifetime_usage_samples = sum(1 for _, usage_attributed in samples if usage_attributed)
    quality = "early_16_usage_weighted_sse_gap_proxy" if early_usage_samples == len(early_samples) else "early_16_sse_event_gap_proxy"
    lifetime_quality = "lifetime_usage_weighted_sse_gap_proxy" if lifetime_usage_samples == len(samples) else "lifetime_sse_event_gap_proxy"
    return {
        "observedUs": statistics.median(early_values),
        "observedP50Us": percentile(early_values, 0.5),
        "observedP90Us": percentile(early_values, 0.9),
        "samples": len(early_values),
        "usageAttributedSamples": early_usage_samples,
        "observationClass": quality,
        "lifetimeObservedUs": statistics.median(lifetime_values),
        "lifetimeObservedP50Us": percentile(lifetime_values, 0.5),
        "lifetimeObservedP90Us": percentile(lifetime_values, 0.9),
        "lifetimeSamples": len(lifetime_values),
        "lifetimeUsageAttributedSamples": lifetime_usage_samples,
        "lifetimeObservationClass": lifetime_quality,
    }


def sigmoid(value):
    if value >= 0:
        decay = math.exp(-value)
        return 1 / (1 + decay)
    growth = math.exp(value)
    return growth / (1 + growth)


def slo_value(tau_ttft_us, tau_e2e_us, ttft_us, e2e_us):
    value = 1.0
    if tau_ttft_us and tau_ttft_us > 0:
        value *= sigmoid((tau_ttft_us - ttft_us) / tau_ttft_us)
    if tau_e2e_us and tau_e2e_us > 0:
        value *= sigmoid((tau_e2e_us - e2e_us) / tau_e2e_us)
    return value


def max_identity_error(records, total_key, component_keys, multiplier_key=None, additive_key=None):
    errors = []
    for record in records:
        total = number(record, total_key)
        components = [number(record, key) for key in component_keys]
        if total is None or any(value is None for value in components):
            continue
        expected = sum(components)
        if multiplier_key is not None:
            multiplier = number(record, multiplier_key)
            if multiplier is None:
                continue
            expected *= multiplier
        if additive_key is not None:
            additive = number(record, additive_key)
            if additive is None:
                continue
            expected += additive
        errors.append(abs(total - expected))
    return max(errors) if errors else None


def max_formula_error(records, total_key, formula):
    errors = []
    for record in records:
        total = number(record, total_key)
        try:
            expected = formula(record)
        except (TypeError, ValueError):
            continue
        if total is not None and expected is not None:
            errors.append(abs(total - expected))
    return max(errors) if errors else None


def summarize_components(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["component"]].append(row)
    result = {}
    for component, component_rows in sorted(grouped.items()):
        residuals = [row["residualUs"] for row in component_rows]
        absolute = [abs(value) for value in residuals]
        result[component] = {
            "count": len(component_rows),
            "biasUs": statistics.fmean(residuals),
            "maeUs": statistics.fmean(absolute),
            "rmseUs": math.sqrt(statistics.fmean(value * value for value in residuals)),
            "residualP50Us": percentile(residuals, 0.5),
            "absErrorP90Us": percentile(absolute, 0.9),
            "observationClasses": dict(Counter(row["observationClass"] for row in component_rows)),
        }
    return result


def analyze(events, client_rows=None, transfer_rows=None):
    client_rows = normalize_client_rows(client_rows or [])
    transfer_rows = normalize_transfer_rows(transfer_rows or [])
    by_request = defaultdict(list)
    for event in events:
        if event.get("requestID"):
            by_request[event["requestID"]].append(event)

    selected_records = [event for event in events if event.get("event") == "decision_selected"]
    selected_groups = defaultdict(list)
    for record in selected_records:
        selected_groups[record.get("requestID", "")].append(record)
    selected_by_request = {
        request_id: max(records, key=lambda row: integer(row, "wallTimeUnixNs", 0))
        for request_id, records in selected_groups.items()
        if request_id
    }
    event_request_ids = set(by_request)
    proxy_to_request_ids = defaultdict(set)
    for event in events:
        request_id = event.get("requestID")
        proxy_request_id = event.get("proxyRequestID")
        if request_id and proxy_request_id:
            proxy_to_request_ids[proxy_request_id].add(request_id)

    clients = {}
    unmatched_client_rows = []
    ambiguous_client_rows = []
    duplicate_client_request_ids = 0
    proxy_resolved_client_rows = 0
    for row in client_rows:
        directly_resolved_request_ids = set()
        proxy_resolved_request_ids = set()
        resolved_request_ids = set()
        for alias in row.get("requestIDAliases", [row["requestID"]]):
            if alias in event_request_ids:
                directly_resolved_request_ids.add(alias)
            proxy_resolved_request_ids.update(proxy_to_request_ids.get(alias, set()))
        resolved_request_ids.update(directly_resolved_request_ids)
        resolved_request_ids.update(proxy_resolved_request_ids)
        if len(resolved_request_ids) == 0:
            unmatched_client_rows.append(row["requestID"])
            continue
        if len(resolved_request_ids) > 1:
            ambiguous_client_rows.append(row["requestID"])
            continue
        resolved_request_id = resolved_request_ids.pop()
        if resolved_request_id in clients:
            duplicate_client_request_ids += 1
            continue
        if not directly_resolved_request_ids and proxy_resolved_request_ids:
            proxy_resolved_client_rows += 1
        clients[resolved_request_id] = {**row, "requestID": resolved_request_id}
    # The deployment harness sends readiness/smoketest requests before the measured
    # workload. When a client trace is supplied, coverage must describe that trace
    # rather than treating those untraced control requests as failed joins.
    if clients:
        selected_by_request = {
            request_id: selected
            for request_id, selected in selected_by_request.items()
            if request_id in clients
        }
    transfers = defaultdict(list)
    for row in transfer_rows:
        transfers[row["requestID"]].append(row)

    component_rows = []
    objective_rows = []
    fully_joined = 0
    missing = Counter()
    ambiguous_transfer_windows = 0
    attributed_transfer_windows = 0
    placement_mismatches = 0
    proxy_id_mismatches = 0
    missing_remote_proxy_ids = 0

    for request_id, selected in sorted(selected_by_request.items()):
        records = by_request[request_id]
        placement = selected.get("placement")
        epp_first = first(records, "epp_first_token")
        epp_complete = last(records, "epp_complete")
        proxy_received = first(records, "proxy_request_received")
        prefill_complete = last(records, "prefill_completed")
        decode_dispatched = first(records, "decode_dispatched")
        decode_first = first(records, "decode_first_event")
        decode_complete = last(records, "decode_completed")
        joined_selected = dict(selected)
        if proxy_received is not None:
            joined_selected["proxyRequestID"] = proxy_received.get("proxyRequestID", "")
            if proxy_received.get("placement") != placement:
                placement_mismatches += 1
        proxy_ids = {row.get("proxyRequestID") for row in records if row.get("proxyRequestID")}
        if len(proxy_ids) > 1:
            proxy_id_mismatches += 1
        if placement == "disaggregated" and not proxy_ids:
            missing_remote_proxy_ids += 1

        required = {
            "epp_first_token": epp_first,
            "epp_complete": epp_complete,
            "proxy_request_received": proxy_received,
            "decode_dispatched": decode_dispatched,
            "decode_first_event": decode_first,
            "decode_completed": decode_complete,
        }
        if placement == "disaggregated":
            required["prefill_completed"] = prefill_complete
        if all(required.values()):
            fully_joined += 1
        else:
            for name, value in required.items():
                if value is None:
                    missing[name] += 1

        predicted_ttft = number(selected, "projectedTTFTUs")
        if predicted_ttft is not None and epp_first is not None:
            observed = number(epp_first, "observedTTFTUs")
            if observed is not None:
                component_rows.append(component_row(
                    request_id, joined_selected, "ttft_epp", predicted_ttft, observed,
                    "epp_response_callback_composite",
                ))
        client = clients.get(request_id)
        if predicted_ttft is not None and client and client.get("ttftUs") is not None:
            component_rows.append(component_row(
                request_id, joined_selected, "ttft_client", predicted_ttft, client["ttftUs"],
                "client_end_to_end",
            ))

        if placement == "disaggregated" and prefill_complete is not None:
            predicted = number(selected, "prefillCompletionUs")
            observed = number(prefill_complete, "stageElapsedUs")
            if predicted is not None and observed is not None:
                component_rows.append(component_row(
                    request_id, joined_selected, "remote_prefill_completion", predicted, observed,
                    "sidecar_http_round_trip_composite",
                ))

        if decode_first is not None:
            observed = number(decode_first, "sinceDecodeDispatchUs")
            if placement == "disaggregated":
                transfer = number(selected, "transferUs")
                decode_admission = number(selected, "decodeAdmissionUs")
                prefill_completion = number(selected, "prefillCompletionUs")
                first_iteration = number(selected, "firstDecodeIterUs")
                post_processing = number(selected, "outputTokenProcessingUs", 0)
                if None not in (transfer, decode_admission, prefill_completion, first_iteration):
                    predicted = max(transfer, decode_admission - prefill_completion, 0) + first_iteration + post_processing
                    if observed is not None and observed >= 0:
                        component_rows.append(component_row(
                            request_id, joined_selected, "remote_post_prefill_to_first_event", predicted, observed,
                            "decode_dispatch_to_first_sse_includes_kv_sync_queueing_and_engine",
                        ))
            else:
                predicted = number(selected, "projectedTTFTUs")
                if predicted is not None and observed is not None and observed >= 0:
                    component_rows.append(component_row(
                        request_id, joined_selected, "local_dispatch_to_first_event", predicted, observed,
                        "decode_dispatch_to_first_sse_includes_admission_prefill_and_engine",
                    ))

        iteration = decode_iteration_estimate(records)
        predicted_iteration = number(selected, "postJoinIterUs")
        if iteration is not None and predicted_iteration is not None:
            extra = dict(iteration)
            observed = extra.pop("observedUs")
            observation_class = extra.pop("observationClass")
            lifetime_observed = extra.pop("lifetimeObservedUs")
            lifetime_observation_class = extra.pop("lifetimeObservationClass")
            lifetime_extra = {
                "observedP50Us": extra.pop("lifetimeObservedP50Us"),
                "observedP90Us": extra.pop("lifetimeObservedP90Us"),
                "samples": extra.pop("lifetimeSamples"),
                "usageAttributedSamples": extra.pop("lifetimeUsageAttributedSamples"),
            }
            component_rows.append(component_row(
                request_id, joined_selected, "decode_iteration", predicted_iteration, observed,
                observation_class, **extra,
            ))
            component_rows.append(component_row(
                request_id, joined_selected, "decode_iteration_lifetime", predicted_iteration,
                lifetime_observed, lifetime_observation_class, **lifetime_extra,
            ))

        predicted_e2e = number(selected, "projectedE2EUs")
        if predicted_e2e is not None and epp_complete is not None:
            observed = number(epp_complete, "observedE2EUs")
            if observed is not None:
                component_rows.append(component_row(
                    request_id, joined_selected, "completion_epp", predicted_e2e, observed,
                    "epp_response_callback_composite",
                ))
        if predicted_e2e is not None and client and client.get("e2eUs") is not None:
            component_rows.append(component_row(
                request_id, joined_selected, "completion_client", predicted_e2e, client["e2eUs"],
                "client_end_to_end",
            ))

        predicted_own_good = number(selected, "ownGood")
        tau_ttft = number(selected, "tauTTFTUs")
        tau_e2e = number(selected, "tauE2EUs")
        if (
            selected.get("noOwnGood") is not True
            and predicted_own_good is not None
            and client
            and client.get("ttftUs") is not None
            and client.get("e2eUs") is not None
            and tau_ttft is not None
            and tau_e2e is not None
        ):
            actual_own_good = slo_value(tau_ttft, tau_e2e, client["ttftUs"], client["e2eUs"])
            objective_rows.append({
                "requestID": request_id,
                "placement": placement,
                "quantity": "own_good",
                "predictedValue": predicted_own_good,
                "observedValue": actual_own_good,
                "residual": actual_own_good - predicted_own_good,
                "observationClass": "client_realized_slo_value",
            })

        for transfer in transfers.get(request_id, []):
            before_count = integer(transfer, "successfulTransfersBefore")
            after_count = integer(transfer, "successfulTransfersAfter")
            before_seconds = number(transfer, "transferTimeSecondsBefore")
            after_seconds = number(transfer, "transferTimeSecondsAfter")
            if None in (before_count, after_count, before_seconds, after_seconds):
                continue
            if after_count - before_count != 1:
                ambiguous_transfer_windows += 1
                continue
            predicted = number(selected, "transferUs")
            if predicted is None:
                continue
            observed = (after_seconds - before_seconds) * 1_000_000
            component_rows.append(component_row(
                request_id, joined_selected, "transfer_unambiguous", predicted, observed,
                "nixl_counter_delta_single_transfer",
            ))
            attributed_transfer_windows += 1

    identity_records = [event for event in events if event.get("event") == "decision_candidate"]
    if not identity_records:
        identity_records = selected_records
    externality_identity_records = [
        record for record in identity_records if record.get("noExternality") is not True
    ]
    remote_identity_records = [
        record for record in identity_records if record.get("placement") == "disaggregated"
    ]
    identities = {
        "rowsChecked": len(identity_records),
        "maxAbsDecodeIterationErrorUs": max_identity_error(
            identity_records,
            "decodeIterUs",
            ("decodeAlphaUs", "decodeReqUs", "decodeKVUs", "decodePfLinearUs", "decodePfAttnUs"),
        ),
        "maxAbsFirstDecodeIterationErrorUs": max_identity_error(
            identity_records,
            "firstDecodeIterUs",
            ("firstDecodeAlphaUs", "firstDecodeReqUs", "firstDecodeKVUs", "firstDecodePfLinearUs", "firstDecodePfAttnUs"),
        ),
        "maxAbsPrefillIterationErrorUs": max_identity_error(
            identity_records,
            "prefillIterUs",
            ("prefillAlphaUs", "prefillReqUs", "prefillKVUs", "prefillLinearUs", "prefillAttnUs"),
        ),
        "maxAbsPrefillCompletionErrorUs": max_formula_error(
            identity_records,
            "prefillCompletionUs",
            lambda row: number(row, "prefillAdmissionUs")
            + number(row, "prefillChunks") * number(row, "prefillIterUs")
            + number(row, "prefillWorkUs"),
        ),
        "maxAbsRemoteLeadErrorUs": max_formula_error(
            identity_records,
            "remoteLeadUs",
            lambda row: number(row, "prefillCompletionUs") + number(row, "transferUs"),
        ),
        "maxAbsDecodeJoinErrorUs": max_formula_error(
            remote_identity_records,
            "decodeJoinUs",
            lambda row: max(number(row, "remoteLeadUs"), number(row, "decodeAdmissionUs")),
        ),
        "maxAbsProjectedTTFTErrorUs": max_formula_error(
            identity_records,
            "projectedTTFTUs",
            lambda row: (
                number(row, "decodeAdmissionUs")
                + number(row, "prefillChunks") * number(row, "decodeIterUs")
                + number(row, "prefillWorkUs")
                + number(row, "outputTokenProcessingUs")
                if row.get("placement") == "local"
                else number(row, "decodeJoinUs")
                + number(row, "firstDecodeIterUs")
                + number(row, "outputTokenProcessingUs")
            ),
        ),
        "maxAbsProjectedE2EErrorUs": max_formula_error(
            identity_records,
            "projectedE2EUs",
            lambda row: number(row, "projectedTTFTUs")
            + number(row, "nHatOut") * number(row, "firstDecodeIterUs"),
        ),
        "maxAbsExternalitySumError": max_identity_error(
            externality_identity_records,
            "externality",
            ("externalityDecode", "externalityCollocPrefill", "externalityPrefillPool"),
        ),
        "maxAbsScoreError": max_formula_error(
            identity_records,
            "J",
            lambda row: number(row, "V") * number(row, "netGoodCost") + number(row, "capacityTotal", 0),
        ),
    }

    candidate_by_key = {
        (record.get("requestID"), integer(record, "candidateIndex")): record
        for record in events
        if record.get("event") == "decision_candidate"
    }
    resident_groups = defaultdict(list)
    for record in events:
        if record.get("event") == "decision_resident_prediction":
            resident_groups[(record.get("requestID"), integer(record, "candidateIndex"))].append(record)
    resident_identity_errors = []
    for key, records in resident_groups.items():
        candidate = candidate_by_key.get(key)
        if candidate is None or candidate.get("noExternality") is True:
            continue
        expected_by_population = {
            "decode": number(candidate, "externalityDecode", 0),
            "collocated_prefill": number(candidate, "externalityCollocPrefill", 0),
            "prefill_pool": number(candidate, "externalityPrefillPool", 0),
        }
        observed_by_population = defaultdict(float)
        for record in records:
            observed_by_population[record.get("externalityPopulation", "")] += number(record, "contribution", 0)
        for population, expected in expected_by_population.items():
            resident_identity_errors.append(abs(observed_by_population[population] - expected))
    identities["maxAbsResidentContributionSumError"] = (
        max(resident_identity_errors) if resident_identity_errors else None
    )

    resident_predictions = [
        event
        for event in events
        if event.get("event") == "decision_resident_prediction"
        and event.get("selected") is True
        and event.get("requestID") in selected_by_request
    ]
    selected_decision_times = {
        request_id: number(selected, "wallTimeUnixNs")
        for request_id, selected in selected_by_request.items()
    }
    resident_rows = []
    for prediction in resident_predictions:
        resident_id = prediction.get("residentRequestID")
        if not resident_id or prediction.get("predictionEligible") is not True:
            continue
        target_event = "epp_complete" if prediction.get("predictionTarget") == "completion" else "epp_first_token"
        actual = last(by_request.get(resident_id, []), target_event)
        predicted_wall = number(prediction, "candidateTargetWallTimeUnixNs")
        actual_wall = number(actual or {}, "wallTimeUnixNs")
        if predicted_wall is None or actual_wall is None:
            continue
        decision_wall = number(prediction, "wallTimeUnixNs")
        later_arrivals = sum(
            1
            for other_request_id, other_wall in selected_decision_times.items()
            if other_request_id != prediction.get("requestID")
            and decision_wall is not None
            and other_wall is not None
            and decision_wall < other_wall < actual_wall
        )
        observation_class = (
            "selected_world_no_later_logged_arrivals"
            if later_arrivals == 0
            else "selected_world_with_subsequent_interference"
        )
        resident_rows.append({
            "decisionRequestID": prediction.get("requestID", ""),
            "residentRequestID": resident_id,
            "population": prediction.get("externalityPopulation", ""),
            "target": prediction.get("predictionTarget", ""),
            "baselinePredictedWallTimeUnixNs": number(prediction, "baselineTargetWallTimeUnixNs"),
            "candidatePredictedWallTimeUnixNs": predicted_wall,
            "actualWallTimeUnixNs": actual_wall,
            "candidateResidualUs": (actual_wall - predicted_wall) / 1_000,
            "laterSelectedArrivalsBeforeTarget": later_arrivals,
            "predictedDelayUs": (
                predicted_wall - number(prediction, "baselineTargetWallTimeUnixNs")
                if number(prediction, "baselineTargetWallTimeUnixNs") is not None
                else None
            ) / 1_000 if number(prediction, "baselineTargetWallTimeUnixNs") is not None else None,
            "predictedContribution": number(prediction, "contribution"),
            "observationClass": observation_class,
        })
        actual_ttft = number(actual or {}, "observedTTFTUs")
        actual_e2e = number(actual or {}, "observedE2EUs")
        tau_ttft = number(prediction, "residentTauTTFTUs")
        tau_e2e = number(prediction, "residentTauE2EUs")
        predicted_value = number(prediction, "candidateValue")
        actual_value = None
        if prediction.get("predictionTarget") == "first_token":
            if actual_ttft is not None and tau_ttft is not None:
                actual_value = 1.0 if tau_ttft <= 0 else sigmoid((tau_ttft - actual_ttft) / tau_ttft)
        elif None not in (actual_ttft, actual_e2e, tau_ttft, tau_e2e):
            actual_value = slo_value(tau_ttft, tau_e2e, actual_ttft, actual_e2e)
        if predicted_value is not None and actual_value is not None:
            resident_rows[-1]["candidatePredictedValue"] = predicted_value
            resident_rows[-1]["actualSelectedWorldValue"] = actual_value
            resident_rows[-1]["candidateValueResidual"] = actual_value - predicted_value

    event_counts = Counter(event.get("event") for event in events)
    selected_n_hat_out = [
        value
        for selected in selected_by_request.values()
        if (value := number(selected, "nHatOut")) is not None
    ]
    selected_prompt_tokens = [
        value
        for selected in selected_by_request.values()
        if (value := number(selected, "promptTokens")) is not None
    ]
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "coverage": {
            "selectedRequests": len(selected_by_request),
            "fullyJoinedRequests": fully_joined,
            "duplicateSelectedRequestIDs": sum(len(rows) - 1 for rows in selected_groups.values()),
            "placementMismatches": placement_mismatches,
            "proxyIDMismatches": proxy_id_mismatches,
            "missingRemoteProxyIDs": missing_remote_proxy_ids,
            "missingRequiredEvents": dict(sorted(missing.items())),
            "clientRowsEligible": len(client_rows),
            "clientRowsResolved": len(clients),
            "clientRowsResolvedViaProxyID": proxy_resolved_client_rows,
            "clientRowsJoined": sum(1 for request_id in selected_by_request if request_id in clients),
            "unmatchedClientRequestIDs": sorted(unmatched_client_rows),
            "ambiguousClientRequestIDs": sorted(ambiguous_client_rows),
            "duplicateClientRequestIDs": duplicate_client_request_ids,
            "placementCounts": dict(sorted(Counter(
                selected.get("placement", "unknown") for selected in selected_by_request.values()
            ).items())),
            "eventCounts": dict(sorted(event_counts.items())),
        },
        "identities": identities,
        "predictionInputs": {
            "nHatOutCounts": dict(sorted(Counter(selected_n_hat_out).items())),
            "promptTokensMin": min(selected_prompt_tokens) if selected_prompt_tokens else None,
            "promptTokensMax": max(selected_prompt_tokens) if selected_prompt_tokens else None,
        },
        "iterationStateCoverage": {
            "candidateRows": len(identity_records),
            "candidateRowsWithDecodePrefillTerms": sum(
                1
                for row in identity_records
                if number(row, "decodeSPf", 0) != 0 or number(row, "decodeUPf", 0) != 0
            ),
            "candidateRowsWithPrefillResidentTerms": sum(
                1
                for row in identity_records
                if number(row, "prefillSPf", 0) != 0 or number(row, "prefillUPf", 0) != 0
            ),
            "interpretation": "Zero mixed-state rows mean this run exercised the clean-snapshot gate; held-out calibration, not this workload, tests additive decode-plus-prefill batching.",
        },
        "components": summarize_components(component_rows),
        "componentsByPlacement": {
            placement: summarize_components([
                row for row in component_rows if row.get("placement") == placement
            ])
            for placement in sorted({row.get("placement") for row in component_rows})
        },
        "objective": {
            "rows": objective_rows,
            "ownGoodResiduals": summarize_residual_values(
                [row["residual"] for row in objective_rows if row["quantity"] == "own_good"]
            ),
            "ownGoodResidualsByPlacement": {
                placement: summarize_residual_values([
                    row["residual"]
                    for row in objective_rows
                    if row["quantity"] == "own_good" and row.get("placement") == placement
                ])
                for placement in sorted({row.get("placement") for row in objective_rows})
            },
        },
        "externality": {
            "selectedResidentPredictions": len(resident_predictions),
            "actualTargetCoverage": len(resident_rows),
            "candidateResiduals": summarize_residual_values(
                [row["candidateResidualUs"] for row in resident_rows]
            ),
            "candidateResidualsByPopulation": {
                population: summarize_residual_values([
                    row["candidateResidualUs"]
                    for row in resident_rows
                    if row.get("population") == population
                ])
                for population in sorted({row.get("population") for row in resident_rows})
            },
            "candidateResidualsByTarget": {
                target: summarize_residual_values([
                    row["candidateResidualUs"]
                    for row in resident_rows
                    if row.get("target") == target
                ])
                for target in sorted({row.get("target") for row in resident_rows})
            },
            "candidateResidualsByObservationClass": {
                observation_class: summarize_residual_values([
                    row["candidateResidualUs"]
                    for row in resident_rows
                    if row.get("observationClass") == observation_class
                ])
                for observation_class in sorted({row.get("observationClass") for row in resident_rows})
            },
            "interpretation": "Rows with later logged arrivals are confounded by subsequent interference. Even no-later-arrival rows validate only the selected-world target prediction; baseline counterfactuals require a fixed-snapshot replay or paired run.",
            "rows": resident_rows,
        },
        "transferAttribution": {
            "unambiguousWindows": attributed_transfer_windows,
            "ambiguousWindows": ambiguous_transfer_windows,
            "rule": "A NIXL metric window is attributed only when successfulTransfersAfter-successfulTransfersBefore equals one.",
        },
        "limitations": [
            "Scheduler admission, scheduled batch contents, and pure engine iteration duration are not directly observable without vLLM instrumentation.",
            "Decode dispatch-to-first-event includes queueing, transfer synchronization, engine work, HTTP, and proxy buffering.",
            "SSE event gaps are token proxies; one event may contain zero, one, or multiple model tokens.",
            "An aggregate NIXL delta is request-attributed only for a single-transfer measurement window.",
            "A single selected execution cannot reveal the realized counterfactual resident externality.",
        ],
    }
    return report, component_rows


def summarize_residual_values(values):
    if not values:
        return {"count": 0}
    absolute = [abs(value) for value in values]
    return {
        "count": len(values),
        "biasUs": statistics.fmean(values),
        "maeUs": statistics.fmean(absolute),
        "rmseUs": math.sqrt(statistics.fmean(value * value for value in values)),
        "residualP50Us": percentile(values, 0.5),
        "absErrorP90Us": percentile(absolute, 0.9),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def strict_failures(report, identity_tolerance_us):
    failures = []
    coverage = report["coverage"]
    if coverage["duplicateSelectedRequestIDs"]:
        failures.append("duplicate decision_selected events")
    if coverage["placementMismatches"]:
        failures.append("EPP/sidecar placement mismatches")
    if coverage["proxyIDMismatches"]:
        failures.append("multiple proxy IDs for one request")
    if coverage["missingRemoteProxyIDs"]:
        failures.append("remote requests without a proxy ID")
    if coverage["fullyJoinedRequests"] != coverage["selectedRequests"]:
        failures.append("incomplete request joins")
    if coverage["clientRowsJoined"] != coverage["clientRowsEligible"]:
        failures.append("successful client rows without decision joins")
    if coverage["ambiguousClientRequestIDs"]:
        failures.append("ambiguous client request ID aliases")
    if coverage["duplicateClientRequestIDs"]:
        failures.append("duplicate client request IDs")
    for name, value in report["identities"].items():
        if name == "rowsChecked" or value is None:
            continue
        if value > identity_tolerance_us:
            failures.append(f"{name}={value} exceeds {identity_tolerance_us}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", help="JSON-log files or directories")
    parser.add_argument("--client-trace", help="trace_data.csv from the workload client")
    parser.add_argument("--transfer-windows", help="CSV containing before/after NIXL counter snapshots")
    parser.add_argument("--report-out", help="write the JSON report to this path")
    parser.add_argument("--rows-out", help="write one prediction-observation row per component")
    parser.add_argument("--externality-rows-out", help="write selected resident prediction rows")
    parser.add_argument("--strict", action="store_true", help="fail on incomplete joins or identity errors")
    parser.add_argument("--identity-tolerance-us", type=float, default=1e-6)
    args = parser.parse_args()

    events, rejected = load_events(args.events)
    clients = load_csv(args.client_trace)
    transfers = load_csv(args.transfer_windows)
    report, rows = analyze(events, clients, transfers)
    report["input"] = {"eventRecords": len(events), "nonEventLines": rejected}

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out:
        output = Path(args.report_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    if args.rows_out:
        write_csv(args.rows_out, rows)
    if args.externality_rows_out:
        write_csv(args.externality_rows_out, report["externality"]["rows"])
    print(rendered)

    if args.strict:
        failures = strict_failures(report, args.identity_tolerance_us)
        if failures:
            print("strict validation failed: " + "; ".join(failures), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
