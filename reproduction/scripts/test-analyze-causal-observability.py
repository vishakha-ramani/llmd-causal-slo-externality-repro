#!/usr/bin/env python3

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze-causal-observability.py")
SPEC = importlib.util.spec_from_file_location("causal_observability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CausalObservabilityAnalyzerTest(unittest.TestCase):
    def test_loads_gzip_event_bundle(self):
        event = {
            "schemaVersion": "causal-slo-externality/v1",
            "event": "decision_selected",
            "requestID": "compressed-request",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps(event) + "\n")

            events, rejected = MODULE.load_events([path])

        self.assertEqual(rejected, 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["requestID"], "compressed-request")

    def test_joins_prediction_to_each_proxy_observation(self):
        common = {
            "schemaVersion": "causal-slo-externality/v1",
            "requestID": "req-1",
            "proxyRequestID": "proxy-1",
            "placement": "disaggregated",
            "decodeEndpoint": "decode-0",
            "prefillEndpoint": "prefill-0",
        }
        selected = {
            **common,
            "source": "epp",
            "event": "decision_selected",
            "wallTimeUnixNs": 1_000_000_000,
            "elapsedUs": 50,
            "projectedTTFTUs": 100,
            "prefillAdmissionUs": 10,
            "prefillChunks": 2,
            "prefillIterUs": 20,
            "prefillWorkUs": 10,
            "prefillCompletionUs": 60,
            "transferUs": 5,
            "remoteLeadUs": 65,
            "decodeAdmissionUs": 90,
            "decodeJoinUs": 90,
            "firstDecodeIterUs": 10,
            "postJoinIterUs": 10,
            "projectedE2EUs": 300,
            "outputTokenProcessingUs": 0,
            "decodeAlphaUs": 1,
            "decodeReqUs": 2,
            "decodeKVUs": 3,
            "decodePfLinearUs": 4,
            "decodePfAttnUs": 0,
            "decodeIterUs": 10,
            "firstDecodeAlphaUs": 1,
            "firstDecodeReqUs": 2,
            "firstDecodeKVUs": 3,
            "firstDecodePfLinearUs": 4,
            "firstDecodePfAttnUs": 0,
            "prefillAlphaUs": 5,
            "prefillReqUs": 0,
            "prefillKVUs": 0,
            "prefillLinearUs": 15,
            "prefillAttnUs": 0,
            "externalityDecode": 0.1,
            "externalityCollocPrefill": 0.2,
            "externalityPrefillPool": 0.3,
            "externality": 0.6,
            "ownGood": 0.5,
            "netGoodCost": 0.1,
            "capacityTotal": 0,
            "V": 8,
            "J": 0.8,
            "nHatOut": 20,
        }
        events = [
            selected,
            {
                **common,
                "event": "decision_selected",
                "requestID": "deployment-smoketest",
                "proxyRequestID": "proxy-smoketest",
                "wallTimeUnixNs": 900_000_000,
            },
            {**common, "source": "epp", "event": "epp_first_token", "wallTimeUnixNs": 1_100_000_000, "elapsedUs": 120, "observedTTFTUs": 120},
            {**common, "source": "epp", "event": "epp_complete", "wallTimeUnixNs": 1_300_000_000, "elapsedUs": 330, "observedE2EUs": 330},
            {**common, "source": "sidecar", "event": "proxy_request_received", "wallTimeUnixNs": 1_000_000_000, "elapsedUs": 0},
            {**common, "source": "sidecar", "event": "prefill_dispatched", "wallTimeUnixNs": 1_000_010_000, "elapsedUs": 10},
            {**common, "source": "sidecar", "event": "prefill_completed", "wallTimeUnixNs": 1_000_080_000, "elapsedUs": 80, "stageElapsedUs": 70},
            {**common, "source": "sidecar", "event": "decode_dispatched", "wallTimeUnixNs": 1_000_090_000, "elapsedUs": 90},
            {**common, "source": "sidecar", "event": "decode_first_event", "wallTimeUnixNs": 1_000_105_000, "elapsedUs": 105, "sinceDecodeDispatchUs": 15},
            {**common, "source": "sidecar", "event": "decode_token_event", "wallTimeUnixNs": 1_000_105_000, "elapsedUs": 105, "eventIndex": 1, "eventGapUs": 0, "tokenDelta": -1, "tokenBearing": False},
            {**common, "source": "sidecar", "event": "decode_token_event", "wallTimeUnixNs": 1_000_115_000, "elapsedUs": 115, "eventIndex": 2, "eventGapUs": 10, "tokenDelta": 1, "tokenBearing": True},
            {**common, "source": "sidecar", "event": "decode_token_event", "wallTimeUnixNs": 1_000_125_000, "elapsedUs": 125, "eventIndex": 3, "eventGapUs": 10, "tokenDelta": 1, "tokenBearing": True},
            {**common, "source": "sidecar", "event": "decode_completed", "wallTimeUnixNs": 1_000_330_000, "elapsedUs": 330, "sinceDecodeDispatchUs": 240},
            {
                **common,
                "source": "epp",
                "event": "decision_resident_prediction",
                "residentRequestID": "resident-1",
                "externalityPopulation": "decode",
                "selected": True,
                "predictionTarget": "completion",
                "predictionEligible": True,
                "candidateTargetWallTimeUnixNs": 2_000_000_000,
                "baselineTargetWallTimeUnixNs": 1_900_000_000,
                "contribution": 0.2,
            },
            {
                **common,
                "source": "epp",
                "event": "epp_complete",
                "requestID": "resident-1",
                "wallTimeUnixNs": 2_100_000_000,
                "elapsedUs": 500,
                "observedE2EUs": 500,
            },
        ]
        clients = [{"requestID": "req-1", "ttftUs": 130, "e2eUs": 350}]
        transfers = [{
            "requestID": "req-1",
            "successfulTransfersBefore": 7,
            "successfulTransfersAfter": 8,
            "transferTimeSecondsBefore": 1.0,
            "transferTimeSecondsAfter": 1.000006,
        }]

        report, rows = MODULE.analyze(events, clients, transfers)
        by_component = {row["component"]: row for row in rows}

        self.assertEqual(by_component["ttft_epp"]["residualUs"], 20)
        self.assertEqual(by_component["ttft_client"]["residualUs"], 30)
        self.assertEqual(by_component["remote_prefill_completion"]["residualUs"], 10)
        self.assertEqual(by_component["remote_post_prefill_to_first_event"]["residualUs"], -25)
        self.assertEqual(by_component["decode_iteration"]["observedUs"], 10)
        self.assertEqual(by_component["decode_iteration_lifetime"]["observedUs"], 10)
        self.assertEqual(by_component["completion_epp"]["residualUs"], 30)
        self.assertEqual(by_component["completion_client"]["residualUs"], 50)
        self.assertAlmostEqual(by_component["transfer_unambiguous"]["residualUs"], 1)
        self.assertEqual(report["coverage"]["selectedRequests"], 1)
        self.assertEqual(report["coverage"]["fullyJoinedRequests"], 1)
        self.assertEqual(report["coverage"]["missingRequiredEvents"], {})
        self.assertEqual(report["coverage"]["placementCounts"], {"disaggregated": 1})
        self.assertEqual(report["externality"]["actualTargetCoverage"], 1)
        self.assertEqual(report["externality"]["candidateResidualsByPopulation"]["decode"]["count"], 1)
        self.assertEqual(report["identities"]["maxAbsScoreError"], 0)

    def test_resolves_disaggregated_client_response_proxy_id(self):
        common = {
            "schemaVersion": "causal-slo-externality/v1",
            "requestID": "req-remote",
            "proxyRequestID": "proxy-remote",
            "placement": "disaggregated",
        }
        events = [
            {
                **common,
                "event": "decision_selected",
                "wallTimeUnixNs": 1,
                "projectedTTFTUs": 10,
                "projectedE2EUs": 20,
            },
            {**common, "event": "epp_first_token", "wallTimeUnixNs": 2, "observedTTFTUs": 11},
            {**common, "event": "epp_complete", "wallTimeUnixNs": 8, "observedE2EUs": 21},
            {**common, "event": "proxy_request_received", "wallTimeUnixNs": 2},
            {**common, "event": "prefill_completed", "wallTimeUnixNs": 3},
            {**common, "event": "decode_dispatched", "wallTimeUnixNs": 4},
            {**common, "event": "decode_first_event", "wallTimeUnixNs": 5},
            {**common, "event": "decode_completed", "wallTimeUnixNs": 7},
        ]
        clients = [{
            "upstream_request_id": "proxy-remote",
            "x_request_id": "client-generated-id",
            "ttftUs": 12,
            "e2eUs": 22,
        }]

        report, rows = MODULE.analyze(events, clients)

        self.assertEqual(report["coverage"]["clientRowsEligible"], 1)
        self.assertEqual(report["coverage"]["clientRowsResolved"], 1)
        self.assertEqual(report["coverage"]["clientRowsResolvedViaProxyID"], 1)
        self.assertEqual(report["coverage"]["clientRowsJoined"], 1)
        self.assertEqual(report["coverage"]["selectedRequests"], 1)
        self.assertTrue(any(row["component"] == "ttft_client" for row in rows))
        self.assertEqual(MODULE.strict_failures(report, 1e-6), [])

    def test_strict_rejects_unmatched_successful_client_row(self):
        report, _ = MODULE.analyze(
            [{
                "schemaVersion": "causal-slo-externality/v1",
                "event": "decision_selected",
                "requestID": "deployment-smoketest",
                "placement": "local",
            }],
            [{"requestID": "missing-workload-request", "ttftUs": 1, "e2eUs": 2}],
        )

        self.assertIn(
            "successful client rows without decision joins",
            MODULE.strict_failures(report, 1e-6),
        )

    def test_transfer_with_multiple_completions_is_not_attributed(self):
        report, rows = MODULE.analyze(
            [{
                "schemaVersion": "causal-slo-externality/v1",
                "source": "epp",
                "event": "decision_selected",
                "requestID": "req-2",
                "placement": "disaggregated",
                "projectedTTFTUs": 1,
                "projectedE2EUs": 2,
            }],
            [],
            [{
                "requestID": "req-2",
                "successfulTransfersBefore": 1,
                "successfulTransfersAfter": 3,
                "transferTimeSecondsBefore": 2,
                "transferTimeSecondsAfter": 2.1,
            }],
        )
        self.assertFalse(any(row["component"] == "transfer_unambiguous" for row in rows))
        self.assertEqual(report["transferAttribution"]["ambiguousWindows"], 1)


if __name__ == "__main__":
    unittest.main()
