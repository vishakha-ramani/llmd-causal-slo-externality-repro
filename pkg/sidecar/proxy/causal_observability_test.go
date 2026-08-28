package proxy

import (
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type capturedObservation struct {
	message string
	values  map[string]any
}

type observationCaptureSink struct {
	mu      sync.Mutex
	values  []any
	records []capturedObservation
}

func (s *observationCaptureSink) Init(logr.RuntimeInfo) {}
func (s *observationCaptureSink) Enabled(int) bool      { return true }
func (s *observationCaptureSink) Error(error, string, ...any) {
}
func (s *observationCaptureSink) WithName(string) logr.LogSink { return s }
func (s *observationCaptureSink) WithValues(values ...any) logr.LogSink {
	clone := &observationCaptureSink{records: s.records}
	clone.values = append(append([]any{}, s.values...), values...)
	return clone
}
func (s *observationCaptureSink) Info(_ int, message string, values ...any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	all := append(append([]any{}, s.values...), values...)
	fields := make(map[string]any, len(all)/2)
	for i := 0; i+1 < len(all); i += 2 {
		key, ok := all[i].(string)
		if ok {
			fields[key] = all[i+1]
		}
	}
	s.records = append(s.records, capturedObservation{message: message, values: fields})
}

func TestCausalObservationCorrelatesSplitSSEEvents(t *testing.T) {
	start := time.Unix(100, 0)
	req := httptest.NewRequest("POST", ChatCompletionsPath, nil)
	req.Header.Set(requestHeaderRequestID, "epp-7")
	observation := newCausalRequestObservation(req, "proxy-9", start)
	observation.setRoute(decisionDisagg, "decode:8000", "prefill:8000")

	sink := &observationCaptureSink{}
	logger := logr.New(sink)
	observation.markDecodeDispatchedAt(logger, start.Add(10*time.Microsecond))
	observation.observeResponseBytesAt(logger,
		[]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n"),
		start.Add(20*time.Microsecond))
	observation.observeResponseBytesAt(logger,
		[]byte("\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"completion_tokens\":1}}\n\n"),
		start.Add(35*time.Microsecond))

	require.Len(t, sink.records, 4)
	assert.Equal(t, eventDecodeDispatched, sink.records[0].values["event"])
	assert.Equal(t, eventDecodeFirstEvent, sink.records[1].values["event"])
	assert.Equal(t, eventDecodeTokenEvent, sink.records[2].values["event"])
	assert.Equal(t, eventDecodeTokenEvent, sink.records[3].values["event"])
	assert.Equal(t, "epp-7", sink.records[3].values["requestID"])
	assert.Equal(t, "proxy-9", sink.records[3].values["proxyRequestID"])
	assert.Equal(t, int64(25), sink.records[3].values["sinceDecodeDispatchUs"])
	assert.Equal(t, 1, sink.records[3].values["completionTokens"])
	assert.Equal(t, 1, sink.records[3].values["tokenDelta"])
	assert.Equal(t, true, sink.records[3].values["tokenBearing"])
}

func TestCausalObservationCarriesCommonSchemaOnEveryEvent(t *testing.T) {
	start := time.Unix(200, 0)
	req := httptest.NewRequest("POST", ChatCompletionsPath, nil)
	req.Header.Set(requestHeaderRequestID, "epp-1")
	observation := newCausalRequestObservation(req, "proxy-1", start)
	observation.setRoute(decisionLocal, "decode:8000", "")

	sink := &observationCaptureSink{}
	logger := logr.New(sink)
	observation.logAt(logger, eventProxyRequestReceived, start)

	require.Len(t, sink.records, 1)
	for _, key := range []string{
		"schemaVersion", "event", "requestID", "proxyRequestID", "wallTimeUnixNs",
		"elapsedUs", "placement", "decodeEndpoint", "prefillEndpoint",
	} {
		assert.Contains(t, sink.records[0].values, key)
	}
}
