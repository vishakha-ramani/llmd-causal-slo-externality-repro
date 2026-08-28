package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	causalObservationSchemaVersion = "causal-slo-externality/v1"

	decisionLocal  = "local"
	decisionDisagg = "disaggregated"

	eventProxyRequestReceived = "proxy_request_received"
	eventPrefillDispatched    = "prefill_dispatched"
	eventPrefillCompleted     = "prefill_completed"
	eventDecodeDispatched     = "decode_dispatched"
	eventDecodeFirstEvent     = "decode_first_event"
	eventDecodeTokenEvent     = "decode_token_event"
	eventDecodeCompleted      = "decode_completed"
	eventProxyCompleted       = "proxy_completed"
)

const causalObservationContextKey contextKey = "causal_observation"

type causalRequestObservation struct {
	mu sync.Mutex

	requestID       string
	proxyRequestID  string
	startedAt       time.Time
	placement       string
	decodeEndpoint  string
	prefillEndpoint string

	decodeDispatchedAt time.Time
	firstEventSeen     bool
	lastEventAt        time.Time
	eventIndex         int
	completionTokens   int
	hasCompletion      bool
	statusCode         int
	responseBytes      int64
	completed          bool
	sseRemainder       []byte
}

func newCausalRequestObservation(r *http.Request, proxyRequestID string, startedAt time.Time) *causalRequestObservation {
	requestID := ""
	if r != nil {
		requestID = r.Header.Get(requestHeaderRequestID)
	}
	return &causalRequestObservation{
		requestID:      requestID,
		proxyRequestID: proxyRequestID,
		startedAt:      startedAt,
	}
}

func causalObservationFromRequest(r *http.Request) *causalRequestObservation {
	if r == nil {
		return nil
	}
	observation, _ := r.Context().Value(causalObservationContextKey).(*causalRequestObservation)
	return observation
}

func (o *causalRequestObservation) proxyID() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.proxyRequestID
}

func (o *causalRequestObservation) setProxyID(proxyRequestID string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.proxyRequestID = proxyRequestID
}

func (o *causalRequestObservation) setRoute(placement, decodeEndpoint, prefillEndpoint string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.placement = placement
	o.decodeEndpoint = decodeEndpoint
	o.prefillEndpoint = prefillEndpoint
}

func (o *causalRequestObservation) commonFieldsAt(event string, now time.Time) []any {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.commonFieldsLocked(event, now)
}

func (o *causalRequestObservation) commonFieldsLocked(event string, now time.Time) []any {
	return []any{
		"schemaVersion", causalObservationSchemaVersion,
		"event", event,
		"source", "sidecar",
		"requestID", o.requestID,
		"proxyRequestID", o.proxyRequestID,
		"wallTimeUnixNs", now.UnixNano(),
		"elapsedUs", now.Sub(o.startedAt).Microseconds(),
		"placement", o.placement,
		"decodeEndpoint", o.decodeEndpoint,
		"prefillEndpoint", o.prefillEndpoint,
	}
}

func (o *causalRequestObservation) logAt(logger logr.Logger, event string, now time.Time, values ...any) {
	fields := o.commonFieldsAt(event, now)
	fields = append(fields, values...)
	logger.Info("Causal SLO observation", fields...)
}

func (o *causalRequestObservation) markDecodeDispatchedAt(logger logr.Logger, now time.Time) {
	o.mu.Lock()
	if !o.decodeDispatchedAt.IsZero() {
		o.mu.Unlock()
		return
	}
	o.decodeDispatchedAt = now
	fields := o.commonFieldsLocked(eventDecodeDispatched, now)
	o.mu.Unlock()
	fields = append(fields, "observationClass", "proxy_boundary")
	logger.Info("Causal SLO observation", fields...)
}

func (o *causalRequestObservation) observeResponseBytesAt(logger logr.Logger, data []byte, now time.Time) {
	o.mu.Lock()
	o.responseBytes += int64(len(data))
	o.sseRemainder = append(o.sseRemainder, data...)
	events, remainder := completeSSEEvents(o.sseRemainder)
	o.sseRemainder = remainder
	o.mu.Unlock()

	for _, event := range events {
		o.observeSSEEventAt(logger, event, now)
	}
}

func (o *causalRequestObservation) observeSSEEventAt(logger logr.Logger, raw []byte, now time.Time) {
	payload, ok := sseData(raw)
	if !ok || payload == "[DONE]" {
		return
	}
	completionTokens, hasCompletion := completionTokensFromPayload([]byte(payload))
	tokenBearing := tokenBearingPayload([]byte(payload))

	o.mu.Lock()
	o.eventIndex++
	eventIndex := o.eventIndex
	eventGapUs := int64(0)
	if !o.lastEventAt.IsZero() {
		eventGapUs = now.Sub(o.lastEventAt).Microseconds()
	}
	o.lastEventAt = now
	sinceDecodeDispatchUs := int64(-1)
	if !o.decodeDispatchedAt.IsZero() {
		sinceDecodeDispatchUs = now.Sub(o.decodeDispatchedAt).Microseconds()
	}
	tokenDelta := -1
	if hasCompletion {
		if o.hasCompletion {
			tokenDelta = completionTokens - o.completionTokens
		} else {
			tokenDelta = completionTokens
		}
		if tokenDelta < 0 {
			tokenDelta = 0
		}
		o.completionTokens = completionTokens
		o.hasCompletion = true
	}
	first := !o.firstEventSeen
	o.firstEventSeen = true
	common := o.commonFieldsLocked(eventDecodeTokenEvent, now)
	o.mu.Unlock()

	values := []any{
		"eventIndex", eventIndex,
		"eventGapUs", eventGapUs,
		"sinceDecodeDispatchUs", sinceDecodeDispatchUs,
		"completionTokens", valueOrUnknown(completionTokens, hasCompletion),
		"tokenDelta", tokenDelta,
		"tokenBearing", tokenBearing,
		"observationClass", "sse_event_proxy",
	}
	if first {
		firstFields := append([]any{}, common...)
		firstFields[3] = eventDecodeFirstEvent
		firstFields = append(firstFields, values...)
		logger.Info("Causal SLO observation", firstFields...)
	}
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("Causal SLO observation", append(common, values...)...)
	}
}

func (o *causalRequestObservation) setStatusCode(statusCode int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.statusCode == 0 {
		o.statusCode = statusCode
	}
}

func (o *causalRequestObservation) completeAt(logger logr.Logger, now time.Time) {
	o.mu.Lock()
	if o.completed {
		o.mu.Unlock()
		return
	}
	o.completed = true
	event := eventProxyCompleted
	decodeElapsedUs := int64(-1)
	if !o.decodeDispatchedAt.IsZero() {
		event = eventDecodeCompleted
		decodeElapsedUs = now.Sub(o.decodeDispatchedAt).Microseconds()
	}
	statusCode := o.statusCode
	if statusCode == 0 && o.responseBytes > 0 {
		statusCode = http.StatusOK
	}
	completionTokens := valueOrUnknown(o.completionTokens, o.hasCompletion)
	fields := o.commonFieldsLocked(event, now)
	fields = append(fields,
		"sinceDecodeDispatchUs", decodeElapsedUs,
		"statusCode", statusCode,
		"responseBytes", o.responseBytes,
		"streamEvents", o.eventIndex,
		"completionTokens", completionTokens,
		"observationClass", "proxy_boundary")
	o.mu.Unlock()
	logger.Info("Causal SLO observation", fields...)
}

func valueOrUnknown(value int, known bool) int {
	if !known {
		return -1
	}
	return value
}

func completeSSEEvents(data []byte) ([][]byte, []byte) {
	var events [][]byte
	for {
		lf := bytes.Index(data, []byte("\n\n"))
		crlf := bytes.Index(data, []byte("\r\n\r\n"))
		index, delimiterLength := lf, 2
		if crlf >= 0 && (index < 0 || crlf < index) {
			index, delimiterLength = crlf, 4
		}
		if index < 0 {
			break
		}
		event := append([]byte(nil), data[:index]...)
		events = append(events, event)
		data = data[index+delimiterLength:]
	}
	return events, append([]byte(nil), data...)
}

func sseData(event []byte) (string, bool) {
	normalized := strings.ReplaceAll(string(event), "\r\n", "\n")
	var lines []string
	for _, line := range strings.Split(normalized, "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		lines = append(lines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
	}
	if len(lines) == 0 {
		return "", false
	}
	return strings.Join(lines, "\n"), true
}

func completionTokensFromPayload(payload []byte) (int, bool) {
	var decoded map[string]any
	if json.Unmarshal(payload, &decoded) != nil {
		return 0, false
	}
	usage, ok := decoded[responseFieldUsage].(map[string]any)
	if !ok {
		return 0, false
	}
	for _, key := range []string{responseFieldCompletionTokens, "output_tokens"} {
		if value, ok := toInt(usage[key]); ok {
			return value, true
		}
	}
	return 0, false
}

func tokenBearingPayload(payload []byte) bool {
	var decoded map[string]any
	if json.Unmarshal(payload, &decoded) != nil {
		return false
	}
	if eventType, _ := decoded["type"].(string); strings.Contains(eventType, "output_text.delta") {
		if delta, _ := decoded["delta"].(string); delta != "" {
			return true
		}
	}
	if text, _ := decoded["text"].(string); text != "" {
		return true
	}
	choices, _ := decoded[responseFieldChoices].([]any)
	for _, rawChoice := range choices {
		choice, _ := rawChoice.(map[string]any)
		if text, _ := choice["text"].(string); text != "" {
			return true
		}
		delta, _ := choice[responseFieldDelta].(map[string]any)
		if content, _ := delta[requestFieldContent].(string); content != "" {
			return true
		}
	}
	return false
}

type causalObservedResponseWriter struct {
	http.ResponseWriter
	logger      logr.Logger
	observation *causalRequestObservation
}

func newCausalObservedResponseWriter(
	w http.ResponseWriter, logger logr.Logger, observation *causalRequestObservation,
) *causalObservedResponseWriter {
	return &causalObservedResponseWriter{ResponseWriter: w, logger: logger, observation: observation}
}

func (w *causalObservedResponseWriter) WriteHeader(statusCode int) {
	w.observation.setStatusCode(statusCode)
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *causalObservedResponseWriter) Write(data []byte) (int, error) {
	w.observation.observeResponseBytesAt(w.logger, data, time.Now())
	return w.ResponseWriter.Write(data)
}

func (w *causalObservedResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *causalObservedResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}
