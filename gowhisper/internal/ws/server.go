package ws

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"

	"github.com/obiente/translate/server-go/internal/audio"
	"github.com/obiente/translate/server-go/internal/config"
	"github.com/obiente/translate/server-go/internal/translation"
	weng "github.com/obiente/translate/server-go/internal/whisper"
)

// min returns the minimum of two integers (Go 1.21+)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers (Go 1.21+)
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type Server struct {
	cfg        config.Config
	upgrader   websocket.Upgrader
	mu         sync.RWMutex
	rooms      map[string]map[*websocket.Conn]*clientMeta
	translator *translation.Client
}

type clientMeta struct {
	peerID    string
	peerLabel string
	channelID string
}

func NewServer(cfg config.Config) *Server {
	return &Server{
		cfg: cfg,
		upgrader: websocket.Upgrader{
			CheckOrigin:     func(r *http.Request) bool { return true },
			ReadBufferSize:  1024 * 16,
			WriteBufferSize: 1024 * 16,
		},
		rooms:      make(map[string]map[*websocket.Conn]*clientMeta),
		translator: translation.New(cfg.TranslationBaseURL, cfg.TranslationTimeoutSec),
	}
}

func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Error().Err(err).Msg("ws upgrade failed")
		return
	}
	defer conn.Close()

	_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error { _ = conn.SetReadDeadline(time.Now().Add(60 * time.Second)); return nil })

	// session state
	var (
		roomID  string
		meta    = &clientMeta{}
		engine  weng.Engine
		samples []float32
		seq     int

		// Session configuration from start event
		sourceLanguage          string
		targetLanguages         []string
		translationAlternatives int

		// Track transcription state
		fullTranscript   string                // accumulated full transcription across all chunks
		finalizedText    string                // text that has been marked as final (don't reset until client final)
		processedSamples int                   // offset: how many samples have been transcribed
		workerStarted    bool                  // track if background worker is running
		samplesMu        sync.Mutex            // protect samples buffer and offset
		exitWorker       = make(chan struct{}) // signal worker to exit
	)

	// channel to receive transcription results from async workers
	type transcriptResult struct {
		delta    string
		full     string
		lang     string
		isFinal  bool
		sequence int
		trans    map[string]any
	}
	resCh := make(chan transcriptResult, 16) // Buffer for parallel results

	// Background transcription worker - runs continuously like WhisperLive
	// Processes new audio as it arrives, tracking offset to avoid re-processing
	startTranscriptionWorker := func() {
		if workerStarted {
			return
		}
		workerStarted = true

		go func() {
			log.Info().Msg("transcription worker started")
			defer log.Info().Msg("transcription worker stopped")

			// Automatic isFinal detection based on text stability (like WhisperLive)
			var lastStableText string
			var stableTextCount int
			const stableThreshold = 2 // Number of passes before marking as final

			for {
				select {
				case <-exitWorker:
					return
				default:
				}

				// Get new unprocessed audio
				samplesMu.Lock()
				if processedSamples >= len(samples) {
					samplesMu.Unlock()
					time.Sleep(50 * time.Millisecond) // Wait for new audio
					continue
				}

				// Calculate context window: include recent context + new audio
				// This is critical for sentence continuity!
				contextSamples := 16000 * 10 // 10 seconds of context
				startOffset := processedSamples - contextSamples
				if startOffset < 0 {
					startOffset = 0
				}

				// Extract CONTEXT + NEW audio for transcription
				audioWithContext := make([]float32, len(samples)-startOffset)
				copy(audioWithContext, samples[startOffset:])
				newSamplesCount := len(samples) - processedSamples
				samplesMu.Unlock()

				audioDuration := float64(len(audioWithContext)) / 16000.0
				newDuration := float64(newSamplesCount) / 16000.0

				// Only process if we have meaningful new audio (>0.3s)
				if newDuration < 0.3 {
					time.Sleep(50 * time.Millisecond)
					continue
				}

				log.Debug().
					Int("new_samples", newSamplesCount).
					Int("total_with_context", len(audioWithContext)).
					Int("processed_offset", processedSamples).
					Float64("context_duration", audioDuration).
					Float64("new_duration", newDuration).
					Msg("worker: processing audio with context")

				// Callback to collect segments (no immediate emission to avoid empty fullText)
				var segments []string
				var lastLang string
				tokenCallback := func(text string, lang string) {
					segments = append(segments, text)
					lastLang = lang
				}

				// Process audio WITH CONTEXT for sentence continuity
				if err := engine.Stream(audioWithContext, tokenCallback); err != nil {
					log.Warn().Err(err).Msg("worker: stream failed")
					time.Sleep(50 * time.Millisecond)
					continue
				}

				// Build result
				result := transcriptResult{
					lang:     lastLang,
					isFinal:  false,
					sequence: seq,
					trans:    map[string]any{},
				}
				if result.lang == "" {
					result.lang = "en"
				}

				fullText := strings.TrimSpace(strings.Join(segments, " "))
				if fullText != "" {
					samplesMu.Lock()
					prevFull := fullTranscript
					prevFinalized := finalizedText
					fullTranscript = fullText
					samplesMu.Unlock()

					// Calculate delta: new text since last emission (excluding already finalized)
					if prevFull != "" && strings.HasPrefix(fullText, prevFull) {
						result.delta = strings.TrimSpace(strings.TrimPrefix(fullText, prevFull))
					} else if prevFinalized != "" && strings.HasPrefix(fullText, prevFinalized) {
						// If full text changed but still starts with finalized, delta is everything after finalized
						result.delta = strings.TrimSpace(strings.TrimPrefix(fullText, prevFinalized))
					} else {
						result.delta = fullText
					}
					result.full = fullText

					// Automatic isFinal detection: find the last complete sentence and check if IT has stabilized
					// This allows emitting final during long speeches when individual sentences complete

					// Find the position of the last sentence-ending punctuation
					sentenceEndings := []rune{'.', '!', '?', '。', '！', '？', '♪', '*', ']', ')'}
					lastSentenceEndPos := -1
					for i := len(fullText) - 1; i >= 0; i-- {
						for _, ch := range sentenceEndings {
							if rune(fullText[i]) == ch {
								lastSentenceEndPos = i
								break
							}
						}
						if lastSentenceEndPos >= 0 {
							break
						}
					}

					// If we found a sentence ending, extract up to and including that ending
					if lastSentenceEndPos >= 0 {
						// Extract the text up to the sentence ending (this is the "complete sentence")
						completedSentence := fullText[:lastSentenceEndPos+1]

						// Only track sentences that are NEW (not already finalized)
						if prevFinalized == "" || !strings.HasPrefix(completedSentence, prevFinalized) || len(completedSentence) > len(prevFinalized) {
							// Check if this completed sentence has stabilized
							if completedSentence == lastStableText {
								stableTextCount++
								log.Debug().
									Int("stable_count", stableTextCount).
									Str("stable_sentence", completedSentence[max(0, len(completedSentence)-50):]).
									Msg("worker: completed sentence stability detected")
							} else {
								// Sentence changed, update tracking
								stableTextCount = 1
								lastStableText = completedSentence
							}

							// Mark as final if the completed sentence hasn't changed over multiple passes
							if stableTextCount >= stableThreshold {
								result.isFinal = true
								samplesMu.Lock()
								finalizedText = completedSentence // Store what we've finalized
								samplesMu.Unlock()

								// For sentence-level finals, send the completed sentence as text (not delta)
								// This matches WhisperLive behavior for sentence finals
								result.delta = completedSentence
								result.full = completedSentence

								log.Info().
									Int("stable_count", stableTextCount).
									Str("final_sentence", completedSentence).
									Int("sentence_length", len(completedSentence)).
									Msg("worker: emitting automatic isFinal (sentence stabilized)")
								// Reset stability tracking after emitting final
								stableTextCount = 0
								lastStableText = ""
							}
						}
					} else {
						// No sentence ending found yet, reset tracking
						if lastStableText != "" {
							log.Debug().Msg("worker: no sentence ending found, resetting stability")
							lastStableText = ""
							stableTextCount = 0
						}
					} // Translations
					if s.cfg.TranslationEnabled && len(targetLanguages) > 0 && result.delta != "" {
						ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.TranslationTimeoutSec)*time.Second)
						alts := translationAlternatives
						if alts < 0 {
							alts = 0
						}
						// Pass source language hint (if available) to the translation service
						if m, err := s.translator.Translate(ctx, result.delta, result.lang, targetLanguages, alts); err != nil {
							log.Warn().Err(err).Str("delta", result.delta).Msg("worker: translation request failed")
						} else if m == nil || len(m) == 0 {
							log.Debug().Str("delta", result.delta).Msg("worker: translation returned no results")
						} else {
							for k, v := range m {
								result.trans[k] = v
							}
						}
						cancel()
					}

					// Send result
					log.Debug().
						Str("delta", result.delta).
						Str("full", result.full[:min(100, len(result.full))]).
						Bool("is_final", result.isFinal).
						Int("delta_len", len(result.delta)).
						Int("full_len", len(result.full)).
						Msg("worker: sending result")
					select {
					case resCh <- result:
					case <-time.After(1 * time.Second):
					}
				} // Update processed offset normally - buffer reset happens in drainResults
				samplesMu.Lock()
				processedSamples += newSamplesCount
				samplesMu.Unlock()

				log.Debug().
					Int("processed_samples", processedSamples).
					Str("full_text", fullText).
					Bool("is_final", result.isFinal).
					Msg("worker: chunk complete")
			}
		}()
	}

	// Helper to drain any ready transcription results
	drainResults := func() {
		for {
			select {
			case r := <-resCh:
				// Send to client
				payload := map[string]any{
					"type":         "transcript",
					"text":         r.delta,
					"fullText":     r.full,
					"language":     r.lang,
					"isFinal":      r.isFinal,
					"sequence":     r.sequence,
					"translations": r.trans,
				}
				if err := conn.WriteJSON(payload); err != nil {
					log.Warn().Err(err).Msg("failed to send transcript")
				} else {
					log.Debug().Str("delta", r.delta).Msg("sent transcript to client")
				}

				// Broadcast to room if joined
				if roomID != "" {
					rp := map[string]any{
						"type":         "room_transcript",
						"room_id":      roomID,
						"peer_id":      meta.peerID,
						"peer_label":   meta.peerLabel,
						"channel_id":   meta.channelID,
						"text":         r.delta,
						"fullText":     r.full,
						"language":     r.lang,
						"isFinal":      r.isFinal,
						"sequence":     r.sequence,
						"translations": r.trans,
					}
					s.broadcast(roomID, conn, meta.peerID, rp)
				}

				// Note: We do NOT reset buffer on automatic isFinal
				// Buffer only resets when client sends explicit final marker
				// This preserves context for continued speech
			default:
				return
			}
		}
	} // Simple loop handling JSON control messages and chunk frames
	for {
		// First, drain any ready transcription results before reading next message
		drainResults()

		mt, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				s.leaveRoom(roomID, conn)
				close(exitWorker) // Signal worker to exit
				if engine != nil {
					_ = engine.Close()
				}
				return
			}
			log.Warn().Err(err).Msg("ws read error")
			s.leaveRoom(roomID, conn)
			close(exitWorker) // Signal worker to exit
			if engine != nil {
				_ = engine.Close()
			}
			return
		}
		// Bump read deadline on any activity
		_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		if mt != websocket.TextMessage {
			continue
		}
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			_ = conn.WriteJSON(map[string]any{"type": "error", "detail": "invalid json"})
			continue
		}
		switch msg["type"] {
		case "ping":
			// keepalive from client: reset deadline and respond
			_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
			_ = conn.WriteJSON(map[string]any{"type": "pong", "ts": msg["ts"]})
		case "start":
			// Capture session configuration
			if v, ok := msg["channel_id"].(string); ok {
				meta.channelID = v
			}
			if v, ok := msg["language"].(string); ok {
				sourceLanguage = v
			}
			if v, ok := msg["target_languages"].([]any); ok {
				targetLanguages = make([]string, 0, len(v))
				for _, lang := range v {
					if langStr, ok := lang.(string); ok {
						targetLanguages = append(targetLanguages, langStr)
					}
				}
			}
			if v, ok := msg["translation_alternatives"].(float64); ok {
				translationAlternatives = int(v)
			}

			log.Info().
				Str("channel", meta.channelID).
				Str("source_lang", sourceLanguage).
				Strs("target_langs", targetLanguages).
				Int("alternatives", translationAlternatives).
				Msg("session started with configuration")

			// Configure engine language if already initialized
			if engine != nil && sourceLanguage != "" {
				engine.SetLanguage(sourceLanguage)
			}

			_ = conn.WriteJSON(map[string]any{"type": "started"})
		case "join_room":
			// Idempotent room join ack
			rid, _ := msg["room_id"].(string)
			if rid == "" {
				break
			}
			if v, ok := msg["peer_id"].(string); ok {
				meta.peerID = v
			}
			if v, ok := msg["peer_label"].(string); ok {
				meta.peerLabel = v
			}
			s.joinRoom(rid, conn, meta)
			roomID = rid
			_ = conn.WriteJSON(map[string]any{"type": "room_joined", "room_id": roomID, "peer_id": meta.peerID, "peer_label": meta.peerLabel})
		case "leave_room":
			s.leaveRoom(roomID, conn)
			roomID = ""
			_ = conn.WriteJSON(map[string]any{"type": "room_left"})
		case "chunk":
			// Decode base64
			b64, _ := msg["data"].(string)
			if b64 == "" {
				continue
			}

			// Inflate audio (WAV or PCM16)
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				_ = conn.WriteJSON(map[string]any{"type": "error", "detail": "invalid base64 audio"})
				continue
			}

			var (
				pcm []float32
				sr  int
			)
			if mt, _ := msg["mime_type"].(string); mt == "audio/pcm" || mt == "audio/L16" || mt == "audio/pcm16" {
				sr = int(asFloat(msg["sample_rate"]))
				pcm, sr, err = audio.DecodePCM16LEToFloat32(raw, sr)
			} else {
				pcm, sr, err = audio.DecodeWAVToFloat32(raw)
			}
			if err != nil {
				log.Warn().Err(err).Msg("audio decode failed")
				_ = conn.WriteJSON(map[string]any{"type": "error", "detail": "decode audio failed"})
				continue
			}

			// Resample to 16kHz if needed
			if len(pcm) > 0 && sr != 16000 && sr > 0 {
				before := len(pcm)
				pcm = audio.ResampleLinear(pcm, sr, 16000)
				log.Debug().Int("before", before).Int("after", len(pcm)).Int("sr", sr).Msg("resampled audio")
			}

			// Append to session buffer
			if len(pcm) > 0 {
				samplesMu.Lock()
				samples = append(samples, pcm...)

				// Keep reasonable buffer size - use rolling 90s window for 1 minute context + 30s overflow
				// This allows whisper to correct itself and finalize sentences properly
				const maxBufferSamples = 90 * 16000 // 90 seconds at 16kHz
				if len(samples) > maxBufferSamples {
					// Discard oldest 30 seconds when buffer is full
					discardSamples := 30 * 16000
					samples = samples[discardSamples:]
					// Adjust processed offset
					if processedSamples > discardSamples {
						processedSamples -= discardSamples
					} else {
						processedSamples = 0
					}
					log.Debug().Int("buffer_samples", len(samples)).Float64("buffer_seconds", float64(len(samples))/16000.0).Msg("trimmed buffer (rolling 90s window)")
				}
				samplesMu.Unlock()

				log.Debug().Int("chunk_samples", len(pcm)).Int("total_samples", len(samples)).Float64("duration_sec", float64(len(samples))/16000.0).Msg("audio chunk received")
			}

			// Lazy engine init
			if engine == nil {
				log.Info().Msg("initializing whisper engine...")
				if e, err := weng.NewEngine(s.cfg.ModelPath); err == nil {
					engine = e
					// Configure language if already set via start event
					if sourceLanguage != "" {
						engine.SetLanguage(sourceLanguage)
					}
					log.Info().Str("language", sourceLanguage).Msg("whisper engine initialized successfully")
				} else {
					log.Error().Err(err).Msg("engine init failed")
					_ = conn.WriteJSON(map[string]any{"type": "error", "detail": "engine init failed"})
					continue
				}
			}

			fseq, _ := msg["sequence"].(float64)
			seq = int(fseq)

			// Start background worker on first chunk
			if engine != nil && !workerStarted {
				startTranscriptionWorker()
			}
		case "stop":
			// Acknowledge
			_ = conn.WriteJSON(map[string]any{"type": "stopped"})
			s.leaveRoom(roomID, conn)
			close(exitWorker) // Signal worker to exit
			if engine != nil {
				_ = engine.Close()
			}
			return
		default:
			_ = conn.WriteJSON(map[string]any{"type": "error", "detail": "unknown message type"})
		}
	}
}

func (s *Server) joinRoom(room string, c *websocket.Conn, meta *clientMeta) {
	if room == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.rooms[room]
	if m == nil {
		m = make(map[*websocket.Conn]*clientMeta)
		s.rooms[room] = m
	}
	m[c] = &clientMeta{peerID: meta.peerID, peerLabel: meta.peerLabel, channelID: meta.channelID}
	s.broadcastRoster(room)
}

func (s *Server) leaveRoom(room string, c *websocket.Conn) {
	if room == "" || c == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if m := s.rooms[room]; m != nil {
		delete(m, c)
		if len(m) == 0 {
			delete(s.rooms, room)
		}
	}
	s.broadcastRoster(room)
}

func (s *Server) broadcast(room string, sender *websocket.Conn, senderPeerID string, payload map[string]any) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	m := s.rooms[room]
	for c, info := range m {
		if c == sender {
			continue
		}
		if info != nil && senderPeerID != "" && info.peerID == senderPeerID {
			continue
		}
		_ = c.WriteJSON(payload)
	}
}

func (s *Server) broadcastRoster(room string) {
	s.mu.RLock()
	m := s.rooms[room]
	members := make([]map[string]any, 0, len(m))
	for _, info := range m {
		if info == nil {
			continue
		}
		members = append(members, map[string]any{
			"peer_id":    info.peerID,
			"peer_label": info.peerLabel,
			"channel_id": info.channelID,
		})
	}
	s.mu.RUnlock()
	payload := map[string]any{"type": "room_roster", "members": members}
	s.mu.RLock()
	for c := range m {
		_ = c.WriteJSON(payload)
	}
	s.mu.RUnlock()
}

func asFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case string:
		f, _ := strconv.ParseFloat(strings.TrimSpace(x), 64)
		return f
	default:
		return 0
	}
}
