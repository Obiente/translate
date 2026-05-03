package ws

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"

	"github.com/obiente/translate/gowhisper/internal/audio"
	"github.com/obiente/translate/gowhisper/internal/config"
	"github.com/obiente/translate/gowhisper/internal/translation"
	weng "github.com/obiente/translate/gowhisper/internal/whisper"
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
	peerID        string
	peerLabel     string
	peerAvatarURL string
	channelID     string
	writeMu       *sync.Mutex
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
		roomID         string
		meta           = &clientMeta{}
		engine         weng.Engine
		samples        []float32
		seq            int
		chunksReceived int
		resultsSent    int
		writeMu        sync.Mutex

		// Session configuration from start event
		sourceLanguage          string
		targetLanguages         []string
		translationAlternatives int

		// Track transcription state
		fullTranscript    string                // current utterance transcript
		finalizedText     string                // text already finalized for current utterance
		confirmedPrefix   string                // confirmed sentence prefix retained across recalculations
		lastPartialText   string                // most recent live text tail
		lastSampleCount   int                   // sample count at last inference pass
		workerStarted     bool                  // track if background worker is running
		samplesMu         sync.Mutex            // protect samples buffer
		exitWorker        = make(chan struct{}) // signal worker to exit
		stopWorkerOnce    sync.Once
		finalizeRequested bool
		dumpCount         int
	)
	meta.writeMu = &writeMu

	stopWorker := func() {
		stopWorkerOnce.Do(func() {
			close(exitWorker)
		})
	}

	sendJSON := func(payload any) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return conn.WriteJSON(payload)
	}

	dumpCurrentAudio := func(tag string) {
		if s.cfg.DebugDumpDir == "" {
			return
		}
		samplesMu.Lock()
		snapshot := append([]float32(nil), samples...)
		samplesMu.Unlock()
		if len(snapshot) == 0 {
			return
		}
		if err := os.MkdirAll(s.cfg.DebugDumpDir, 0o755); err != nil {
			log.Warn().Err(err).Str("dir", s.cfg.DebugDumpDir).Msg("failed to create debug dump dir")
			return
		}
		wavBytes, err := audio.EncodePCM16MonoWAV(snapshot, 16000)
		if err != nil {
			log.Warn().Err(err).Msg("failed to encode debug wav")
			return
		}
		dumpCount++
		name := fmt.Sprintf(
			"%s_%s_%s_%02d_%d.wav",
			safeFileComponent(roomID),
			safeFileComponent(meta.peerID),
			safeFileComponent(tag),
			dumpCount,
			time.Now().UnixMilli(),
		)
		path := filepath.Join(s.cfg.DebugDumpDir, name)
		if err := os.WriteFile(path, wavBytes, 0o644); err != nil {
			log.Warn().Err(err).Str("path", path).Msg("failed to write debug wav")
			return
		}
		rms, peak := audio.SignalStats(snapshot)
		log.Info().
			Str("path", path).
			Int("samples", len(snapshot)).
			Float64("seconds", float64(len(snapshot))/16000.0).
			Float64("rms", rms).
			Float64("peak", peak).
			Msg("wrote debug audio dump")
	}

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
	resultWriterDone := make(chan struct{})

	writeTranscriptResult := func(r transcriptResult) {
		resultsSent++
		if resultsSent == 1 || resultsSent%10 == 0 {
			log.Info().
				Int("results_sent", resultsSent).
				Bool("is_final", r.isFinal).
				Int("delta_len", len(r.delta)).
				Int("full_len", len(r.full)).
				Str("room_id", roomID).
				Str("peer_id", meta.peerID).
				Msg("ws transcript emitted")
		}

		payload := map[string]any{
			"type":         "transcript",
			"text":         r.delta,
			"fullText":     r.full,
			"language":     r.lang,
			"isFinal":      r.isFinal,
			"sequence":     r.sequence,
			"translations": r.trans,
		}
		if err := sendJSON(payload); err != nil {
			log.Warn().Err(err).Msg("failed to send transcript")
		} else {
			log.Debug().Str("delta", r.delta).Msg("sent transcript to client")
		}

		if roomID != "" {
			rp := map[string]any{
				"type":            "room_transcript",
				"room_id":         roomID,
				"peer_id":         meta.peerID,
				"peer_label":      meta.peerLabel,
				"peer_avatar_url": meta.peerAvatarURL,
				"channel_id":      meta.channelID,
				"text":            r.delta,
				"fullText":        r.full,
				"language":        r.lang,
				"isFinal":         r.isFinal,
				"sequence":        r.sequence,
				"translations":    r.trans,
			}
			s.broadcast(roomID, conn, meta.peerID, rp)
		}
	}

	go func() {
		defer close(resultWriterDone)
		for {
			select {
			case r := <-resCh:
				writeTranscriptResult(r)
			case <-exitWorker:
				for {
					select {
					case r := <-resCh:
						writeTranscriptResult(r)
					default:
						return
					}
				}
			}
		}
	}()

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
			var repeatedTranscriptCount int
			var lastRepeatedTranscript string
			const stableThreshold = 2          // Number of passes before marking as final
			const autoFinalRepeatThreshold = 2 // Repeated identical transcript triggers a final
			const workTick = 75 * time.Millisecond
			const minStepSamples = 800          // 50ms at 16kHz
			const maxWindowSamples = 16000 * 3  // 3 seconds rolling utterance window
			const keepRecentSamples = 16000 * 1 // 1 second context after a final

			for {
				select {
				case <-exitWorker:
					return
				default:
				}

				// Work over a rolling utterance snapshot so we can revise captions
				// as the sentence continues, similar to the Python path.
				samplesMu.Lock()
				totalSamples := len(samples)
				if totalSamples == 0 {
					if finalizeRequested {
						trimmed := strings.TrimSpace(fullTranscript)
						if trimmed != "" && trimmed != finalizedText {
							result := transcriptResult{
								delta:    trimmed,
								full:     trimmed,
								lang:     sourceLanguage,
								isFinal:  true,
								sequence: seq,
								trans:    map[string]any{},
							}
							if result.lang == "" {
								result.lang = "en"
							}

							shouldTranslateFinal := s.cfg.TranslationEnabled && len(targetLanguages) > 0
							if shouldTranslateFinal {
								ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.TranslationTimeoutSec)*time.Second)
								alts := translationAlternatives
								if alts < 0 {
									alts = 0
								}
								if m, err := s.translator.Translate(ctx, result.delta, result.lang, targetLanguages, alts); err != nil {
									log.Warn().Err(err).Str("delta", result.delta).Msg("worker: translation request failed")
								} else if m != nil {
									for k, v := range m {
										result.trans[k] = v
									}
								}
								cancel()
							}

						finalizedText = trimmed
						finalizeRequested = false
						dumpCurrentAudio("finalize_idle")
						log.Info().Str("text", trimmed).Msg("worker: forcing final transcript after speaker inactivity")
						select {
							case resCh <- result:
							case <-time.After(1 * time.Second):
							}
						} else {
							finalizeRequested = false
						}
					}
					samplesMu.Unlock()
					time.Sleep(workTick)
					continue
				}

				newSamplesCount := totalSamples - lastSampleCount
				if newSamplesCount < 0 {
					newSamplesCount = totalSamples
				}

				startOffset := 0
				if totalSamples > maxWindowSamples {
					startOffset = totalSamples - maxWindowSamples
				}

				audioWithContext := make([]float32, totalSamples-startOffset)
				copy(audioWithContext, samples[startOffset:totalSamples])
				samplesMu.Unlock()

				audioDuration := float64(len(audioWithContext)) / 16000.0
				newDuration := float64(newSamplesCount) / 16000.0
				windowRMS, windowPeak := audio.SignalStats(audioWithContext)

				// Only process when enough new audio arrived, unless we're
				// explicitly finalizing because the speaker stopped.
				if !finalizeRequested && newSamplesCount < minStepSamples {
					time.Sleep(workTick)
					continue
				}

				log.Debug().
					Int("new_samples", newSamplesCount).
					Int("total_with_context", len(audioWithContext)).
					Int("total_samples", totalSamples).
					Float64("context_duration", audioDuration).
					Float64("new_duration", newDuration).
					Msg("worker: processing audio with context")

				// Process a short bounded batch with context.
				_, fullText, detectedLang, err := engine.Process(audioWithContext)
				if err != nil {
					log.Warn().Err(err).Msg("worker: process failed")
					time.Sleep(workTick)
					continue
				}

				// Build result
				result := transcriptResult{
					lang:     detectedLang,
					isFinal:  false,
					sequence: seq,
					trans:    map[string]any{},
				}
				if result.lang == "" {
					result.lang = "en"
				}

				fullText = strings.TrimSpace(fullText)
				if isBlankAudioText(fullText) {
					speechLike := audioDuration >= 0.75 && (windowRMS >= 0.004 || windowPeak >= 0.03)
					if speechLike {
						log.Warn().
							Float64("duration_sec", audioDuration).
							Float64("rms", windowRMS).
							Float64("peak", windowPeak).
							Str("lang", result.lang).
							Msg("worker: suppressing [BLANK_AUDIO] for speech-like audio window")
						if finalizeRequested {
							finalizeRequested = false
							dumpCurrentAudio("blank_suppressed")
						}
						lastSampleCount = totalSamples
						time.Sleep(workTick)
						continue
					}
				}
				if fullText != "" {
					prevFull := fullTranscript
					prevFinalized := finalizedText

					// Strip the already-confirmed prefix from the raw text so the
					// live tail can keep being revised without duplicating finals.
					liveText := fullText
					if confirmedPrefix != "" {
						strippedPrefix := strings.TrimSpace(confirmedPrefix)
						if strippedPrefix != "" && strings.HasPrefix(liveText, strippedPrefix) {
							liveText = strings.TrimSpace(strings.TrimPrefix(liveText, strippedPrefix))
						}
					}
					fullText = strings.TrimSpace(liveText)
					if fullText == "" && !finalizeRequested {
						lastSampleCount = totalSamples
						time.Sleep(workTick)
						continue
					}

					if confirmedPrefix != "" && fullText != "" {
						fullTranscript = strings.TrimSpace(confirmedPrefix + " " + fullText)
					} else if confirmedPrefix != "" {
						fullTranscript = strings.TrimSpace(confirmedPrefix)
					} else {
						fullTranscript = fullText
					}

					// Calculate delta: new text since last emission (excluding already finalized)
					if lastPartialText != "" && strings.HasPrefix(fullText, lastPartialText) {
						result.delta = strings.TrimSpace(strings.TrimPrefix(fullText, lastPartialText))
					} else if prevFinalized != "" && strings.HasPrefix(fullTranscript, prevFinalized) {
						// If full text changed but still starts with finalized, delta is everything after finalized
						result.delta = strings.TrimSpace(strings.TrimPrefix(fullTranscript, prevFinalized))
					} else if prevFull != "" && strings.HasPrefix(fullTranscript, prevFull) {
						result.delta = strings.TrimSpace(strings.TrimPrefix(fullTranscript, prevFull))
					} else {
						result.delta = fullTranscript
					}
					result.full = fullTranscript

					// Automatic isFinal detection: find the last complete sentence and check if IT has stabilized
					// This allows emitting final during long speeches when individual sentences complete

					// Find the position of the last sentence-ending punctuation
					sentenceEndings := []rune{'.', '!', '?', '。', '！', '？', '♪', '*', ']', ')'}
					lastSentenceEndPos := -1
					for i := len(fullTranscript) - 1; i >= 0; i-- {
						for _, ch := range sentenceEndings {
							if rune(fullTranscript[i]) == ch {
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
						completedSentence := fullTranscript[:lastSentenceEndPos+1]

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
								finalizedText = completedSentence // Store what we've finalized
								confirmedPrefix = completedSentence

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
					}

					if fullTranscript != "" && fullTranscript == lastRepeatedTranscript {
						repeatedTranscriptCount++
					} else {
						repeatedTranscriptCount = 1
						lastRepeatedTranscript = fullTranscript
					}

					if !result.isFinal && repeatedTranscriptCount >= autoFinalRepeatThreshold {
						result.isFinal = true
						finalizedText = fullTranscript
						confirmedPrefix = ""
						repeatedTranscriptCount = 0
						lastRepeatedTranscript = ""
						log.Info().Str("text", fullTranscript).Msg("worker: auto-final triggered by repeated transcript")
					}

					// Partial translations are optional. To keep the room feeling live,
					// default to translating only finals unless explicitly enabled.
					shouldTranslate := s.cfg.TranslationEnabled &&
						len(targetLanguages) > 0 &&
						result.delta != "" &&
						(result.isFinal || s.cfg.TranslatePartials)
					if shouldTranslate {
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

					if finalizeRequested {
						result.isFinal = true
						result.delta = fullTranscript
						result.full = fullTranscript
						finalizedText = fullTranscript
						finalizeRequested = false
						dumpCurrentAudio("finalize_request")
						log.Info().Str("text", fullTranscript).Msg("worker: promoting transcript to final after finalize request")
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

					if result.isFinal {
						samplesMu.Lock()
						if len(samples) > keepRecentSamples {
							samples = append([]float32(nil), samples[len(samples)-keepRecentSamples:]...)
						} else {
							samples = append([]float32(nil), samples...)
						}
						samplesMu.Unlock()
						lastPartialText = ""
						fullTranscript = ""
						finalizedText = ""
						confirmedPrefix = ""
						lastSampleCount = 0
						repeatedTranscriptCount = 0
						lastRepeatedTranscript = ""
					} else {
						lastPartialText = fullText
						lastSampleCount = totalSamples
					}
				} else {
					lastSampleCount = totalSamples
				}

				log.Debug().
					Int("last_sample_count", lastSampleCount).
					Str("full_text", fullTranscript).
					Bool("is_final", result.isFinal).
					Msg("worker: chunk complete")

				time.Sleep(workTick)
			}
		}()
	}

	// Simple loop handling JSON control messages and chunk frames.
	for {
		mt, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				s.leaveRoom(roomID, conn)
				stopWorker()
				<-resultWriterDone
				if engine != nil {
					_ = engine.Close()
				}
				return
			}
			log.Warn().Err(err).Msg("ws read error")
			s.leaveRoom(roomID, conn)
			stopWorker()
			<-resultWriterDone
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
			_ = sendJSON(map[string]any{"type": "error", "detail": "invalid json"})
			continue
		}
		switch msg["type"] {
		case "ping":
			// keepalive from client: reset deadline and respond
			_ = conn.SetReadDeadline(time.Now().Add(60 * time.Second))
			_ = sendJSON(map[string]any{"type": "pong", "ts": msg["ts"]})
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

			_ = sendJSON(map[string]any{"type": "started"})
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
			if v, ok := msg["peer_avatar_url"].(string); ok {
				meta.peerAvatarURL = v
			}
			s.joinRoom(rid, conn, meta)
			roomID = rid
			log.Info().
				Str("room_id", roomID).
				Str("peer_id", meta.peerID).
				Str("peer_label", meta.peerLabel).
				Str("peer_avatar_url", meta.peerAvatarURL).
				Str("channel_id", meta.channelID).
				Msg("ws room joined")
			_ = sendJSON(map[string]any{
				"type":            "room_joined",
				"room_id":         roomID,
				"peer_id":         meta.peerID,
				"peer_label":      meta.peerLabel,
				"peer_avatar_url": meta.peerAvatarURL,
			})
		case "leave_room":
			s.leaveRoom(roomID, conn)
			roomID = ""
			_ = sendJSON(map[string]any{"type": "room_left"})
		case "chunk":
			// Decode base64
			b64, _ := msg["data"].(string)
			if b64 == "" {
				continue
			}

			// Inflate audio (WAV or PCM16)
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				_ = sendJSON(map[string]any{"type": "error", "detail": "invalid base64 audio"})
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
				_ = sendJSON(map[string]any{"type": "error", "detail": "decode audio failed"})
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
				chunksReceived++
				rms, peak := audio.SignalStats(pcm)
				samplesMu.Lock()
				samples = append(samples, pcm...)

				// Keep reasonable buffer size - use rolling 90s window for 1 minute context + 30s overflow
				// This allows whisper to correct itself and finalize sentences properly
				const maxBufferSamples = 90 * 16000 // 90 seconds at 16kHz
				if len(samples) > maxBufferSamples {
					// Discard oldest 30 seconds when buffer is full
					discardSamples := 30 * 16000
					samples = samples[discardSamples:]
					if lastSampleCount > discardSamples {
						lastSampleCount -= discardSamples
					} else {
						lastSampleCount = 0
					}
					log.Debug().Int("buffer_samples", len(samples)).Float64("buffer_seconds", float64(len(samples))/16000.0).Msg("trimmed buffer (rolling 90s window)")
				}
				samplesMu.Unlock()

				if chunksReceived == 1 || chunksReceived%20 == 0 {
					log.Info().
						Int("chunks_received", chunksReceived).
						Int("chunk_samples", len(pcm)).
						Int("total_samples", len(samples)).
						Float64("duration_sec", float64(len(samples))/16000.0).
						Float64("rms", rms).
						Float64("peak", peak).
						Str("room_id", roomID).
						Str("peer_id", meta.peerID).
						Str("channel_id", meta.channelID).
						Msg("ws audio chunk received")
				}
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
					_ = sendJSON(map[string]any{"type": "error", "detail": "engine init failed"})
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
			_ = sendJSON(map[string]any{"type": "stopped"})
			s.leaveRoom(roomID, conn)
			stopWorker()
			<-resultWriterDone
			if engine != nil {
				_ = engine.Close()
			}
			return
		case "finalize":
			finalizeRequested = true
			_ = sendJSON(map[string]any{"type": "finalizing"})
		default:
			_ = sendJSON(map[string]any{"type": "error", "detail": "unknown message type"})
		}
	}
}

func (s *Server) joinRoom(room string, c *websocket.Conn, meta *clientMeta) {
	if room == "" {
		return
	}
	s.mu.Lock()
	m := s.rooms[room]
	if m == nil {
		m = make(map[*websocket.Conn]*clientMeta)
		s.rooms[room] = m
	}
	writeMu := meta.writeMu
	if writeMu == nil {
		writeMu = &sync.Mutex{}
	}
	m[c] = &clientMeta{
		peerID:        meta.peerID,
		peerLabel:     meta.peerLabel,
		peerAvatarURL: meta.peerAvatarURL,
		channelID:     meta.channelID,
		writeMu:       writeMu,
	}
	s.mu.Unlock()
	s.broadcastRoster(room)
}

func (s *Server) leaveRoom(room string, c *websocket.Conn) {
	if room == "" || c == nil {
		return
	}
	s.mu.Lock()
	if m := s.rooms[room]; m != nil {
		delete(m, c)
		if len(m) == 0 {
			delete(s.rooms, room)
		}
	}
	s.mu.Unlock()
	s.broadcastRoster(room)
}

func (s *Server) broadcast(room string, sender *websocket.Conn, senderPeerID string, payload map[string]any) {
	s.mu.RLock()
	m := s.rooms[room]
	recipients := make([]struct {
		conn *websocket.Conn
		meta *clientMeta
	}, 0, len(m))
	for c, info := range m {
		if c == sender {
			continue
		}
		if info != nil && senderPeerID != "" && info.peerID == senderPeerID {
			continue
		}
		recipients = append(recipients, struct {
			conn *websocket.Conn
			meta *clientMeta
		}{conn: c, meta: info})
	}
	s.mu.RUnlock()

	for _, recipient := range recipients {
		writeJSON(recipient.conn, recipient.meta, payload)
	}
}

func (s *Server) broadcastRoster(room string) {
	s.mu.RLock()
	m := s.rooms[room]
	members := make([]map[string]any, 0, len(m))
	recipients := make([]struct {
		conn *websocket.Conn
		meta *clientMeta
	}, 0, len(m))
	for _, info := range m {
		if info == nil {
			continue
		}
		members = append(members, map[string]any{
			"peer_id":         info.peerID,
			"peer_label":      info.peerLabel,
			"peer_avatar_url": info.peerAvatarURL,
			"channel_id":      info.channelID,
		})
	}
	for c, info := range m {
		recipients = append(recipients, struct {
			conn *websocket.Conn
			meta *clientMeta
		}{conn: c, meta: info})
	}
	s.mu.RUnlock()

	payload := map[string]any{"type": "room_roster", "members": members}
	for _, recipient := range recipients {
		writeJSON(recipient.conn, recipient.meta, payload)
	}
}

func safeFileComponent(input string) string {
	input = strings.TrimSpace(input)
	if input == "" {
		return "unknown"
	}
	replacer := strings.NewReplacer(
		"/", "_",
		"\\", "_",
		":", "_",
		" ", "_",
		".", "_",
	)
	return replacer.Replace(input)
}

func isBlankAudioText(text string) bool {
	normalized := strings.ToUpper(strings.TrimSpace(text))
	return normalized == "[BLANK_AUDIO]" || normalized == "BLANK_AUDIO"
}

func writeJSON(c *websocket.Conn, meta *clientMeta, payload any) {
	if c == nil {
		return
	}
	if meta != nil && meta.writeMu != nil {
		meta.writeMu.Lock()
		defer meta.writeMu.Unlock()
	}
	_ = c.WriteJSON(payload)
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
