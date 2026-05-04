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
	cfg            config.Config
	upgrader       websocket.Upgrader
	mu             sync.RWMutex
	rooms          map[string]map[*websocket.Conn]*clientMeta
	translator     *translation.Client
	engineProvider func() (weng.Engine, error)
	engineMu       sync.Mutex
	engine         weng.Engine
}

type clientMeta struct {
	peerID        string
	peerLabel     string
	peerAvatarURL string
	channelID     string
	writeMu       *sync.Mutex
}

func NewServer(cfg config.Config) *Server {
	return NewServerWithEngine(cfg, nil)
}

func NewServerWithEngine(cfg config.Config, engineProvider func() (weng.Engine, error)) *Server {
	return &Server{
		cfg: cfg,
		upgrader: websocket.Upgrader{
			CheckOrigin:     func(r *http.Request) bool { return true },
			ReadBufferSize:  1024 * 16,
			WriteBufferSize: 1024 * 16,
		},
		rooms:          make(map[string]map[*websocket.Conn]*clientMeta),
		translator:     translation.New(cfg.TranslationBaseURL, cfg.TranslationTimeoutSec),
		engineProvider: engineProvider,
	}
}

func (s *Server) ensureEngine() (weng.Engine, error) {
	if s.engineProvider != nil {
		return s.engineProvider()
	}
	s.engineMu.Lock()
	defer s.engineMu.Unlock()
	if s.engine != nil {
		return s.engine, nil
	}
	created, err := weng.NewEnginePool(s.cfg.ModelPath, s.cfg.EnginePoolSize)
	if err != nil {
		return nil, err
	}
	s.engine = created
	return s.engine, nil
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
	const finalizeQuietPeriod = 1500 * time.Millisecond
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
		finalizeAfter     time.Time
		lastSpeechAt      time.Time
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

			type inferenceRequest struct {
				id                int64
				totalSamples      int
				bufferSamples     int
				caughtUp          bool
				audioWithContext  []float32
				inferenceSamples  []float32
				windowRMS         float64
				windowPeak        float64
				finalizeRequested bool
			}

			type inferenceResponse struct {
				req          inferenceRequest
				fullText     string
				detectedLang string
				err          error
				elapsed      time.Duration
			}

			// Live utterance state: keep a revisable live suffix and a committed
			// prefix, then only finalize when the transcript stabilizes or the
			// speaker explicitly stops.
			var lastStableText string
			var stableTextCount int
			var repeatedTranscriptCount int
			var lastRepeatedTranscript string
			var nextRequestID int64
			var inFlight bool
			var inFlightReq *inferenceRequest
			var pendingReq *inferenceRequest
			const stableThreshold = 2          // Number of passes before marking as final
			const autoFinalRepeatThreshold = 2 // Repeated identical transcript triggers a final
			const workTick = 100 * time.Millisecond
			const minStepSamples = 1600              // 100ms at 16kHz for steady partial updates
			const minLiveInferenceSamples = 8000     // 500ms before first live hypothesis
			const minFinalizeInferenceSamples = 9600 // 600ms before trusting finalize retry
			liveInferenceWindowSamples := max(8000, s.cfg.LiveWindowMs*16)
			maxWindowSamples := max(liveInferenceWindowSamples, s.cfg.MaxWindowMs*16)
			const keepRecentSamples = 16000 * 2     // 2 seconds context after a final
			const contextOverlapSamples = 16000 / 2 // 0.5 second overlap between catch-up windows

			reqCh := make(chan inferenceRequest, 1)
			respCh := make(chan inferenceResponse, 1)

			go func() {
				for {
					select {
					case <-exitWorker:
						return
					case req := <-reqCh:
						started := time.Now()
						done := make(chan struct{})
						go func(req inferenceRequest) {
							timer := time.NewTimer(2 * time.Second)
							defer timer.Stop()
							select {
							case <-done:
							case <-exitWorker:
							case <-timer.C:
								log.Warn().
									Int64("request_id", req.id).
									Int("samples", len(req.inferenceSamples)).
									Int("processed_until", req.totalSamples).
									Int("buffer_samples", req.bufferSamples).
									Bool("caught_up", req.caughtUp).
									Bool("finalize", req.finalizeRequested).
									Msg("worker: whisper inference still running after 2s")
							}
						}(req)
						_, fullText, detectedLang, err := engine.ProcessWithLanguage(req.inferenceSamples, sourceLanguage)
						close(done)
						resp := inferenceResponse{
							req:          req,
							fullText:     strings.TrimSpace(fullText),
							detectedLang: detectedLang,
							err:          err,
							elapsed:      time.Since(started),
						}
						select {
						case respCh <- resp:
						case <-exitWorker:
							return
						}
					}
				}
			}()

			dispatchRequest := func(req inferenceRequest) {
				inFlight = true
				reqCopy := req
				inFlightReq = &reqCopy
				select {
				case reqCh <- req:
				case <-exitWorker:
				}
			}

			for {
				if !finalizeRequested && !finalizeAfter.IsZero() && !time.Now().Before(finalizeAfter) {
					finalizeRequested = true
					finalizeAfter = time.Time{}
					log.Debug().
						Str("room_id", roomID).
						Str("peer_id", meta.peerID).
						Msg("worker: applying deferred finalize after quiet period")
				}

				select {
				case <-exitWorker:
					return
				case resp := <-respCh:
					inFlight = false
					inFlightReq = nil
					if resp.err != nil {
						log.Warn().
							Err(resp.err).
							Dur("elapsed", resp.elapsed).
							Int64("request_id", resp.req.id).
							Msg("worker: process failed")
						if pendingReq != nil {
							req := *pendingReq
							pendingReq = nil
							dispatchRequest(req)
						}
						continue
					}

					log.Debug().
						Dur("elapsed", resp.elapsed).
						Int("samples", len(resp.req.inferenceSamples)).
						Int("processed_until", resp.req.totalSamples).
						Int("buffer_samples", resp.req.bufferSamples).
						Bool("caught_up", resp.req.caughtUp).
						Bool("finalize", resp.req.finalizeRequested).
						Int64("request_id", resp.req.id).
						Msg("worker: inference complete")

					// Build result
					result := transcriptResult{
						lang:     resp.detectedLang,
						isFinal:  false,
						sequence: seq,
						trans:    map[string]any{},
					}
					if result.lang == "" {
						result.lang = "en"
					}

					fullText := resp.fullText
					inferenceDuration := float64(len(resp.req.inferenceSamples)) / 16000.0
					speechLike := isSpeechLikeWindow(inferenceDuration, resp.req.windowRMS, resp.req.windowPeak)
					if isBlankAudioText(fullText) {
						if speechLike {
							log.Warn().
								Float64("duration_sec", inferenceDuration).
								Float64("rms", resp.req.windowRMS).
								Float64("peak", resp.req.windowPeak).
								Str("lang", result.lang).
								Msg("worker: suppressing [BLANK_AUDIO] for speech-like audio window")
							if resp.req.finalizeRequested {
								candidate := strings.TrimSpace(joinTranscript(confirmedPrefix, lastPartialText))
								if candidate != "" && !isBlankAudioText(candidate) {
									fullTranscript = candidate
									fullText = lastPartialText
								} else {
									finalizeRequested = false
									dumpCurrentAudio("blank_suppressed")
									lastSampleCount = resp.req.totalSamples
									if pendingReq != nil {
										req := *pendingReq
										pendingReq = nil
										dispatchRequest(req)
									}
									continue
								}
							} else {
								lastSampleCount = resp.req.totalSamples
								if pendingReq != nil {
									req := *pendingReq
									pendingReq = nil
									dispatchRequest(req)
								}
								continue
							}
						}
					}

					if fullText != "" {
						prevFull := fullTranscript
						prevFinalized := finalizedText
						prevLive := lastPartialText

						liveText := stripCommittedPrefix(fullText, confirmedPrefix)
						fullText = strings.TrimSpace(liveText)
						if fullText == "" && !resp.req.finalizeRequested {
							lastSampleCount = resp.req.totalSamples
							if pendingReq != nil {
								req := *pendingReq
								pendingReq = nil
								dispatchRequest(req)
							}
							continue
						}

						liveMerged := mergeRollingTranscript(prevLive, fullText)
						fullTranscript = joinTranscript(confirmedPrefix, liveMerged)

						result.delta = computeTextDelta(prevFull, fullTranscript)
						result.full = fullTranscript

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

						if lastSentenceEndPos >= 0 {
							completedSentence := fullTranscript[:lastSentenceEndPos+1]
							if prevFinalized == "" || !strings.HasPrefix(completedSentence, prevFinalized) || len(completedSentence) > len(prevFinalized) {
								if completedSentence == lastStableText {
									stableTextCount++
									log.Debug().
										Int("stable_count", stableTextCount).
										Str("stable_sentence", completedSentence[max(0, len(completedSentence)-50):]).
										Msg("worker: completed sentence stability detected")
								} else {
									stableTextCount = 1
									lastStableText = completedSentence
								}

								if stableTextCount >= stableThreshold {
									result.isFinal = true
									finalizedText = completedSentence
									confirmedPrefix = completedSentence
									result.delta = completedSentence
									result.full = completedSentence
									log.Info().
										Int("stable_count", stableTextCount).
										Str("final_sentence", completedSentence).
										Int("sentence_length", len(completedSentence)).
										Msg("worker: emitting automatic isFinal (sentence stabilized)")
									stableTextCount = 0
									lastStableText = ""
								}
							}
						} else if lastStableText != "" {
							log.Debug().Msg("worker: no sentence ending found, resetting stability")
							lastStableText = ""
							stableTextCount = 0
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

						if resp.req.finalizeRequested {
							if (fullTranscript == "" || isBlankAudioText(fullTranscript)) && len(resp.req.audioWithContext) >= minFinalizeInferenceSamples {
								_, retryFullText, retryLang, retryErr := engine.ProcessWithLanguage(resp.req.audioWithContext, sourceLanguage)
								if retryErr != nil {
									log.Warn().Err(retryErr).Msg("worker: finalize retry process failed")
								} else {
									retryFullText = strings.TrimSpace(retryFullText)
									if retryFullText != "" && !isBlankAudioText(retryFullText) {
										retryLive := stripCommittedPrefix(retryFullText, confirmedPrefix)
										fullTranscript = joinTranscript(confirmedPrefix, mergeRollingTranscript(lastPartialText, retryLive))
										result.lang = retryLang
										if result.lang == "" {
											result.lang = "en"
										}
										log.Info().
											Str("text", retryFullText).
											Str("lang", result.lang).
											Msg("worker: finalize retry recovered non-blank transcript")
									}
								}
							}

							result.isFinal = true
							result.delta = fullTranscript
							result.full = fullTranscript
							finalizedText = fullTranscript
							finalizeRequested = false
							dumpCurrentAudio("finalize_request")
							log.Info().Str("text", fullTranscript).Msg("worker: promoting transcript to final after finalize request")
						}

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
							nextLastSampleCount := 0
							samplesMu.Lock()
							if resp.req.totalSamples < len(samples) {
								// Keep any audio that arrived while this inference was running.
								// Without this, a final result from an older window can trim away
								// backlog and permanently drop part of the conversation.
								contextStart := max(0, resp.req.totalSamples-keepRecentSamples)
								samples = append([]float32(nil), samples[contextStart:]...)
								nextLastSampleCount = resp.req.totalSamples - contextStart
							} else if len(samples) > keepRecentSamples {
								samples = append([]float32(nil), samples[len(samples)-keepRecentSamples:]...)
							} else {
								samples = append([]float32(nil), samples...)
							}
							samplesMu.Unlock()
							lastPartialText = ""
							fullTranscript = ""
							finalizedText = ""
							confirmedPrefix = ""
							lastSampleCount = nextLastSampleCount
							repeatedTranscriptCount = 0
							lastRepeatedTranscript = ""
						} else {
							lastPartialText = fullText
							lastSampleCount = resp.req.totalSamples
						}
					} else {
						lastSampleCount = resp.req.totalSamples
					}

					log.Debug().
						Int("last_sample_count", lastSampleCount).
						Str("full_text", fullTranscript).
						Bool("is_final", result.isFinal).
						Msg("worker: chunk complete")

					if pendingReq != nil {
						req := *pendingReq
						pendingReq = nil
						dispatchRequest(req)
					}
					continue
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

				hasLiveHypothesis := strings.TrimSpace(fullTranscript) != "" || strings.TrimSpace(lastPartialText) != ""

				targetSamples := totalSamples
				if newSamplesCount > maxWindowSamples {
					targetSamples = lastSampleCount + maxWindowSamples
				} else {
					windowSamples := liveInferenceWindowSamples
					targetSamples = min(totalSamples, lastSampleCount+windowSamples)
				}

				startOffset := max(0, lastSampleCount-contextOverlapSamples)
				if targetSamples-startOffset > maxWindowSamples {
					startOffset = targetSamples - maxWindowSamples
				}

				audioWithContext := make([]float32, targetSamples-startOffset)
				copy(audioWithContext, samples[startOffset:targetSamples])
				samplesMu.Unlock()

				audioDuration := float64(len(audioWithContext)) / 16000.0
				requestedNewSamples := targetSamples - lastSampleCount
				newDuration := float64(requestedNewSamples) / 16000.0
				windowRMS, windowPeak := audio.SignalStats(audioWithContext)
				requestCaughtUp := targetSamples == totalSamples
				quietFor := time.Duration(0)
				if !lastSpeechAt.IsZero() {
					quietFor = time.Since(lastSpeechAt)
				}
				requestFinalize := finalizeRequested && requestCaughtUp && quietFor >= finalizeQuietPeriod

				// Keep feeding audio into Whisper while the speaker is talking so
				// the UI can track the sentence in motion. We are stricter about
				// the very first hypothesis than we are about updates to an
				// existing live hypothesis.
				if !requestFinalize && requestedNewSamples < minStepSamples {
					time.Sleep(workTick)
					continue
				}
				if !requestFinalize && !hasLiveHypothesis && len(audioWithContext) < minLiveInferenceSamples {
					time.Sleep(workTick)
					continue
				}

				inferenceSamples := audioWithContext
				if !requestFinalize && requestCaughtUp && len(inferenceSamples) > liveInferenceWindowSamples {
					inferenceSamples = inferenceSamples[len(inferenceSamples)-liveInferenceWindowSamples:]
				}

				nextRequestID++
				req := inferenceRequest{
					id:                nextRequestID,
					totalSamples:      targetSamples,
					bufferSamples:     totalSamples,
					caughtUp:          requestCaughtUp,
					audioWithContext:  append([]float32(nil), audioWithContext...),
					inferenceSamples:  append([]float32(nil), inferenceSamples...),
					windowRMS:         windowRMS,
					windowPeak:        windowPeak,
					finalizeRequested: requestFinalize,
				}

				if inFlightReq != nil &&
					inFlightReq.totalSamples == req.totalSamples &&
					inFlightReq.finalizeRequested == req.finalizeRequested {
					time.Sleep(workTick)
					continue
				}

				if pendingReq != nil &&
					pendingReq.totalSamples == req.totalSamples &&
					pendingReq.finalizeRequested == req.finalizeRequested {
					time.Sleep(workTick)
					continue
				}

				log.Debug().
					Int("new_samples", newSamplesCount).
					Int("total_with_context", len(audioWithContext)).
					Int("processed_until", targetSamples).
					Int("buffer_samples", totalSamples).
					Float64("context_duration", audioDuration).
					Float64("new_duration", newDuration).
					Dur("quiet_for", quietFor).
					Bool("caught_up", requestCaughtUp).
					Bool("finalize", requestFinalize).
					Msg("worker: queued audio for inference")

				if !inFlight {
					dispatchRequest(req)
				} else {
					pendingReq = &req
				}

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
				return
			}
			log.Warn().Err(err).Msg("ws read error")
			s.leaveRoom(roomID, conn)
			stopWorker()
			<-resultWriterDone
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
				isSilentChunk := rms < 0.0005 && peak < 0.002
				if isSilentChunk {
					if chunksReceived == 1 || chunksReceived%20 == 0 {
						log.Debug().
							Int("chunks_received", chunksReceived).
							Int("chunk_samples", len(pcm)).
							Float64("rms", rms).
							Float64("peak", peak).
							Str("room_id", roomID).
							Str("peer_id", meta.peerID).
							Msg("dropping near-silent chunk")
					}
					continue
				}
				if finalizeRequested {
					finalizeRequested = false
					finalizeAfter = time.Time{}
					log.Debug().
						Int("chunks_received", chunksReceived).
						Int("chunk_samples", len(pcm)).
						Float64("rms", rms).
						Float64("peak", peak).
						Str("room_id", roomID).
						Str("peer_id", meta.peerID).
						Msg("canceled pending finalize because speech resumed")
				}
				lastSpeechAt = time.Now()
				samplesMu.Lock()
				samples = append(samples, pcm...)

				// Keep a rolling buffer, but only trim audio the worker has
				// already processed. If Whisper falls behind, trimming
				// unprocessed samples here would silently drop conversation.
				const maxBufferSamples = 90 * 16000 // 90 seconds at 16kHz
				if len(samples) > maxBufferSamples {
					excessSamples := len(samples) - maxBufferSamples
					processedTrimLimit := max(0, lastSampleCount-(3*16000))
					discardSamples := min(max(excessSamples, 30*16000), processedTrimLimit)
					if discardSamples > 0 {
						samples = samples[discardSamples:]
						lastSampleCount -= discardSamples
					} else {
						log.Warn().
							Int("buffer_samples", len(samples)).
							Int("last_sample_count", lastSampleCount).
							Float64("buffer_seconds", float64(len(samples))/16000.0).
							Msg("audio buffer over target; preserving unprocessed backlog")
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
				if e, err := s.ensureEngine(); err == nil {
					engine = e
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
			return
		case "finalize":
			if !lastSpeechAt.IsZero() && time.Since(lastSpeechAt) < finalizeQuietPeriod {
				finalizeAfter = lastSpeechAt.Add(finalizeQuietPeriod)
				log.Debug().
					Dur("delay", time.Until(finalizeAfter)).
					Str("room_id", roomID).
					Str("peer_id", meta.peerID).
					Msg("deferred finalize until speech has been quiet")
			} else {
				finalizeRequested = true
				finalizeAfter = time.Time{}
			}
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

func isSpeechLikeWindow(durationSec, rms, peak float64) bool {
	return durationSec >= 0.75 && (rms >= 0.004 || peak >= 0.03)
}

func joinTranscript(prefix, tail string) string {
	prefix = strings.TrimSpace(prefix)
	tail = strings.TrimSpace(tail)
	if prefix == "" {
		return tail
	}
	if tail == "" {
		return prefix
	}
	return strings.TrimSpace(prefix + " " + tail)
}

func stripCommittedPrefix(text, prefix string) string {
	text = strings.TrimSpace(text)
	prefix = strings.TrimSpace(prefix)
	if text == "" || prefix == "" {
		return text
	}
	if strings.HasPrefix(text, prefix) {
		return strings.TrimSpace(strings.TrimPrefix(text, prefix))
	}
	matched := 0
	for end := len(prefix); end > 0; end-- {
		candidate := prefix[:end]
		if strings.HasPrefix(text, candidate) {
			matched = end
			break
		}
	}
	if matched > 0 {
		return strings.TrimSpace(text[matched:])
	}
	return text
}

func computeTextDelta(previous, current string) string {
	previous = strings.TrimSpace(previous)
	current = strings.TrimSpace(current)
	if current == "" {
		return ""
	}
	if previous == "" {
		return current
	}
	if strings.HasPrefix(current, previous) {
		return strings.TrimSpace(strings.TrimPrefix(current, previous))
	}

	prevWords := strings.Fields(previous)
	currWords := strings.Fields(current)
	bestWords := 0
	maxWordOverlap := min(len(prevWords), len(currWords))
	for overlap := maxWordOverlap; overlap >= 1; overlap-- {
		match := true
		for i := 0; i < overlap; i++ {
			if !strings.EqualFold(prevWords[len(prevWords)-overlap+i], currWords[i]) {
				match = false
				break
			}
		}
		if match {
			bestWords = overlap
			break
		}
	}
	if bestWords > 0 {
		return strings.TrimSpace(strings.Join(currWords[bestWords:], " "))
	}

	return current
}

func mergeRollingTranscript(previous, current string) string {
	previous = strings.TrimSpace(previous)
	current = strings.TrimSpace(current)
	if previous == "" {
		return current
	}
	if current == "" {
		return previous
	}
	if previous == current {
		return current
	}
	if strings.HasSuffix(previous, current) {
		return previous
	}
	if strings.HasPrefix(current, previous) {
		return current
	}

	maxOverlap := min(len(previous), len(current))
	best := 0
	for overlap := maxOverlap; overlap >= 1; overlap-- {
		if strings.EqualFold(previous[len(previous)-overlap:], current[:overlap]) {
			best = overlap
			break
		}
	}
	if best > 0 {
		return strings.TrimSpace(previous + current[best:])
	}

	prevWords := strings.Fields(previous)
	currWords := strings.Fields(current)
	bestWords := 0
	maxWordOverlap := min(len(prevWords), len(currWords))
	for overlap := maxWordOverlap; overlap >= 1; overlap-- {
		match := true
		for i := 0; i < overlap; i++ {
			if !strings.EqualFold(prevWords[len(prevWords)-overlap+i], currWords[i]) {
				match = false
				break
			}
		}
		if match {
			bestWords = overlap
			break
		}
	}
	if bestWords > 0 {
		merged := append(append([]string{}, prevWords...), currWords[bestWords:]...)
		return strings.TrimSpace(strings.Join(merged, " "))
	}

	if len(current) > len(previous) {
		return current
	}
	return previous
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
