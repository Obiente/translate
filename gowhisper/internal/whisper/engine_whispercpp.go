//go:build whisper_cpp

package whisper

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	whisperpkg "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/rs/zerolog/log"
)

// EngineCPP is the whisper.cpp-backed implementation of Engine. Exported for type assertion in ws/server.go.
type EngineCPP struct {
	model             whisperpkg.Model
	threads           uint
	WorkWindowSamples int        // streaming window size in samples (exported)
	ContextSamples    int        // context size in samples (exported)
	language          string     // configured language ("auto" for auto-detection)
	mu                sync.Mutex // Protect concurrent access to the model
}

func NewEngine(modelPath string) (Engine, error) {
	// Determine thread count
	// Default to a conservative thread count to avoid oversubscribing
	// native threading libraries and triggering instability in C code.
	defaultThreads := runtime.NumCPU()
	if defaultThreads < 1 {
		defaultThreads = 1
	}
	threads := uint(defaultThreads)
	if v := os.Getenv("WHISPER_THREADS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			threads = uint(n)
			log.Info().Int("threads", n).Msg("whisper: using configured thread count")
		}
	} else {
		log.Info().Uint("threads", threads).Msg("whisper: using default thread count")
	}
	if v := os.Getenv("WHISPER_MAX_THREADS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && threads > uint(n) {
			log.Warn().Uint("requested_threads", threads).Int("clamped_to", n).Msg("whisper: clamping thread count to WHISPER_MAX_THREADS")
			threads = uint(n)
		}
	}

	// Streaming config: work window and context size
	// Using 1 minute context like WhisperLive for better accuracy and sentence completion
	workWindowSamples := 8000 // default: 0.5s at 16kHz (minimum processing chunk)
	contextSamples := 960000  // default: 60s at 16kHz (1 minute context for accuracy)
	if v := os.Getenv("WHISPER_WORK_WINDOW_SAMPLES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			workWindowSamples = n
			log.Info().Int("workWindowSamples", n).Float64("seconds", float64(n)/16000.0).Msg("whisper: using configured work window")
		}
	}
	if v := os.Getenv("WHISPER_CONTEXT_SAMPLES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			contextSamples = n
			log.Info().Int("contextSamples", n).Float64("seconds", float64(n)/16000.0).Msg("whisper: using configured context size")
		}
	}

	log.Info().
		Int("workWindowSamples", workWindowSamples).
		Float64("workWindowSeconds", float64(workWindowSamples)/16000.0).
		Int("contextSamples", contextSamples).
		Float64("contextSeconds", float64(contextSamples)/16000.0).
		Msg("whisper: streaming configuration")

	m, err := whisperpkg.New(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	log.Info().Str("model", modelPath).Msg("whisper: model loaded successfully")
	return &EngineCPP{
		model:             m,
		threads:           threads,
		WorkWindowSamples: workWindowSamples,
		ContextSamples:    contextSamples,
		language:          "auto", // default to auto-detection
	}, nil
}

func (e *EngineCPP) Close() error {
	if e.model != nil {
		e.model.Close()
	}
	return nil
}

// SetLanguage configures the language for transcription. Use "auto" for auto-detection.
func (e *EngineCPP) SetLanguage(lang string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.language = normalizeLanguage(lang)
	log.Info().Str("language", e.language).Msg("whisper: language configured")
}

// GetStreamingConfig returns the work window and context size in samples.
func (e *EngineCPP) GetStreamingConfig() (int, int) {
	return e.WorkWindowSamples, e.ContextSamples
}

// Stream processes audio and invokes the callback for each new segment as soon as it's available.
// It configures the context for greedy decoding and single-segment mode to stream partials faster.
func (e *EngineCPP) Stream(samples []float32, onSegment func(text string, lang string)) (err error) {
	e.mu.Lock()
	language := e.language
	e.mu.Unlock()
	return e.streamWithLanguage(samples, language, onSegment)
}

func (e *EngineCPP) streamWithLanguage(samples []float32, language string, onSegment func(text string, lang string)) (err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Error().Interface("panic", r).Stack().Msg("whisper: panic in stream, recovering")
			err = fmt.Errorf("stream panic: %v", r)
		}
	}()
	if len(samples) == 0 {
		return nil
	}
	language = normalizeLanguage(language)

	// Serialize access to the model to prevent concurrent processing crashes
	e.mu.Lock()
	defer e.mu.Unlock()

	// Use configured context and work window sizes (exported fields)
	contextSamples := e.ContextSamples
	workWindowSamples := e.WorkWindowSamples

	// Only process if we have enough samples for a work window
	if len(samples) < workWindowSamples {
		return nil
	}
	// Truncate to context size (sliding window)
	if len(samples) > contextSamples {
		samples = samples[len(samples)-contextSamples:]
	}

	var ctx whisperpkg.Context
	ctx, err = e.model.NewContext()
	if err != nil {
		return fmt.Errorf("create context: %w", err)
	}
	ctx.SetThreads(e.threads)
	_ = ctx.SetLanguage(language)
	ctx.SetSplitOnWord(false)
	ctx.SetTokenTimestamps(false)
	ctx.SetMaxSegmentLength(0)
	ctx.SetMaxTokensPerSegment(0)
	ctx.SetAudioCtx(0)

	// Protect against passing extremely long audio into whisper.cpp which
	// can cause native crashes in some versions of the bindings or when
	// memory is constrained. Limit to a safe maximum (10 seconds by
	// default) and log a warning if we truncated.
	const maxStreamSeconds = 10
	const sampleRate = 16000
	const maxStreamSamples = maxStreamSeconds * sampleRate
	if len(samples) > maxStreamSamples {
		log.Warn().Int("samples", len(samples)).Int("max", maxStreamSamples).Msg("whisper: truncating too-long streaming audio before processing to avoid native crash")
		samples = samples[len(samples)-maxStreamSamples:]
	}

	log.Info().Int("samples", len(samples)).Msg("whisper: starting stream process")

	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return fmt.Errorf("process audio: %w", err)
	}

	lang := ""
	if language != "" && language != "auto" {
		lang = language
	} else {
		lang = ctx.DetectedLanguage()
		if lang == "" {
			lang = "auto"
		}
	}

	emitted := 0
	for {
		seg, nextErr := ctx.NextSegment()
		if nextErr != nil {
			if nextErr == io.EOF {
				break
			}
			return fmt.Errorf("read segment: %w", nextErr)
		}
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		emitted++
		if onSegment != nil {
			onSegment(text, lang)
		}
	}

	log.Info().
		Int("samples", len(samples)).
		Int("segments", emitted).
		Str("language", lang).
		Msg("whisper: stream process complete")
	return nil
}

// Process implements Engine by running a full-context transcription.
// This method is thread-safe but processes serially to avoid whisper.cpp crashes.
func (e *EngineCPP) Process(samples []float32) (delta, full, lang string, err error) {
	e.mu.Lock()
	language := e.language
	e.mu.Unlock()
	return e.ProcessWithLanguage(samples, language)
}

func (e *EngineCPP) ProcessWithLanguage(samples []float32, language string) (delta, full, lang string, err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Error().Interface("panic", r).Stack().Msg("whisper: panic in process, recovering")
			err = fmt.Errorf("process panic: %v", r)
		}
	}()
	if len(samples) == 0 {
		return "", "", "", nil
	}
	language = normalizeLanguage(language)

	// Serialize access to the model to prevent concurrent processing crashes
	e.mu.Lock()
	defer e.mu.Unlock()

	// Validate sample rate (whisper expects 16kHz)
	// If samples are too short, return empty (< 100ms)
	if len(samples) < 1600 {
		log.Debug().Int("samples", len(samples)).Msg("whisper: skipping too-short audio")
		return "", "", "", nil
	}

	// Limit max audio length to prevent crashes (30 seconds at 16kHz)
	const maxSamples = 30 * 16000
	if len(samples) > maxSamples {
		log.Warn().Int("samples", len(samples)).Int("max", maxSamples).Msg("whisper: truncating long audio")
		samples = samples[len(samples)-maxSamples:]
	}

	// Create a new context for each processing call
	var ctx whisperpkg.Context
	ctx, err = e.model.NewContext()
	if err != nil {
		return "", "", "", fmt.Errorf("create context: %w", err)
	}

	// Configure context for optimal streaming performance
	ctx.SetThreads(e.threads)
	ctx.SetLanguage(language)     // Use configured language
	ctx.SetSplitOnWord(false)     // Keep live decoding fast
	ctx.SetTokenTimestamps(false) // Token timestamps are expensive and unused
	ctx.SetMaxSegmentLength(0)    // No artificial limit on segment length
	ctx.SetMaxTokensPerSegment(0) // No token limit
	ctx.SetAudioCtx(0)            // Use full audio context

	// Process the audio samples
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		log.Error().Err(err).Int("samples", len(samples)).Msg("whisper: process failed")
		return "", "", "", fmt.Errorf("process audio: %w", err)
	}

	// Collect all segments
	var segments []string
	for {
		seg, newErr := ctx.NextSegment()
		if newErr != nil {
			if newErr == io.EOF {
				break
			}
			log.Warn().Err(newErr).Msg("whisper: error reading segment")
			break
		}

		// Trim whitespace and add segment
		text := strings.TrimSpace(seg.Text)
		if text != "" {
			segments = append(segments, text)
		}
	}

	// Build full transcription
	full = strings.TrimSpace(strings.Join(segments, " "))

	// Get detected language
	if language != "" && language != "auto" {
		lang = language
	} else if lang == "" {
		lang = ctx.DetectedLanguage()
	}

	log.Info().
		Str("full", full).
		Str("lang", lang).
		Int("segments", len(segments)).
		Int("samples", len(samples)).
		Msg("whisper: transcription complete")

	// For streaming, we don't compute delta here - let the caller handle it
	// based on their previous state
	return "", full, lang, nil
}

func normalizeLanguage(lang string) string {
	lang = strings.TrimSpace(strings.ToLower(lang))
	if lang == "" {
		return "auto"
	}
	return lang
}
