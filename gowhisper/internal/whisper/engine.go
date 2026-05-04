package whisper

import (
	"fmt"
	"sync/atomic"

	"github.com/rs/zerolog/log"
)

// Engine is a small interface for whisper transcription.
// Implementations may be a no-op (stub) or backed by whisper.cpp (build tag: whisper_cpp).
type Engine interface {
	// Process runs transcription over the provided PCM32F samples.
	// Returns (deltaText, fullText, language).
	Process(samples []float32) (string, string, string, error)
	// ProcessWithLanguage runs transcription with a per-call language.
	// Use "auto" or "" for auto-detection.
	ProcessWithLanguage(samples []float32, lang string) (string, string, string, error)
	// Stream runs transcription and calls back for each new segment as it becomes available.
	// The callback should be fast and non-blocking to avoid stalling decoding.
	Stream(samples []float32, onSegment func(text string, lang string)) error
	// SetLanguage configures the language for transcription. Use "auto" for auto-detection.
	SetLanguage(lang string)
	// GetStreamingConfig returns the work window and context size in samples.
	// Returns (workWindowSamples, contextSamples).
	GetStreamingConfig() (int, int)
	Close() error
}

type enginePool struct {
	engines []Engine
	next    atomic.Uint64
}

func NewEnginePool(modelPath string, size int) (Engine, error) {
	if size <= 1 {
		log.Info().Int("pool_size", 1).Msg("whisper: creating engine pool")
		return NewEngine(modelPath)
	}
	log.Info().Int("pool_size", size).Msg("whisper: creating engine pool")
	engines := make([]Engine, 0, size)
	for i := 0; i < size; i++ {
		engine, err := NewEngine(modelPath)
		if err != nil {
			for _, created := range engines {
				_ = created.Close()
			}
			return nil, fmt.Errorf("create engine %d/%d: %w", i+1, size, err)
		}
		engines = append(engines, engine)
	}
	return &enginePool{engines: engines}, nil
}

func (p *enginePool) pick() Engine {
	if len(p.engines) == 1 {
		return p.engines[0]
	}
	idx := p.next.Add(1) - 1
	return p.engines[int(idx%uint64(len(p.engines)))]
}

func (p *enginePool) Process(samples []float32) (string, string, string, error) {
	return p.pick().Process(samples)
}

func (p *enginePool) ProcessWithLanguage(samples []float32, lang string) (string, string, string, error) {
	return p.pick().ProcessWithLanguage(samples, lang)
}

func (p *enginePool) Stream(samples []float32, onSegment func(text string, lang string)) error {
	return p.pick().Stream(samples, onSegment)
}

func (p *enginePool) SetLanguage(lang string) {
	for _, engine := range p.engines {
		engine.SetLanguage(lang)
	}
}

func (p *enginePool) GetStreamingConfig() (int, int) {
	return p.engines[0].GetStreamingConfig()
}

func (p *enginePool) Close() error {
	var firstErr error
	for _, engine := range p.engines {
		if err := engine.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
