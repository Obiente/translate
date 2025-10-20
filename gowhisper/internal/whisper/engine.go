package whisper

// Engine is a small interface for whisper transcription.
// Implementations may be a no-op (stub) or backed by whisper.cpp (build tag: whisper_cpp).
type Engine interface {
	// Process runs transcription over the provided PCM32F samples.
	// Returns (deltaText, fullText, language).
	Process(samples []float32) (string, string, string, error)
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
