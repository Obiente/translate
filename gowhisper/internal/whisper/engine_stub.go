//go:build !whisper_cpp

package whisper

// Default stub (no cgo) so the project builds without whisper_cpp tag.
type stubEngine struct{}

func NewEngine(modelPath string) (Engine, error) { return &stubEngine{}, nil }
func (e *stubEngine) Close() error               { return nil }
func (e *stubEngine) Process(samples []float32) (string, string, string, error) {
	return "", "", "", nil
}
func (e *stubEngine) Stream(samples []float32, onSegment func(text string, lang string)) error {
	return nil
}
func (e *stubEngine) SetLanguage(lang string) {}
func (e *stubEngine) GetStreamingConfig() (int, int) {
	return 8000, 960000 // fallback: 0.5s work window, 60s context (1 minute)
}
