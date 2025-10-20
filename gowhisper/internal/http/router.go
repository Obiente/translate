package http

import (
	"encoding/json"
	"net/http"

	"github.com/obiente/translate/server-go/internal/config"
	"github.com/obiente/translate/server-go/internal/ws"
)

func NewRouter() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	// Streaming transcription WebSocket
	cfg := config.Load()
	wss := ws.NewServer(cfg)
	mux.HandleFunc("/ws/transcribe", wss.Handle)
	return mux
}
