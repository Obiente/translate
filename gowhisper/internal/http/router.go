package http

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/obiente/translate/gowhisper/internal/audio"
	"github.com/obiente/translate/gowhisper/internal/config"
	"github.com/obiente/translate/gowhisper/internal/translation"
	weng "github.com/obiente/translate/gowhisper/internal/whisper"
	"github.com/obiente/translate/gowhisper/internal/ws"
)

func NewRouter() http.Handler {
	mux := http.NewServeMux()
	cfg := config.Load()
	translator := translation.New(cfg.TranslationBaseURL, cfg.TranslationTimeoutSec)
	var engineMu sync.Mutex
	var engine weng.Engine

	ensureEngine := func() (weng.Engine, error) {
		engineMu.Lock()
		defer engineMu.Unlock()
		if engine != nil {
			return engine, nil
		}
		created, err := weng.NewEngine(cfg.ModelPath)
		if err != nil {
			return nil, err
		}
		engine = created
		return engine, nil
	}

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"ok":                  true,
			"translationsEnabled": cfg.TranslationEnabled,
			"translationBase":     cfg.TranslationBaseURL,
		})
	})

	mux.HandleFunc("/translate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req translation.TranslateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Text) == "" {
			writeJSON(w, map[string]any{"translatedText": "", "alternatives": []string{}})
			return
		}
		ctx, cancel := timeoutContext(r, cfg.TranslationTimeoutSec)
		defer cancel()

		if len(req.Targets) > 0 {
			translations, err := translator.Translate(ctx, req.Text, req.Source, req.Targets, req.Alternatives)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			writeJSON(w, map[string]any{"translations": translations})
			return
		}

		entry, err := translator.TranslateOne(ctx, req.Text, req.Source, req.Target, req.Alternatives)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, translation.LibreTranslateResponse{
			TranslatedText:   entry.Primary,
			Alternatives:     entry.Alternatives,
			DetectedLanguage: entry.DetectedLanguage,
		})
	})

	mux.HandleFunc("/detect", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Text string `json:"q"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		ctx, cancel := timeoutContext(r, cfg.TranslationTimeoutSec)
		defer cancel()
		results, err := translator.Detect(ctx, req.Text)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, results)
	})

	mux.HandleFunc("/languages", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		ctx, cancel := timeoutContext(r, cfg.TranslationTimeoutSec)
		defer cancel()
		languages, err := translator.Languages(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, languages)
	})

	mux.HandleFunc("/transcribe", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := r.ParseMultipartForm(64 << 20); err != nil {
			http.Error(w, "invalid multipart form", http.StatusBadRequest)
			return
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "file is required", http.StatusBadRequest)
			return
		}
		defer file.Close()
		content, err := io.ReadAll(file)
		if err != nil || len(content) == 0 {
			http.Error(w, "empty audio", http.StatusBadRequest)
			return
		}

		pcm, err := decodeUploadAudio(content, header.Header.Get("Content-Type"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		eng, err := ensureEngine()
		if err != nil {
			http.Error(w, fmt.Sprintf("engine init failed: %v", err), http.StatusInternalServerError)
			return
		}
		if lang := strings.TrimSpace(r.FormValue("language")); lang != "" {
			eng.SetLanguage(lang)
		} else {
			eng.SetLanguage("auto")
		}

		text, full, language, err := eng.Process(pcm)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if strings.TrimSpace(full) != "" {
			text = full
		}

		targets := parseTargets(r.FormValue("target_languages"))
		if len(targets) == 0 {
			targets = parseTargets(r.FormValue("targets"))
		}
		alts, _ := strconv.Atoi(strings.TrimSpace(r.FormValue("translation_alternatives")))
		ctx, cancel := timeoutContext(r, cfg.TranslationTimeoutSec)
		defer cancel()
		translations, err := translator.Translate(ctx, text, language, targets, alts)
		if err != nil {
			translations = map[string]map[string]any{}
		}

		writeJSON(w, map[string]any{
			"text":         text,
			"language":     language,
			"translations": translations,
		})
	})

	// Streaming transcription WebSocket
	wss := ws.NewServer(cfg)
	mux.HandleFunc("/ws/transcribe", wss.Handle)
	return mux
}

func parseTargets(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var values []string
	if err := json.Unmarshal([]byte(raw), &values); err != nil {
		values = strings.Split(raw, ",")
	}
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		out = append(out, trimmed)
	}
	return out
}

func decodeUploadAudio(content []byte, contentType string) ([]float32, error) {
	if strings.Contains(contentType, "wav") || (len(content) > 12 && string(content[:4]) == "RIFF") {
		pcm, sampleRate, err := audio.DecodeWAVToFloat32(content)
		if err != nil {
			return nil, err
		}
		if sampleRate != 16000 && sampleRate > 0 {
			pcm = audio.ResampleLinear(pcm, sampleRate, 16000)
		}
		return pcm, nil
	}

	tmp, err := os.CreateTemp("", "obiente-upload-*")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		return nil, err
	}
	_ = tmp.Close()

	cmd := exec.Command(
		"ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-i", tmp.Name(),
		"-ac", "1",
		"-ar", "16000",
		"-f", "s16le",
		"pipe:1",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("decode audio failed; install ffmpeg for webm/ogg/mp4 uploads")
	}
	pcm, _, err := audio.DecodePCM16LEToFloat32(out, 16000)
	return pcm, err
}

func writeJSON(w http.ResponseWriter, payload any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(payload)
}

func timeoutContext(r *http.Request, timeoutSec int) (context.Context, context.CancelFunc) {
	if timeoutSec <= 0 {
		timeoutSec = 8
	}
	return context.WithTimeout(r.Context(), time.Duration(timeoutSec)*time.Second)
}
