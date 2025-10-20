package main

import (
	"net/http"
	"os"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/obiente/translate/gowhisper/internal/config"
	serverhttp "github.com/obiente/translate/gowhisper/internal/http"
)

func main() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnixMs
	lvl := zerolog.InfoLevel
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		if l, err := zerolog.ParseLevel(v); err == nil {
			lvl = l
		}
	}
	log.Logger = log.Level(lvl)

	cfg := config.Load()
	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      serverhttp.NewRouter(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	log.Info().Str("addr", cfg.Addr).Msg("whisper-go server starting")
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal().Err(err).Msg("server failed")
	}
}
