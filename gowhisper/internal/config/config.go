package config

import (
	"os"
	"strconv"
)

type Config struct {
	Addr                  string
	ModelPath             string
	TranslationBaseURL    string
	TranslationEnabled    bool
	TranslationTimeoutSec int
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getenvBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		switch v {
		case "0", "false", "no", "off", "False", "FALSE":
			return false
		default:
			return true
		}
	}
	return def
}

func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func Load() Config {
	return Config{
		Addr:                  getenv("WHISPER_GO_ADDR", ":8080"),
		ModelPath:             getenv("WHISPER_MODEL_PATH", "./models/ggml-base.en.bin"),
		TranslationBaseURL:    getenv("TRANSLATION_BASE_URL", "https://libretranslate.obiente.cloud"),
		TranslationEnabled:    getenvBool("WHISPER_SERVER_TRANSLATIONS", true),
		TranslationTimeoutSec: getenvInt("TRANSLATION_TIMEOUT", 8),
	}
}
