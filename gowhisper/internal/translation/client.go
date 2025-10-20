package translation

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	base    string
	http    *http.Client
	timeout time.Duration
}

func New(base string, timeoutSec int) *Client {
	if timeoutSec <= 0 {
		timeoutSec = 8
	}
	return &Client{
		base:    base,
		http:    &http.Client{Timeout: time.Duration(timeoutSec) * time.Second},
		timeout: time.Duration(timeoutSec) * time.Second,
	}
}

// Translate requests translations for text into targets.
// It calls the translation endpoint once per target using the LibreTranslate
// compatible payload (q, source, target, format, alternatives) and returns
// a map[target] => { primary: string, alternatives: []string } so the
// websocket server can forward a normalized translations object to clients.
func (c *Client) Translate(ctx context.Context, text string, source string, targets []string, altLimit int) (map[string]map[string]any, error) {
	if c == nil || c.base == "" || len(targets) == 0 || strings.TrimSpace(text) == "" {
		return map[string]map[string]any{}, nil
	}

	out := make(map[string]map[string]any, len(targets))

	// Prepare a sensible source value: use provided source if non-empty, otherwise "auto"
	src := strings.TrimSpace(source)
	if src == "" {
		src = "auto"
	}

	for _, tgt := range targets {
		payload := map[string]any{
			"q":      text,
			"source": src,
			"target": tgt,
			"format": "text",
		}
		if altLimit > 0 {
			payload["alternatives"] = altLimit
		}

		b, _ := json.Marshal(payload)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/translate", bytes.NewReader(b))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.http.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("translation http %d for target %s", resp.StatusCode, tgt)
		}

		// Expect LibreTranslate-like response: translatedText, alternatives, detectedLanguage
		var lr struct {
			TranslatedText   string   `json:"translatedText"`
			Alternatives     []string `json:"alternatives"`
			DetectedLanguage string   `json:"detectedLanguage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&lr); err != nil {
			return nil, err
		}

		// Build back a flexible structure the frontend normalizer can understand
		entry := map[string]any{"primary": strings.TrimSpace(lr.TranslatedText)}
		if len(lr.Alternatives) > 0 {
			alts := make([]string, 0, len(lr.Alternatives))
			for _, a := range lr.Alternatives {
				s := strings.TrimSpace(a)
				if s != "" {
					alts = append(alts, s)
				}
			}
			if len(alts) > 0 {
				entry["alternatives"] = alts
			}
		}
		if lr.DetectedLanguage != "" {
			entry["detectedLanguage"] = lr.DetectedLanguage
		}
		out[tgt] = entry
	}

	return out, nil
}
