package translation

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Client struct {
	base    string
	http    *http.Client
	timeout time.Duration
	cache   sync.Map
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

type TranslationEntry struct {
	Primary          string   `json:"primary"`
	Alternatives     []string `json:"alternatives,omitempty"`
	DetectedLanguage string   `json:"detectedLanguage,omitempty"`
}

type TranslateRequest struct {
	Text         string   `json:"q"`
	Source       string   `json:"source"`
	Target       string   `json:"target"`
	Format       string   `json:"format"`
	Alternatives int      `json:"alternatives,omitempty"`
	Targets      []string `json:"targets,omitempty"`
}

type LibreTranslateResponse struct {
	TranslatedText   string          `json:"translatedText"`
	Alternatives     []string        `json:"alternatives"`
	DetectedLanguage json.RawMessage `json:"detectedLanguage"`
}

func decodeDetectedLanguage(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return strings.TrimSpace(asString)
	}

	var asObject map[string]any
	if err := json.Unmarshal(raw, &asObject); err == nil {
		for _, key := range []string{"language", "lang", "code"} {
			if value, ok := asObject[key].(string); ok {
				return strings.TrimSpace(value)
			}
		}
	}

	var asArray []map[string]any
	if err := json.Unmarshal(raw, &asArray); err == nil {
		for _, item := range asArray {
			for _, key := range []string{"language", "lang", "code"} {
				if value, ok := item[key].(string); ok {
					return strings.TrimSpace(value)
				}
			}
		}
	}

	return ""
}

type LanguageEntry struct {
	Code string `json:"code"`
	Name string `json:"name"`
}

type DetectResult struct {
	Language   string  `json:"language"`
	Confidence float64 `json:"confidence"`
}

func normalizeBase(base string) string {
	return strings.TrimRight(strings.TrimSpace(base), "/")
}

func (c *Client) endpoint(path string) string {
	return normalizeBase(c.base) + path
}

func (c *Client) cacheKey(text string, source string, target string, altLimit int) string {
	return strings.Join([]string{
		strings.TrimSpace(text),
		strings.TrimSpace(source),
		strings.TrimSpace(target),
		fmt.Sprintf("%d", altLimit),
	}, "\x00")
}

func normalizeAlternatives(values []string, primary string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	primary = strings.TrimSpace(primary)
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || trimmed == primary || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		out = append(out, trimmed)
	}
	return out
}

func normalizeLang(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func (c *Client) TranslateOne(ctx context.Context, text string, source string, target string, altLimit int) (TranslationEntry, error) {
	if c == nil || strings.TrimSpace(c.base) == "" {
		return TranslationEntry{}, nil
	}

	text = strings.TrimSpace(text)
	target = normalizeLang(target)
	source = normalizeLang(source)
	if source == "" {
		source = "auto"
	}
	if altLimit < 0 {
		altLimit = 0
	}
	if altLimit > 5 {
		altLimit = 5
	}
	if text == "" || target == "" {
		return TranslationEntry{}, nil
	}
	if source != "auto" && source == target {
		entry := TranslationEntry{Primary: text, Alternatives: []string{}, DetectedLanguage: source}
		c.cache.Store(c.cacheKey(text, source, target, altLimit), entry)
		return entry, nil
	}

	key := c.cacheKey(text, source, target, altLimit)
	if cached, ok := c.cache.Load(key); ok {
		if entry, ok := cached.(TranslationEntry); ok {
			return entry, nil
		}
	}

	payload := TranslateRequest{
		Text:         text,
		Source:       source,
		Target:       target,
		Format:       "text",
		Alternatives: altLimit,
	}

	b, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint("/translate"), bytes.NewReader(b))
	if err != nil {
		return TranslationEntry{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return TranslationEntry{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return TranslationEntry{}, fmt.Errorf("translation http %d for target %s: %s", resp.StatusCode, target, strings.TrimSpace(string(body)))
	}

	var lr LibreTranslateResponse
	if err := json.NewDecoder(resp.Body).Decode(&lr); err != nil {
		return TranslationEntry{}, err
	}

	entry := TranslationEntry{
		Primary:          strings.TrimSpace(lr.TranslatedText),
		Alternatives:     normalizeAlternatives(lr.Alternatives, lr.TranslatedText),
		DetectedLanguage: decodeDetectedLanguage(lr.DetectedLanguage),
	}
	if normalizeLang(entry.DetectedLanguage) == target {
		entry.Primary = text
		entry.Alternatives = []string{}
	}
	c.cache.Store(key, entry)
	return entry, nil
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
	var mu sync.Mutex
	var wg sync.WaitGroup
	errCh := make(chan error, len(targets))
	seen := make(map[string]bool, len(targets))

	for _, rawTarget := range targets {
		tgt := normalizeLang(rawTarget)
		if tgt == "" || seen[tgt] {
			continue
		}
		seen[tgt] = true
		wg.Add(1)
		go func(target string) {
			defer wg.Done()
			entry, err := c.TranslateOne(ctx, text, source, target, altLimit)
			if err != nil {
				errCh <- err
				return
			}
			if entry.Primary == "" && len(entry.Alternatives) == 0 {
				return
			}
			payload := map[string]any{"primary": entry.Primary}
			if len(entry.Alternatives) > 0 {
				payload["alternatives"] = entry.Alternatives
			}
			if entry.DetectedLanguage != "" {
				payload["detectedLanguage"] = entry.DetectedLanguage
			}
			mu.Lock()
			out[target] = payload
			mu.Unlock()
		}(tgt)
	}
	wg.Wait()
	close(errCh)

	if len(out) > 0 {
		return out, nil
	}
	for err := range errCh {
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}

func (c *Client) Detect(ctx context.Context, text string) ([]DetectResult, error) {
	text = strings.TrimSpace(text)
	if c == nil || strings.TrimSpace(c.base) == "" || text == "" {
		return []DetectResult{}, nil
	}

	b, _ := json.Marshal(map[string]string{"q": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint("/detect"), bytes.NewReader(b))
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
		return nil, fmt.Errorf("detect http %d", resp.StatusCode)
	}

	var results []DetectResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func (c *Client) Languages(ctx context.Context) ([]LanguageEntry, error) {
	if c == nil || strings.TrimSpace(c.base) == "" {
		return []LanguageEntry{}, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint("/languages"), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("languages http %d", resp.StatusCode)
	}

	var entries []LanguageEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}
	return entries, nil
}
