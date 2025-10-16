<template>
  <div class="app" :class="{ 'app-focused': showFocusedView }">
    <header class="page-header" v-if="!showFocusedView">
      <div class="page-topbar">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true"></span>
          <span class="brand-name">Obiente Translate</span>
        </div>
        <div class="header-cta">
          <button
            class="button focus-cta"
            type="button"
            @click="showFocusedView = true"
            title="Enter Focus Mode"
          >
            🎯 Enter Focus Mode
          </button>
        </div>
      </div>
      <div class="page-hero">
        <h1>Live Multi-Source Translator</h1>
        <p>
          Add participants, pick their input sources, and let the app
          translate conversations in real time.
        </p>
      </div>
    </header>

    <section class="global-status" v-if="globalStatus && !showFocusedView">
      <div class="status-indicator" :class="globalStatus.type">
        {{ globalStatus.message }}
      </div>
    </section>

    <section class="workspace" v-if="!showFocusedView">
      <aside class="participants-panel">
        <header class="panel-header">
          <h2>Participants</h2>
          <span class="participant-count">{{ channels.length }}</span>
        </header>

        <div class="panel-actions">
          <button
            class="button primary"
            type="button"
            @click="addMicrophoneChannel"
            :disabled="!speechSupported"
          >
            Add Microphone
          </button>
          <button
            class="button"
            type="button"
            @click="addSystemChannel"
            :disabled="hasSystemChannel || !desktopCaptureSupported"
          >
            Add System Audio
          </button>
        </div>

        <div class="mode-control">
          <label for="transcription-mode">Transcription mode</label>
          <select id="transcription-mode" v-model="transcriptionMode">
            <option value="chunked">Chunked uploads (stable)</option>
            <option value="streaming">Streaming (beta)</option>
          </select>
          <p class="mode-hint">
            Chunked uploads send periodic audio blobs. Streaming keeps a live
            WebSocket open for lower latency.
          </p>
          <p class="mode-hint warning" v-if="isAnyChannelActive">
            Active participants continue using their current mode until
            restarted.
          </p>
        </div>

        <p v-if="!speechSupported" class="support-hint">
          Audio capture is unavailable in this browser.
        </p>
        <p v-else-if="!desktopCaptureSupported" class="support-hint">
          System audio capture is unavailable in this browser.
        </p>

        <ul class="participants-list">
          <li v-for="channel in channels" :key="channel.id">
            <article
              class="participant-card"
              :class="{
                selected: channel.id === selectedChannelId,
                active: channel.isActive,
              }"
            >
              <button
                class="participant-select"
                type="button"
                @click="selectChannel(channel.id)"
              >
                <div class="participant-label">
                  <span>{{ channel.label }}</span>
                  <small>
                    {{
                      channel.sourceType === "system"
                        ? "System audio capture"
                        : resolveMicrophoneLabel(channel)
                    }}
                  </small>
                </div>
                <div
                  class="participant-state"
                  :class="{ listening: channel.isActive }"
                >
                  <template v-if="channel.status">
                    {{ channel.status.message }}
                  </template>
                  <template v-else>
                    {{ channel.isActive ? "Listening…" : "Idle" }}
                  </template>
                </div>
              </button>

              <div class="participant-controls">
                <button
                  class="button primary"
                  :class="{ active: channel.isActive }"
                  type="button"
                  @click.stop="toggleChannel(channel.id)"
                >
                  {{ channel.isActive ? "Pause" : "Start" }}
                </button>
                <button
                  class="button"
                  type="button"
                  @click.stop="removeChannel(channel.id)"
                >
                  Remove
                </button>
              </div>

              <div class="participant-status" v-if="channel.status">
                <div class="status-indicator" :class="channel.status.type">
                  {{ channel.status.message }}
                </div>
              </div>
            </article>
          </li>
        </ul>

        <p v-if="channels.length === 0" class="empty-hint">
          Add a participant to begin listening.
        </p>
      </aside>

      <div class="detail-panel" v-if="selectedChannel">
        <header class="detail-header">
          <div>
            <h2>{{ selectedChannel.label }}</h2>
            <p class="channel-meta">
              {{
                selectedChannel.sourceType === "system"
                  ? "System audio input"
                  : "Microphone input"
              }}
              <span v-if="selectedChannel.detectedLanguage">
                • Detected:
                {{ getLanguageName(selectedChannel.detectedLanguage) }}
              </span>
            </p>
          </div>
          <div class="detail-actions">
            <button
              class="button primary"
              :class="{ active: selectedChannel.isActive }"
              type="button"
              @click="toggleChannel(selectedChannel.id)"
            >
              {{ selectedChannel.isActive ? "Pause" : "Start Listening" }}
            </button>
            <button
              class="button"
              type="button"
              @click="removeChannel(selectedChannel.id)"
            >
              Remove
            </button>
          </div>
        </header>

        <div class="detail-controls">
          <div class="control-group">
            <label for="channel-label">Participant name</label>
            <input
              id="channel-label"
              type="text"
              v-model="selectedChannel.label"
            />
          </div>

          <div class="control-group">
            <label for="source-language">Source language</label>
            <select
              id="source-language"
              v-model="selectedChannel.sourceLanguage"
            >
              <option value="auto">Auto detect</option>
              <option
                v-for="language in languages"
                :key="language.code"
                :value="language.code"
              >
                {{ language.name }}
              </option>
            </select>
          </div>

          <div
            class="control-group"
            v-if="selectedChannel.sourceType === 'microphone'"
          >
            <label for="microphone-select">Microphone</label>
            <select
              id="microphone-select"
              v-model="selectedChannel.microphoneDeviceId"
              @change="handleMicrophoneSelectionChange(selectedChannel)"
            >
              <option value="">System default</option>
              <option
                v-for="option in audioInputOptions"
                :key="option.deviceId"
                :value="option.deviceId"
              >
                {{ option.label }}
              </option>
            </select>
            <button
              class="button"
              type="button"
              @click="refreshAudioInputs"
              :disabled="isRefreshingAudioInputs"
            >
              {{ isRefreshingAudioInputs ? "Refreshing…" : "Refresh devices" }}
            </button>
          </div>

          <div class="control-group">
            <label for="target-languages">Target languages</label>
            <select
              id="target-languages"
              v-model="selectedChannel.targetLanguages"
              multiple
              size="4"
            >
              <option
                v-for="language in languages"
                :key="language.code"
                :value="language.code"
              >
                {{ language.name }}
              </option>
            </select>
          </div>

          <div class="control-group">
            <label class="auto-speak-toggle">
              <input type="checkbox" v-model="selectedChannel.autoSpeak" />
              Auto play translations
            </label>
            <span v-if="selectedChannel.autoSpeak" class="tts-indicator">
              {{ isSpeaking ? "Speaking…" : "Ready to speak" }}
            </span>
          </div>
        </div>

        <section class="detail-transcript">
          <article class="transcript-current">
            <h3>Live transcript</h3>
            <p v-if="selectedChannel.liveTranscript">
              {{ selectedChannel.liveTranscript }}
            </p>
            <p v-else class="placeholder">No live transcription yet.</p>
          </article>
          <article class="transcript-final">
            <h3>Last message</h3>
            <p v-if="selectedChannel.lastFinalTranscript">
              {{ selectedChannel.lastFinalTranscript }}
            </p>
            <p v-else class="placeholder">
              Waiting for a finalised transcript.
            </p>
          </article>
        </section>

        <section class="detail-translations">
          <div class="translations-header">
            <h3>Translations</h3>

            <label
              v-if="showTranslationAlternatives"
              class="alternatives-count"
            >
              Alternatives per language
              <select
                v-model.number="translationAlternativesLimit"
                :disabled="!showTranslationAlternatives"
              >
                <option
                  v-for="option in alternativeCountOptions"
                  :key="option"
                  :value="option"
                >
                  {{ option }}
                </option>
              </select>
            </label>
            <label class="alternatives-toggle">
              <input type="checkbox" v-model="showTranslationAlternatives" />
              Show alternative translations
            </label>
          </div>
          <div
            class="live-translation-preview"
            v-if="Object.keys(selectedChannel.liveTranslations).length"
          >
            <h4>Live preview</h4>
            <ul>
              <li
                v-for="(translation, code) in selectedChannel.liveTranslations"
                :key="code"
              >
                <strong>{{ getLanguageName(code) }}:</strong>
                <span>
                  {{
                    translation.primary || translation.alternatives[0] || "—"
                  }}
                </span>
                <ul
                  v-if="
                    showTranslationAlternatives &&
                    getUniqueAlternatives(translation).length
                  "
                  class="translation-alternatives"
                >
                  <li
                    v-for="(alternative, index) in getUniqueAlternatives(
                      translation
                    )"
                    :key="index"
                  >
                    {{ alternative }}
                  </li>
                </ul>
              </li>
            </ul>
          </div>
          <div
            class="translation-list"
            v-if="Object.keys(selectedChannel.translations).length"
          >
            <div
              class="translation-item"
              v-for="(translation, code) in selectedChannel.translations"
              :key="code"
            >
              <strong>{{ getLanguageName(code) }}</strong>
              <p>{{ translation.primary }}</p>
              <ul
                v-if="
                  showTranslationAlternatives &&
                  getUniqueAlternatives(translation).length
                "
                class="translation-alternatives"
              >
                <li
                  v-for="(alternative, index) in getUniqueAlternatives(
                    translation
                  )"
                  :key="index"
                >
                  {{ alternative }}
                </li>
              </ul>
            </div>
          </div>
          <p
            v-else-if="!Object.keys(selectedChannel.liveTranslations).length"
            class="placeholder"
          >
            No translations yet.
          </p>
        </section>
      </div>

      <div class="detail-panel empty" v-else>
        <p>Select a participant to view their details and controls.</p>
      </div>
    </section>

    <!-- Normal conversation view -->
    <section class="conversation-feed" v-if="!showFocusedView">
      <header class="feed-header">
        <h2>Live Conversation</h2>
        <div class="feed-controls">
          <button
            class="button focus-toggle"
            :class="{ active: showFocusedView }"
            @click="showFocusedView = !showFocusedView"
            type="button"
          >
            {{ showFocusedView ? "📋 Normal View" : "🎯 Focus View" }}
          </button>
          <label class="alternatives-toggle">
            <input type="checkbox" v-model="showTranslationAlternatives" />
            Show alternatives
          </label>
          <span class="participant-count"
            >{{ channels.filter((c) => c.isActive).length }} active</span
          >
        </div>
      </header>

      <ConversationFeed
        :channels="channels"
        :recent-history="recentHistory"
        :show-translation-alternatives="showTranslationAlternatives"
        :is-speaking="isSpeaking"
        :get-language-flag="getLanguageFlag"
        :get-language-name="getLanguageName"
        :get-unique-alternatives="getUniqueAlternatives"
        :format-time="formatTime"
        :select-alternative="selectAlternative"
        :play-translation="playTranslation"
      />
    </section>

    <!-- Focused transcription view - Full Screen -->
    <FocusedView
      v-if="showFocusedView"
      :channels="channels"
      :recent-history="recentHistory"
      :get-language-flag="getLanguageFlag"
      :get-language-name="getLanguageName"
      :show-focused-view="showFocusedView"
      :toggle-channel="toggleChannel"
      :add-microphone-channel="addMicrophoneChannel"
      :add-system-channel="addSystemChannel"
      @toggle-focus="showFocusedView = !showFocusedView"
    />

    <section class="history" v-if="!showFocusedView">
      <h2>Recent History</h2>
      <div class="history-list" v-if="recentHistory.length">
        <article
          class="history-item"
          v-for="entry in recentHistory"
          :key="entry.id"
        >
          <header>
            <span>{{ entry.channelLabel }}</span>
            <span>{{ formatTimestamp(entry.timestamp) }}</span>
          </header>
          <p>{{ entry.transcript }}</p>
          <ul>
            <li v-for="(translation, code) in entry.translations" :key="code">
              <strong>{{ getLanguageName(code) }}:</strong>
              {{ translation.primary }}
              <ul
                v-if="
                  showTranslationAlternatives &&
                  getUniqueAlternatives(translation).length
                "
                class="translation-alternatives compact"
              >
                <li
                  v-for="(alternative, index) in getUniqueAlternatives(
                    translation
                  )"
                  :key="index"
                >
                  {{ alternative }}
                </li>
              </ul>
            </li>
          </ul>
        </article>
      </div>
      <p v-else class="placeholder">No recorded history yet.</p>
    </section>

    <footer class="site-footer" v-if="!showFocusedView">
      <div>
        <span>Obiente Translate</span>
        <span class="dot">•</span>
        <span>by Obiente, LLC · Hosted on Obiente Cloud</span>
      </div>
      <div>
        <a href="/about">About</a>
        <span class="dot">•</span>
        <a href="/privacy">Privacy</a>
        <span class="dot">•</span>
        <a href="/terms">Terms</a>
        <span class="dot">•</span>
        <a
          href="https://github.com/Obiente/translate"
          target="_blank"
          rel="noopener"
          >GitHub</a
        >
        <span class="dot">•</span>
        <a href="https://github.com/Obiente" target="_blank" rel="noopener"
          >@Obiente</a
        >
        <span class="dot">•</span>
        <a href="https://obiente.org" target="_blank" rel="noopener"
          >obiente.org</a
        >
        <span class="dot">•</span>
        <a href="https://obiente.com" target="_blank" rel="noopener"
          >obiente.com</a
        >
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
  import { ref, onMounted, watch } from "vue";
  import { useLiveTranslation } from "./composables/useLiveTranslation";
  import FocusedView from "./components/FocusedView.vue";
  import ConversationFeed from "./components/ConversationFeed.vue";
  import "./app.css";
  import { FOCUSED_VIEW_STATE_KEY } from "./constants/storage";

  // Local reactive state
  // Default to settings (non-focused) on first load; then restore last state if available
  const showFocusedView = ref(false);

  onMounted(() => {
    try {
      const saved = localStorage.getItem(FOCUSED_VIEW_STATE_KEY);
      if (saved === "true" || saved === "false") {
        showFocusedView.value = saved === "true";
      }
    } catch {}
  });

  watch(showFocusedView, (val) => {
    try {
      localStorage.setItem(FOCUSED_VIEW_STATE_KEY, String(val));
    } catch {}
  });

  const {
    languages,
    channels,
    selectedChannelId,
    selectedChannel,
    recentHistory,
    globalStatus,
    showTranslationAlternatives,
    translationAlternativesLimit,
    transcriptionMode,
    speechSupported,
    desktopCaptureSupported,
    hasSystemChannel,
    isAnyChannelActive,
    isSpeaking,
    audioInputOptions,
    isRefreshingAudioInputs,
    addMicrophoneChannel,
    addSystemChannel,
    removeChannel,
    toggleChannel,
    selectChannel,
    getLanguageName,
    formatTimestamp,
    handleMicrophoneSelectionChange,
    refreshAudioInputs,
    resolveMicrophoneLabel,
  } = useLiveTranslation();

  // Helper functions for the improved UI
  const getLanguageFlag = (code: string): string => {
    const flags: Record<string, string> = {
      en: "🇺🇸",
      es: "🇪🇸",
      fr: "🇫🇷",
      de: "🇩🇪",
      it: "🇮🇹",
      pt: "🇵🇹",
      ru: "🇷🇺",
      ja: "🇯🇵",
      ko: "🇰🇷",
      zh: "🇨🇳",
      ar: "🇸🇦",
      hi: "🇮🇳",
      tr: "🇹🇷",
      pl: "🇵🇱",
      nl: "🇳🇱",
    };
    return flags[code] || "🌐";
  };

  const formatTime = (timestamp: number | string): string => {
    const date =
      typeof timestamp === "string" ? new Date(timestamp) : new Date(timestamp);
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const selectAlternative = (
    channelId: string,
    languageCode: string,
    alternative: string
  ) => {
    const channel = channels.value.find((c) => c.id === channelId);
    if (channel && channel.liveTranslations[languageCode]) {
      // Move selected alternative to primary position
      const translation = channel.liveTranslations[languageCode];
      const alternatives = [...translation.alternatives];
      const altIndex = alternatives.indexOf(alternative);
      if (altIndex > -1) {
        alternatives.splice(altIndex, 1);
        alternatives.unshift(translation.primary);
        translation.primary = alternative;
        translation.alternatives = alternatives;
      }
    }
  };

  const playTranslation = async (text: string, languageCode: string) => {
    if (isSpeaking.value) return;
    try {
      // Use the composable's speak functionality if available
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = languageCode;
        window.speechSynthesis.speak(utterance);
      }
    } catch (error) {
      console.warn("TTS failed:", error);
    }
  };

  // Helper function to get unique alternatives, excluding duplicates and the primary translation
  const getUniqueAlternatives = (translation: any): string[] => {
    if (
      !translation ||
      !translation.alternatives ||
      !Array.isArray(translation.alternatives)
    ) {
      return [];
    }

    const primary = (translation.primary || "").trim();
    const alternatives = translation.alternatives
      .filter(
        (alt: any): alt is string =>
          typeof alt === "string" && alt.trim() !== "" && alt.trim() !== primary
      )
      .map((alt: string) => alt.trim());

    // Remove duplicates using Set
    return Array.from(new Set(alternatives));
  };

  const alternativeCountOptions = [1, 2, 3, 4, 5];
  defineExpose({
    languages,
    channels,
    selectedChannelId,
    selectedChannel,
    recentHistory,
    globalStatus,
    showTranslationAlternatives,
    translationAlternativesLimit,
    transcriptionMode,
    speechSupported,
    desktopCaptureSupported,
    hasSystemChannel,
    isAnyChannelActive,
    isSpeaking,
    audioInputOptions,
    isRefreshingAudioInputs,
    addMicrophoneChannel,
    addSystemChannel,
    removeChannel,
    toggleChannel,
    selectChannel,
    getLanguageName,
    formatTimestamp,
    handleMicrophoneSelectionChange,
    refreshAudioInputs,
    resolveMicrophoneLabel,
  });
</script>

<style scoped>
  .site-footer {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.85rem;
    margin-top: 2rem;
  }
  .site-footer .dot {
    margin: 0 1rem;
    opacity: 0.6;
  }
  .site-footer a {
    color: #1db954;
    text-decoration: none;
    opacity: 0.9;
  }
  .site-footer a:hover {
    text-decoration: underline;
    opacity: 1;
  }
</style>
