<template>
  <div class="app">
    <header class="page-header">
      <h1>Live Multi-Source Translator</h1>
      <p>
        Add participants, pick their input sources, and let the app translate conversations in real time.
      </p>
    </header>

    <section class="global-status" v-if="globalStatus">
      <div class="status-indicator" :class="globalStatus.type">
        {{ globalStatus.message }}
      </div>
    </section>

    <section class="workspace">
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
            Chunked uploads send periodic audio blobs. Streaming keeps a live WebSocket open for lower latency.
          </p>
          <p class="mode-hint warning" v-if="isAnyChannelActive">
            Active participants continue using their current mode until restarted.
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
            <select id="source-language" v-model="selectedChannel.sourceLanguage">
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

          <div class="control-group" v-if="selectedChannel.sourceType === 'microphone'">
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
            <label class="alternatives-toggle">
              <input
                type="checkbox"
                v-model="showTranslationAlternatives"
              />
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
                <strong>{{ getLanguageName(code) }}:</strong> {{ translation }}
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
                v-if="showTranslationAlternatives && translation.alternatives.length"
                class="translation-alternatives"
              >
                <li
                  v-for="(alternative, index) in translation.alternatives"
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

    <section class="live-overview">
      <header class="overview-header">
        <h2>Live Overview</h2>
        <span class="participant-count">{{ channels.length }}</span>
      </header>
      <div class="overview-grid">
        <article
          class="overview-card"
          v-for="channel in channels"
          :key="channel.id"
          :class="{ active: channel.isActive }"
        >
          <header class="overview-card-header">
            <strong>{{ channel.label }}</strong>
            <small v-if="channel.status">{{ channel.status.message }}</small>
          </header>

          <div class="overview-transcript">
            <span class="label">Live</span>
            <span>{{ channel.liveTranscript || "—" }}</span>
          </div>

          <div class="overview-transcript">
            <span class="label">Last</span>
            <span>{{ channel.lastFinalTranscript || "—" }}</span>
          </div>

          <ul
            class="overview-translations live"
            v-if="Object.keys(channel.liveTranslations).length"
          >
            <li
              v-for="(translation, code) in channel.liveTranslations"
              :key="code"
            >
              <strong>{{ getLanguageName(code) }} (live):</strong> {{ translation }}
            </li>
          </ul>

          <ul
            class="overview-translations"
            v-if="Object.keys(channel.translations).length"
          >
            <li v-for="(translation, code) in channel.translations" :key="code">
              <strong>{{ getLanguageName(code) }}:</strong> {{ translation.primary }}
              <ul
                v-if="showTranslationAlternatives && translation.alternatives.length"
                class="translation-alternatives compact"
              >
                <li
                  v-for="(alternative, index) in translation.alternatives"
                  :key="index"
                >
                  {{ alternative }}
                </li>
              </ul>
            </li>
          </ul>
          <p
            v-else-if="!Object.keys(channel.liveTranslations).length"
            class="placeholder"
          >
            No translations yet.
          </p>
        </article>
      </div>
    </section>

    <section class="history">
      <h2>Recent History</h2>
      <div class="history-list" v-if="recentHistory.length">
        <article class="history-item" v-for="entry in recentHistory" :key="entry.id">
          <header>
            <span>{{ entry.channelLabel }}</span>
            <span>{{ formatTimestamp(entry.timestamp) }}</span>
          </header>
          <p>{{ entry.transcript }}</p>
          <ul>
            <li v-for="(translation, code) in entry.translations" :key="code">
              <strong>{{ getLanguageName(code) }}:</strong> {{ translation.primary }}
              <ul
                v-if="showTranslationAlternatives && translation.alternatives.length"
                class="translation-alternatives compact"
              >
                <li
                  v-for="(alternative, index) in translation.alternatives"
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
  </div>
</template>

<script setup lang="ts">
import { useLiveTranslation } from "./composables/useLiveTranslation";

const {
  languages,
  channels,
  selectedChannelId,
  selectedChannel,
  recentHistory,
  globalStatus,
  showTranslationAlternatives,
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

defineExpose({
  languages,
  channels,
  selectedChannelId,
  selectedChannel,
  recentHistory,
  globalStatus,
  showTranslationAlternatives,
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
  .app {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .page-header {
    text-align: center;
  }

  .page-header h1 {
    margin-bottom: 0.5rem;
    color: #646cff;
  }

  .global-status {
    display: flex;
    justify-content: center;
  }

  .workspace {
    display: grid;
    grid-template-columns: minmax(250px, 320px) 1fr;
    gap: 1.5rem;
    align-items: flex-start;
  }

  .participants-panel {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .participant-count {
    background: #646cff;
    color: #fff;
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.85rem;
  }

  .panel-actions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .mode-control {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }

  .mode-hint {
    font-size: 0.85rem;
    color: #b0b0b0;
    margin: 0;
  }

  .mode-hint.warning {
    color: #ffb74d;
  }

  .support-hint {
    color: #ffb74d;
    font-size: 0.85rem;
  }

  .participants-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0;
    margin: 0;
    list-style: none;
  }

  .participant-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 0.75rem;
    border: 1px solid transparent;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    transition: border-color 0.2s, background 0.2s;
  }

  .participant-card.selected {
    border-color: #646cff;
    background: rgba(100, 108, 255, 0.12);
  }

  .participant-card.active {
    border-color: rgba(76, 175, 80, 0.6);
  }

  .participant-select {
    border: none;
    background: transparent;
    color: inherit;
    font: inherit;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    cursor: pointer;
    text-align: left;
  }

  .participant-label span {
    font-weight: 600;
  }

  .participant-label small {
    display: block;
    margin-top: 0.25rem;
    color: #b0b0b0;
    font-size: 0.85rem;
  }

  .participant-state {
    font-size: 0.85rem;
    color: #b0b0b0;
  }

  .participant-state.listening {
    color: #4caf50;
  }

  .participant-controls {
    display: flex;
    gap: 0.5rem;
  }

  .participant-status {
    margin-top: 0.5rem;
  }

  .empty-hint {
    color: #b0b0b0;
    font-size: 0.9rem;
  }

  .detail-panel {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    min-height: 100%;
  }

  .detail-panel.empty {
    align-items: center;
    justify-content: center;
    text-align: center;
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
  }

  .detail-header h2 {
    margin: 0;
    font-size: 1.4rem;
  }

  .channel-meta {
    margin: 0.35rem 0 0;
    color: #b0b0b0;
    font-size: 0.9rem;
  }

  .detail-actions {
    display: flex;
    gap: 0.75rem;
  }

  .detail-controls {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    align-items: end;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  select,
  input[type="checkbox"],
  button,
  input[type="text"] {
    font: inherit;
  }

  select,
  input[type="text"] {
    background: rgba(0, 0, 0, 0.35);
    color: inherit;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 6px;
    padding: 0.5rem;
  }

  .auto-speak-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
  }

  .button {
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.08);
    color: inherit;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .button.primary {
    background: #646cff;
    border-color: #646cff;
    color: #fff;
  }

  .button.primary.danger {
    background: #ef5350;
    border-color: #ef5350;
  }

  .button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.15);
  }

  .button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .button.active {
    background: #ef5350;
    border-color: #ef5350;
  }

  .detail-transcript {
    display: grid;
    gap: 1rem;
  }

  .transcript-current,
  .transcript-final {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .placeholder {
    color: #a0a0a0;
    font-style: italic;
  }

  .detail-translations {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .live-translation-preview {
    background: rgba(0, 0, 0, 0.25);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .live-translation-preview h4 {
    margin: 0;
    font-size: 1rem;
    color: #dcdcff;
  }

  .live-translation-preview ul {
    margin: 0;
    padding-left: 0;
    list-style: none;
    display: grid;
    gap: 0.4rem;
  }

  .translations-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .alternatives-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    color: #d0d0d0;
  }

  .translation-list {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }

  .translation-item {
    background: rgba(0, 0, 0, 0.25);
    padding: 0.75rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .translation-alternatives {
    margin: 0.5rem 0 0;
    padding-left: 1.1rem;
    display: grid;
    gap: 0.25rem;
    color: #d0d0d0;
    font-size: 0.9rem;
    list-style: disc;
  }

  .translation-alternatives.compact {
    margin-top: 0.25rem;
  }

  .live-overview {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .overview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .overview-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  }

  .overview-card {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .overview-card.active {
    border-color: rgba(76, 175, 80, 0.6);
  }

  .overview-card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.75rem;
  }

  .overview-transcript {
    display: flex;
    gap: 0.5rem;
  }

  .overview-transcript .label {
    color: #b0b0b0;
    font-size: 0.85rem;
  }

  .overview-translations {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.4rem;
  }

  .overview-translations.live {
    opacity: 0.85;
  }

  .tts-indicator {
    color: #ffb74d;
    font-size: 0.9rem;
  }

  .status-indicator {
    padding: 0.75rem 1rem;
    border-radius: 6px;
  }

  .status-indicator.info {
    background: rgba(100, 148, 255, 0.2);
  }

  .status-indicator.success {
    background: rgba(76, 175, 80, 0.2);
  }

  .status-indicator.error {
    background: rgba(244, 67, 54, 0.2);
  }

  .status-indicator.processing,
  .status-indicator.listening {
    background: rgba(255, 213, 79, 0.2);
  }

  .history {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .history-list {
    display: grid;
    gap: 1rem;
  }

  .history-item {
    background: rgba(0, 0, 0, 0.35);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .history-item header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }

  .history-item ul {
    margin: 0.5rem 0 0;
    padding-left: 1rem;
    display: grid;
    gap: 0.25rem;
  }

  @media (min-width: 540px) {
    .panel-actions {
      flex-direction: row;
    }
  }

  @media (max-width: 860px) {
    .workspace {
      grid-template-columns: 1fr;
    }

    .detail-panel {
      order: 2;
    }

    .participants-panel {
      order: 1;
    }
  }
</style>
