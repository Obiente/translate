<template>
  <div class="conversation-stream">
    <!-- Live transcription indicators -->
    <div
      class="live-indicator"
      v-for="channel in activeChannelsWithContent"
      :key="`live-${channel.id}`"
    >
      <div class="speaker-badge live">
        <div class="speaker-name">{{ channel.label }}</div>
        <div class="live-pulse"></div>
      </div>

      <div class="message-content live">
        <div class="original-text" v-if="channel.liveTranscript">
          <span class="typing-indicator">{{ channel.liveTranscript }}</span>
        </div>

        <div
          class="translations-live"
          v-if="Object.keys(channel.liveTranslations).length"
        >
          <div
            class="translation-item live"
            v-for="(translation, code) in channel.liveTranslations"
            :key="code"
          >
            <div class="translation-header">
              <span class="language-flag">{{ getLanguageFlag(code) }}</span>
              <span class="language-name">{{ getLanguageName(code) }}</span>
            </div>
            <div class="translation-text typing">
              {{ translation.primary || translation.alternatives[0] || "..." }}
            </div>
            <div
              class="alternatives-list"
              v-if="
                showTranslationAlternatives &&
                getUniqueAlternatives(translation).length
              "
            >
              <button
                class="alternative-option"
                v-for="(alt, idx) in getUniqueAlternatives(translation)"
                :key="idx"
                @click="selectAlternative(channel.id, code, alt)"
              >
                {{ alt }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Final messages from history -->
    <div class="message" v-for="entry in recentHistory" :key="entry.id">
      <div class="speaker-badge final">
        <div class="speaker-name">{{ entry.channelLabel }}</div>
        <div class="timestamp">{{ formatTime(entry.timestamp) }}</div>
      </div>

      <div class="message-content final">
        <div class="original-text">
          {{ entry.transcript }}
        </div>

        <div
          class="translations-final"
          v-if="Object.keys(entry.translations).length"
        >
          <div
            class="translation-item final"
            v-for="(translation, code) in entry.translations"
            :key="code"
          >
            <div class="translation-header">
              <span class="language-flag">{{ getLanguageFlag(code) }}</span>
              <span class="language-name">{{ getLanguageName(code) }}</span>
              <button
                class="play-translation"
                @click="playTranslation(translation.primary, code)"
                :disabled="isSpeaking"
              >
                🔊
              </button>
            </div>
            <div class="translation-text">
              {{ translation.primary }}
            </div>
            <div
              class="alternatives-list"
              v-if="
                showTranslationAlternatives &&
                getUniqueAlternatives(translation).length
              "
            >
              <span
                class="alternative-option"
                v-for="(alt, idx) in getUniqueAlternatives(translation)"
                :key="idx"
              >
                {{ alt }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div
      class="empty-state"
      v-if="!recentHistory.length && !activeChannelsWithContent.length"
    >
      <div class="empty-icon">💬</div>
      <h3>Start a conversation</h3>
      <p>Add participants and start listening to see live translations here.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { computed } from "vue";

  interface Translation {
    primary: string;
    alternatives: string[];
  }

  interface Channel {
    id: string;
    label: string;
    isActive: boolean;
    liveTranscript: string;
    liveTranslations: Record<string, Translation>;
  }

  interface HistoryEntry {
    id: string;
    channelLabel: string;
    timestamp: number | string;
    transcript: string;
    translations: Record<string, Translation>;
  }

  interface Props {
    channels: Channel[];
    recentHistory: HistoryEntry[];
    showTranslationAlternatives: boolean;
    isSpeaking: boolean;
    getLanguageFlag: (code: string) => string;
    getLanguageName: (code: string) => string;
    getUniqueAlternatives: (translation: Translation) => string[];
    formatTime: (timestamp: number | string) => string;
    selectAlternative: (
      channelId: string,
      languageCode: string,
      alternative: string
    ) => void;
    playTranslation: (text: string, languageCode: string) => Promise<void>;
  }

  const props = defineProps<Props>();

  const activeChannelsWithContent = computed(() =>
    props.channels.filter(
      (c) =>
        c.isActive &&
        (c.liveTranscript || Object.keys(c.liveTranslations).length)
    )
  );
</script>
<style scoped src="../focused-view.css"></style>
