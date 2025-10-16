<template>
  <div
    class="focused-transcription"
    :class="{ 'focused-transcription--history': !isLive }"
  >
    <!-- Speaker Section -->
    <div class="focused-speaker">
      <div
        class="speaker-avatar"
        :class="{ 'speaker-avatar--history': !isLive }"
      >
        {{ channel.label.charAt(0).toUpperCase() }}
      </div>
      <div class="speaker-info">
        <h3 class="speaker-name">{{ channel.label }}</h3>
        <div class="live-status" v-if="isLive">
          <div class="pulse-dot"></div>
          <span>Speaking live</span>
        </div>
        <div class="history-status" v-else>
          <span>Recent</span>
        </div>
      </div>
    </div>

    <!-- Original Text (Small) -->
    <div
      ref="originalTextRef"
      class="focused-original-text"
      v-if="channel.liveTranscript"
    >
      {{ channel.liveTranscript }}
    </div>

    <!-- Translations (Large - Main Focus) -->
    <div
      class="focused-translations"
      v-if="Object.keys(channel.liveTranslations).length"
    >
      <div
        class="focused-translation"
        v-for="(translation, code) in channel.liveTranslations"
        :key="code"
      >
        <div class="translation-lang">
          <span class="language-flag">{{ getLanguageFlag(code) }}</span>
          <span class="language-name">{{ getLanguageName(code) }}</span>
        </div>
        <div
          :ref="el => translationRefs[code] = el as HTMLElement"
          class="translation-text"
          :class="{ typing: isLive }"
        >
          <LyricsDisplay
            :text="translation.primary || translation.alternatives[0] || '...'"
            :streaming="isLive"
            :isFinal="props.isFinal || !isLive"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { ref, watch, nextTick } from "vue";
  import LyricsDisplay from "./LyricsDisplay.vue";

  interface Translation {
    primary: string;
    alternatives: string[];
  }

  interface Channel {
    id: string;
    label: string;
    liveTranscript: string;
    liveTranslations: Record<string, Translation>;
  }

  interface Props {
    channel: Channel;
    getLanguageFlag: (code: string) => string;
    getLanguageName: (code: string) => string;
    isLive?: boolean;
    isFinal?: boolean;
  }

  const props = withDefaults(defineProps<Props>(), {
    isLive: true,
    isFinal: false,
  });

  // Refs for auto-scrolling
  const originalTextRef = ref<HTMLElement | null>(null);
  const translationRefs = ref<Record<string, HTMLElement | null>>({});

  // Auto-scroll helpers
  const scrollToBottom = (element: HTMLElement | null) => {
    if (element && element.scrollHeight > element.clientHeight) {
      element.scrollTop = element.scrollHeight;
    }
  };

  const scrollTranslationToCenter = (element: HTMLElement | null) => {
    if (!element) return;

    // If we're inside the unified full-screen feed, center the current line
    const feed = element.closest(".full-screen-feed") as HTMLElement | null;
    const current = element.querySelector(
      ".translation-line.current"
    ) as HTMLElement | null;
    if (feed && current) {
      try {
        current.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      } catch (err) {
        // fall through to other strategies
      }
    }

    // Fallback: scroll to bottom for non-fullscreen or if no current line found
    scrollToBottom(element);
  };

  // Watch for live transcript changes and auto-scroll
  watch(
    () => props.channel.liveTranscript,
    () => {
      if (props.isLive) {
        nextTick(() => {
          scrollToBottom(originalTextRef.value);
        });
      }
    }
  );

  // Watch for translation changes and auto-scroll
  watch(
    () => props.channel.liveTranslations,
    () => {
      if (props.isLive) {
        nextTick(() => {
          Object.values(translationRefs.value).forEach(
            scrollTranslationToCenter
          );
        });
      }
    },
    { deep: true }
  );
</script>
<style scoped src="../focused-view.css"></style>
