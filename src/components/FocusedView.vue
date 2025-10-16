<template>
  <div class="focused-view" :class="{ 'full-screen': isFullScreen }">
    <!-- Header with controls - Hidden in full screen -->
    <header class="focused-header" v-if="!isFullScreen">
      <div class="focused-controls">
        <div class="channel-quick-controls">
          <div class="speaking-badges" v-if="activeChannelsWithContent.length">
            <ParticipantChip
              v-for="channel in activeChannelsWithContent"
              :key="`badge-${channel.id}`"
              :label="channel.label"
            />
          </div>
          <button
            v-for="channel in props.channels"
            :key="channel.id"
            class="button channel-toggle"
            :class="{ active: channel.isActive }"
            @click="props.toggleChannel(channel.id)"
            type="button"
            :title="`${channel.isActive ? 'Pause' : 'Start'} ${channel.label}`"
          >
            {{ channel.isActive ? "⏸️" : "▶️" }} {{ channel.label }}
          </button>
        </div>
        <div class="header-actions">
          <button
            class="button"
            @click="toggleFullScreen"
            type="button"
            title="Enter Full Screen"
          >
            ⛶ Full Screen
          </button>

          <button
            class="button active"
            @click="$emit('toggleFocus')"
            type="button"
          >
            📋 Exit Focus
          </button>
        </div>
      </div>
    </header>

    <!-- Full Screen Controls - Minimal floating controls -->
    <div class="full-screen-controls" v-if="isFullScreen">
      <button
        class="button full-screen-exit"
        @click="toggleFullScreen"
        type="button"
        title="Exit Full Screen"
      >
        ✕
      </button>
    </div>

    <!-- Content Area - Bottom Origin Layout -->
    <div class="focused-content">
      <!-- When not in full-screen, keep separate sections for history and live -->
      <template v-if="!isFullScreen">
        <div class="embedded-lyrics" v-if="unifiedFeed && unifiedFeed.segments.length">
          <LyricsDisplay
            :segments="unifiedFeed.segments"
            :streaming="unifiedFeed.streaming"
            :isFinal="unifiedFeed.isFinal"
            :speed="30"
            :centerOnStable="true"
            overlayTop="clamp(80px, 14vh, 140px)"
          />
        </div>
      </template>

      <!-- Full screen: one unified Spotify-style interleaved feed (history + live) -->
      <template v-else>
        <div class="full-screen-participants-bar" v-if="participants.length">
          <ParticipantChip
            v-for="p in participants"
            :key="p.id"
            :label="p.label"
          />
        </div>
        <div class="full-screen-feed" v-if="unifiedFeed && unifiedFeed.segments.length">
          <LyricsDisplay
            :segments="unifiedFeed.segments"
            :streaming="unifiedFeed.streaming"
            :isFinal="unifiedFeed.isFinal"
            :speed="30"
            :centerOnStable="true"
            overlayTop="clamp(56px, 12vh, 120px)"
          />
        </div>
      </template>

      <!-- Empty State -->
      <div
        class="focused-empty"
        v-if="
          !activeChannelsWithContent.length &&
          !recentHistoryWithTranslations.length
        "
      >
        <div class="empty-animation">
          <div class="microphone-icon">🎤</div>
          <div class="sound-waves">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
          </div>
        </div>
        <h3>Waiting for speech...</h3>
        <p>
          Start speaking into any active microphone to see live translations
          here
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { ref, computed, onMounted, onUnmounted } from "vue";

  // ...existing code...

  const isFullScreen = ref(false);

  const toggleFullScreen = () => {
    isFullScreen.value = !isFullScreen.value;
  };

  // Handle ESC key to exit full screen
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape" && isFullScreen.value) {
      isFullScreen.value = false;
    }
  };

  // Add/remove event listener for ESC key
  onMounted(() => {
    document.addEventListener("keydown", handleKeyDown);
  });

  onUnmounted(() => {
    document.removeEventListener("keydown", handleKeyDown);
  });

  // Auto-enter fullscreen after a period of inactivity in focus mode
  const inactivityTimeoutMs = 30000; // 30 seconds
  let inactivityTimer: number | null = null;
  const scheduleInactivity = () => {
    if (inactivityTimer !== null) {
      clearTimeout(inactivityTimer);
      inactivityTimer = null;
    }
    if (!isFullScreen.value) {
      inactivityTimer = window.setTimeout(() => {
        isFullScreen.value = true;
      }, inactivityTimeoutMs);
    }
  };
  const onUserActivity = () => {
    scheduleInactivity();
  };
  onMounted(() => {
    ["mousemove", "mousedown", "keydown", "touchstart", "pointermove"].forEach((ev) =>
      window.addEventListener(ev, onUserActivity, { passive: true })
    );
    scheduleInactivity();
  });
  onUnmounted(() => {
    ["mousemove", "mousedown", "keydown", "touchstart", "pointermove"].forEach((ev) =>
      window.removeEventListener(ev, onUserActivity as any)
    );
    if (inactivityTimer !== null) {
      clearTimeout(inactivityTimer);
      inactivityTimer = null;
    }
  });

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
    isFinal?: boolean;
  }

  interface HistoryEntry {
    id: string;
    channelId: string;
    channelLabel: string;
    timestamp: string | number;
    transcript: string;
    translations: Record<string, Translation>;
  }

  interface Props {
    channels: Channel[];
    recentHistory: HistoryEntry[];
    getLanguageFlag: (code: string) => string;
    getLanguageName: (code: string) => string;
    showFocusedView: boolean;
    toggleChannel: (channelId: string) => void;
    addSystemChannel: () => void;
  }

  defineEmits<{
    toggleFocus: [];
  }>();

  const props = defineProps<Props>();

  const activeChannelsWithContent = computed(() =>
    props.channels.filter(
      (c) =>
        c.isActive &&
        (c.liveTranscript || Object.keys(c.liveTranslations).length)
    )
  );

  // Get recent history with translations (limit to last 5 entries for Musixmatch-style view)
  const recentHistoryWithTranslations = computed(() =>
    props.recentHistory
      .filter((entry) => Object.keys(entry.translations).length > 0)
      .slice(0, 5)
  );

  import LyricsDisplay from "./LyricsDisplay.vue";
  import ParticipantChip from "./ParticipantChip.vue";

  // Utilities moved into ParticipantChip component

  const participants = computed(() =>
    props.channels
      .filter((c) => c.isActive)
      .map((c) => ({ id: c.id, label: c.label }))
  );

  // Choose a display language across the conversation
  const chooseDisplayLanguage = (): string | null => {
    const counts = new Map<string, number>();
    for (const ch of props.channels) {
      Object.keys(ch.liveTranslations || {}).forEach((code) => {
        counts.set(code, (counts.get(code) || 0) + 1);
      });
    }
    if (counts.size === 0) return "en"; // fallback
    let best: string | null = null;
    let bestCount = -1;
    counts.forEach((cnt, code) => {
      if (cnt > bestCount) {
        best = code;
        bestCount = cnt;
      }
    });
    return best;
  };

  type Segment = { id?: string; text: string; originalText?: string; speaker?: string; timestamp?: number | string };
  const unifiedFeed = computed(() => {
    const displayCode = chooseDisplayLanguage();
    const segs: Segment[] = [];

    // History first (already finalized), sorted by timestamp ascending
    const history = [...recentHistoryWithTranslations.value].sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    for (const entry of history) {
      let t: Translation | undefined = undefined;
      if (displayCode) {
        t = entry.translations?.[displayCode];
      }
      const text = (t?.primary || (t?.alternatives?.[0] ?? "") || entry.transcript || "").trim();
      if (!text) continue;
      const originalText = t ? (entry.transcript || "") : undefined;
      segs.push({
        id: `h-${entry.id}`,
        text,
        originalText,
        speaker: entry.channelLabel,
        timestamp: entry.timestamp,
      });
    }

    // Live segments (one per channel), placed at the end as the most recent
    for (const ch of props.channels) {
      let t: Translation | undefined = undefined;
      if (displayCode) {
        t = ch.liveTranslations?.[displayCode];
      }
      const live = t?.primary
        ? t.primary
        : Array.isArray(t?.alternatives) && t.alternatives.length
        ? t.alternatives[0]
        : "";
      const text = (live || ch.liveTranscript || "").trim();
      if (!text) continue;
      const originalText = live ? (ch.liveTranscript || "") : undefined;
      segs.push({ id: `l-${ch.id}`, text, originalText, speaker: ch.label, timestamp: Date.now() });
    }

    // Sort again by timestamp as live segments use now()
    segs.sort((a, b) => (new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime()));

    const anyActive = props.channels.some((c) => c.isActive && !c.isFinal);
    return {
      segments: segs,
      streaming: anyActive,
      isFinal: !anyActive,
    };
  });

  // Feed-level centering handled by each LyricsDisplay (stable-centering)

  // ...existing code...

  // Removed old transcription card helpers after embedding unified lyrics feed
</script>
<style scoped src="../focused-view.css"></style>
