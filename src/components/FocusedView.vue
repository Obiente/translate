<template>
  <div class="focused-view" :class="{ 'full-screen': isFullScreen }">
    <!-- Header with controls - Hidden in full screen -->
    <header class="focused-header" v-if="!isFullScreen">
      <h1 class="focused-title">Obiente Translate</h1>
      <div class="channel-bar">
        <div class="speaking-badges" v-if="activeChannelsWithContent.length">
          <ParticipantChip
            v-for="channel in activeChannelsWithContent"
            :key="`badge-${channel.id}`"
            :label="channel.label"
          />
        </div>
        <!-- Always show room participants (avatars) when in a room, even if no one is actively speaking -->
        <div class="participants-bar" v-if="participants.length">
          <ParticipantChip
            v-for="p in participants"
            :key="p.id"
            :label="p.label + (p.sourceType === 'room' ? ' · room' : '')"
          />
        </div>
        <div class="channel-chips">
          <ParticipantChip
            v-for="channel in props.channels"
            :key="channel.id"
            :label="channel.label"
            :title="`${channel.isActive ? 'Pause' : 'Start'} ${channel.label}`"
            :icon="channel.isActive ? '⏸' : '▶️'"
            :isActive="channel.isActive"
            :interactive="true"
            @click="props.toggleChannel(channel.id)"
          />
        </div>
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

    <!-- Subtle watermark in fullscreen -->
    <div class="brand-watermark" v-if="isFullScreen">
      Obiente Translate • Obiente Cloud
    </div>

    <!-- Content Area - Bottom Origin Layout -->
    <div class="focused-content">
      <!-- When not in full-screen, keep separate sections for history and live -->
      <template v-if="!isFullScreen">
        <div
          class="embedded-lyrics"
          v-if="unifiedFeed && unifiedFeed.segments.length"
        >
          <LyricsDisplay
            :segments="unifiedFeed.segments"
            :streaming="unifiedFeed.streaming"
            :isFinal="unifiedFeed.isFinal"
            :speed="30"
            :centerOnStable="true"
            :alwaysShowInitials="room.isInRoom()"
            :speakingChannels="activeChannelsWithContent"
          />
        </div>
      </template>

      <!-- Full screen: one unified Spotify-style interleaved feed (history + live) -->
      <template v-else>
        <div class="full-screen-participants-bar" v-if="participants.length">
          <ParticipantChip
            v-for="p in participants"
            :key="p.id"
            :label="p.label + (p.sourceType === 'room' ? ' · room' : '')"
          />
        </div>
        <div
          class="full-screen-feed"
          v-if="unifiedFeed && unifiedFeed.segments.length"
        >
          <LyricsDisplay
            :segments="unifiedFeed.segments"
            :streaming="unifiedFeed.streaming"
            :isFinal="unifiedFeed.isFinal"
            :speed="30"
            :centerOnStable="true"
            :alwaysShowInitials="room.isInRoom()"
            :speakingChannels="activeChannelsWithContent"
          />
        </div>
      </template>

      <!-- Empty State -->
      <div
        class="focused-empty"
        v-if="
          !activeChannelsWithContent.length &&
          !recentHistoryWithTranslations.length &&
          !participants.length
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
  import { useRoomManager } from "../composables/useRoomManager";

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
    ["mousemove", "mousedown", "keydown", "touchstart", "pointermove"].forEach(
      (ev) => window.addEventListener(ev, onUserActivity, { passive: true })
    );
    scheduleInactivity();
  });
  onUnmounted(() => {
    ["mousemove", "mousedown", "keydown", "touchstart", "pointermove"].forEach(
      (ev) => window.removeEventListener(ev, onUserActivity as any)
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
    // Used to choose which translation to display per channel
    targetLanguages?: string[];
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
      (c: Channel) =>
        c.isActive &&
        (c.liveTranscript || Object.keys(c.liveTranslations).length)
    )
  );

  // Get recent history with translations — keep a generous window so users
  // do not lose context during longer conversations.
  const recentHistoryWithTranslations = computed(() =>
    props.recentHistory
      .filter((entry: HistoryEntry) => Object.keys(entry.translations).length > 0)
      .slice(0, 20)
  );

  import LyricsDisplay from "./LyricsDisplay.vue";
  import ParticipantChip from "./ParticipantChip.vue";

  // Utilities moved into ParticipantChip component

  // Merge active channel speakers with room roster so avatars/names show even when mic is off.
  const room = useRoomManager();
  const participants = computed(() => {
    const map = new Map<string, { id: string; label: string; sourceType: 'channel' | 'room' }>();
    const hasLocalActive = props.channels.some((c: Channel) => c.isActive && (c as any).sourceType !== 'room');
    // Helper: does a channel exist with given id (covers synthetic room channels too)
    const hasChannelId = (id: string) => props.channels.some((c: Channel) => c.id === id && c.isActive);

    // Active channels (include both microphone/system and synthetic 'room' channels)
    for (const c of props.channels) {
      if (c.isActive) {
        const st = (c as any).sourceType === 'room' ? 'room' : 'channel';
        map.set(`ch:${c.id}`, { id: `ch:${c.id}`, label: c.label, sourceType: st });
      }
    }
    // Room roster
    if (room.isInRoom()) {
      for (const m of room.members) {
        const pid = m.peerId || 'unknown';
        const label = m.peerLabel || 'Guest';
        // Skip if this peer is already represented by an active channel (synthetic room channel uses peerId as channel id)
        if (pid && hasChannelId(pid)) continue;
        // If this is self and we have any local active channel, prefer the local channel representation
        const isSelf = !!room.peerId.value && pid === room.peerId.value;
        if (isSelf && hasLocalActive) continue;
        if (!map.has(`room:${pid}`)) map.set(`room:${pid}`, { id: `room:${pid}`, label, sourceType: 'room' });
      }
    }
    return Array.from(map.values());
  });

  // No global display language — pick per-channel translation and never fall back to original

  type Segment = {
    id?: string;
    text: string;
    originalText?: string;
    speaker?: string;
    timestamp?: number | string;
  };
  const unifiedFeed = computed(() => {
    const segs: Segment[] = [];

    // Build quick lookup for channels by id
    const channelById = new Map<string, Channel>(props.channels.map((c: Channel) => [c.id, c]));

    // History first (already finalized), sorted by timestamp ascending
    const history = [...recentHistoryWithTranslations.value].sort(
      (a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    for (const entry of history) {
      const ch = channelById.get(entry.channelId);
      const preferred = Array.isArray(ch?.targetLanguages)
        ? ch!.targetLanguages!
        : [];
      const translations = entry.translations || {};
      let chosen: Translation | undefined;
      for (const code of preferred) {
        if (translations[code]) { chosen = translations[code]; break; }
      }
      if (!chosen) {
        const firstCode = Object.keys(translations)[0];
        if (firstCode) chosen = translations[firstCode];
      }
      if (!chosen) continue; // No server translation; do not show original
      const text = (chosen.primary || chosen.alternatives?.[0] || "").trim();
      if (!text) continue;
      segs.push({
        id: `h-${entry.id}`,
        text,
        originalText: undefined,
        speaker: entry.channelLabel,
        timestamp: entry.timestamp,
      });
    }

    // Collect already-seen translated texts from history to deduplicate live
    const historyTexts = new Set(segs.map((s) => s.text));

    // Live segments (one per channel), placed at the end as the most recent
    for (const ch of props.channels) {
      const preferred = Array.isArray(ch.targetLanguages) ? ch.targetLanguages : [];
      const translations = ch.liveTranslations || {};
      let chosen: Translation | undefined;
      for (const code of preferred) {
        if (translations[code]) { chosen = translations[code]; break; }
      }
      if (!chosen) {
        const firstCode = Object.keys(translations)[0];
        if (firstCode) chosen = translations[firstCode];
      }
      if (!chosen) continue; // No server translation yet — do not show original/liveTranscript
      const text = (chosen.primary || chosen.alternatives?.[0] || "").trim();
      if (!text) continue;
      // Skip live segments whose translation already appears in history —
      // this prevents the brief duplicate flash during live→history transition.
      if (historyTexts.has(text)) continue;
      segs.push({
        id: `l-${ch.id}`,
        text,
        originalText: undefined,
        speaker: ch.label,
        timestamp: Date.now(),
      });
    }

    // Sort again by timestamp as live segments use now()
    segs.sort(
      (a, b) =>
        new Date(a.timestamp || 0).getTime() -
        new Date(b.timestamp || 0).getTime()
    );

    const anyActive = props.channels.some((c: Channel) => c.isActive && !c.isFinal);
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
