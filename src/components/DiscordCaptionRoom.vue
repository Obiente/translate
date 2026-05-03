<template>
  <main class="discord-room">
    <section class="discord-room__top">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">Obiente Translate</span>
      </div>
      <div class="discord-room__status" :class="room.status.value">
        <span class="live-dot" aria-hidden="true"></span>
        <span>{{ statusLabel }}</span>
      </div>
    </section>

    <section class="discord-room__hero">
      <div class="discord-room__hero-copy">
        <span class="eyebrow">Discord live captions</span>
        <h1>{{ heroTitle }}</h1>
        <p class="discord-room__hero-summary">
          {{ heroSummary }}
        </p>
        <div class="discord-room__meta">
          <p>
            <span class="meta-label">Room</span>
            <code>{{ sessionId }}</code>
          </p>
          <p>
            <span class="meta-label">Listeners</span>
            <strong>{{ speakerCount || 0 }}</strong>
          </p>
        </div>
      </div>
      <aside v-if="sessionStatus" class="discord-room__session">
        <span>Bot session</span>
        <strong :class="sessionStatus.status">{{ botStatusLabel }}</strong>
        <small>{{ sessionStatus.status_message || sessionStatus.live_url }}</small>
      </aside>
    </section>

    <section class="discord-room__stage" aria-label="Live translated captions">
      <article
        v-for="speaker in speakerCards"
        :key="speaker.id"
        class="speaker-lane"
        :class="{ active: speaker.isActive }"
      >
        <header>
          <img
            v-if="speaker.avatarUrl"
            class="speaker-avatar speaker-avatar--image"
            :src="speaker.avatarUrl"
            :alt="speaker.label"
          />
          <span v-else class="speaker-avatar">{{ speaker.initials }}</span>
          <div>
            <strong>{{ speaker.label }}</strong>
            <small>{{ speaker.meta }}</small>
          </div>
        </header>
        <p class="speaker-translation">
          {{ speaker.translation || speaker.transcript || "Listening..." }}
        </p>
        <p class="speaker-source" v-if="speaker.translation && speaker.transcript">
          {{ speaker.transcript }}
        </p>
      </article>

      <div class="discord-room__empty" v-if="!speakerCards.length">
        <div class="discord-room__empty-orb">
          <span class="pulse-ring" aria-hidden="true"></span>
        </div>
        <div class="discord-room__empty-copy">
          <strong>Live room is ready</strong>
          <p>{{ emptyStateMessage }}</p>
        </div>
      </div>
    </section>

    <section class="discord-room__feed">
      <header>
        <h2>Timeline</h2>
        <span>{{ timeline.length }} lines</span>
      </header>
      <ol v-if="timeline.length">
        <li v-for="item in timeline" :key="item.key">
          <span>{{ item.time }}</span>
          <strong>{{ item.label }}</strong>
          <p>{{ item.translation || item.transcript }}</p>
        </li>
      </ol>
      <p v-else class="placeholder">No captions have arrived yet.</p>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRoomManager, type RoomTranscript } from "../composables/useRoomManager";

const props = defineProps<{
  sessionId: string;
}>();

const room = useRoomManager();
const ACTIVE_WINDOW_MS = 5000;
const SESSION_POLL_MS = 5000;
const MPEGV_SERVE_BASE = ((import.meta.env.VITE_MPEGV_SERVE_URL as string | undefined)?.trim() || "")
  .replace(/\/$/, "");

interface TranslationSessionStatus {
  id: string;
  live_url?: string;
  status: "starting" | "connected" | "ready" | "degraded" | "stopped" | "failed" | "stale";
  status_message?: string;
  date_updated?: string;
}

const sessionStatus = ref<TranslationSessionStatus | null>(null);
const sessionPollError = ref("");
let sessionPollTimer: number | null = null;

onMounted(async () => {
  await room.joinRoom(props.sessionId, undefined, "Caption viewer");
  await pollSessionStatus();
  if (MPEGV_SERVE_BASE) {
    sessionPollTimer = window.setInterval(pollSessionStatus, SESSION_POLL_MS);
  }
});

onUnmounted(() => {
  if (sessionPollTimer !== null) {
    clearInterval(sessionPollTimer);
    sessionPollTimer = null;
  }
});

const statusLabel = computed(() => {
  if (room.status.value === "connected") return "Live";
  if (room.status.value === "connecting") return "Connecting";
  if (room.status.value === "error") return "Reconnecting";
  return "Idle";
});

const speakerCount = computed(() =>
  room.members.filter((member) => member.peerLabel !== "Caption viewer").length,
);

const botStatusLabel = computed(() => {
  const status = sessionStatus.value?.status;
  if (status === "ready") return "Receiving audio";
  if (status === "connected") return "Voice connected";
  if (status === "degraded") return "Voice degraded";
  if (status === "stale") return "No heartbeat";
  if (status === "failed") return "Failed";
  if (status === "stopped") return "Stopped";
  return "Starting";
});

const emptyStateMessage = computed(() => {
  const status = sessionStatus.value?.status;
  if (status === "degraded") {
    return sessionStatus.value?.status_message ||
      "The bot joined Discord, but the voice connection is not ready yet.";
  }
  if (status === "stale") {
    return "The bot has not reported a heartbeat recently. Try restarting the voice translation session.";
  }
  if (status === "failed") {
    return sessionStatus.value?.status_message || "The bot could not start this voice translation session.";
  }
  if (status === "stopped") return "This voice translation session has ended.";
  if (sessionPollError.value) return sessionPollError.value;
  return "Open Discord and start talking in the connected voice channel.";
});

const captions = computed(() =>
  room.transcripts
    .filter((entry) => entry.roomId === props.sessionId)
    .slice(-80),
);

const latestLine = computed(() => {
  const latest = [...captions.value].reverse().find((entry) =>
    bestTranslation(entry) || entry.fullText || entry.text
  );
  if (!latest) return "";
  return bestTranslation(latest) || latest.fullText || latest.text || "";
});

const heroTitle = computed(() => {
  if (!speakerCards.value.length) return "Live room ready";
  const activeSpeaker = speakerCards.value[0];
  return activeSpeaker.label;
});

const heroSummary = computed(() => {
  if (!speakerCards.value.length) return emptyStateMessage.value;
  return latestLine.value || "Captions will appear here as soon as speech is detected.";
});

const speakerCards = computed(() => {
  const bySpeaker = new Map<string, RoomTranscript>();
  for (const entry of captions.value) {
    bySpeaker.set(speakerKey(entry), entry);
  }

  return [...bySpeaker.values()]
    .map((entry) => {
      const label = speakerLabel(entry);
      const timestamp = Date.parse(entry.timestamp);
      return {
        id: speakerKey(entry),
        label,
        initials: initialsFor(label),
        avatarUrl: entry.peerAvatarUrl || memberAvatarFor(entry),
        language: entry.language,
        meta: [entry.language || "auto", entry.isFinal ? "final" : "live"].join(" - "),
        transcript: entry.fullText || entry.text || "",
        translation: bestTranslation(entry),
        isActive: Number.isFinite(timestamp) && Date.now() - timestamp < ACTIVE_WINDOW_MS,
        timestamp,
      };
    })
    .sort((a, b) => b.timestamp - a.timestamp);
});

const timeline = computed(() =>
  [...captions.value]
    .reverse()
    .filter((entry) => entry.isFinal && (bestTranslation(entry) || entry.fullText || entry.text))
    .slice(0, 40)
    .map((entry, index) => ({
      key: `${entry.peerId || entry.channelId || "speaker"}-${entry.sequence ?? index}-${entry.timestamp}`,
      label: speakerLabel(entry),
      transcript: entry.fullText || entry.text || "",
      translation: bestTranslation(entry),
      time: new Date(entry.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    })),
);

function bestTranslation(entry: RoomTranscript): string {
  if (!entry.translations) return "";
  const first = Object.values(entry.translations).find((value) => {
    if (typeof value === "string") return value.trim();
    if (value && typeof value === "object" && "primary" in value) {
      return String((value as { primary?: unknown }).primary ?? "").trim();
    }
    return "";
  });
  if (typeof first === "string") return first.trim();
  if (first && typeof first === "object" && "primary" in first) {
    return String((first as { primary?: unknown }).primary ?? "").trim();
  }
  return "";
}

function speakerKey(entry: RoomTranscript): string {
  return entry.channelId || entry.peerId || "speaker";
}

function speakerLabel(entry: RoomTranscript): string {
  const raw = (entry.peerLabel || memberLabelFor(entry) || entry.peerId || entry.channelId || "Speaker").trim();
  const mention = raw.match(/^<@!?(\d+)>$/);
  if (mention) return `Discord ${mention[1].slice(-4)}`;
  return raw;
}

function memberLabelFor(entry: RoomTranscript): string {
  const member = room.members.find((item) =>
    (item.channelId && entry.channelId && item.channelId === entry.channelId) ||
    (item.peerId && entry.peerId && item.peerId === entry.peerId)
  );
  return member?.peerLabel || "";
}

function memberAvatarFor(entry: RoomTranscript): string {
  const member = room.members.find((item) =>
    (item.channelId && entry.channelId && item.channelId === entry.channelId) ||
    (item.peerId && entry.peerId && item.peerId === entry.peerId)
  );
  return member?.peerAvatarUrl || "";
}

function initialsFor(label: string): string {
  const parts = label.replace(/[^a-zA-Z0-9\s]/g, " ").trim().split(/\s+/);
  return (parts[0]?.[0] || "S").toUpperCase() + (parts[1]?.[0] || "").toUpperCase();
}

async function pollSessionStatus(): Promise<void> {
  if (!MPEGV_SERVE_BASE) return;
  try {
    const response = await fetch(
      `${MPEGV_SERVE_BASE}/translation-sessions/${encodeURIComponent(props.sessionId)}`,
      {
        headers: { Accept: "application/json" },
        cache: "no-store",
      },
    );
    if (!response.ok) {
      sessionPollError.value = response.status === 404
        ? "The bot has not registered this session yet."
        : "Could not read the bot session status.";
      return;
    }
    sessionStatus.value = await response.json() as TranslationSessionStatus;
    sessionPollError.value = "";
  } catch {
    sessionPollError.value = "Could not reach the bot session service.";
  }
}
</script>
