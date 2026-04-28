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
      <div>
        <span class="eyebrow">Discord live captions</span>
        <h1>{{ latestLine || "Waiting for speech" }}</h1>
        <p>
          Room <code>{{ sessionId }}</code>
          <span v-if="speakerCount">· {{ speakerCount }} speakers connected</span>
        </p>
      </div>
    </section>

    <section class="discord-room__stage" aria-label="Live translated captions">
      <article
        v-for="speaker in speakerCards"
        :key="speaker.id"
        class="speaker-lane"
        :class="{ active: speaker.isActive }"
      >
        <header>
          <span class="speaker-avatar">{{ speaker.initials }}</span>
          <div>
            <strong>{{ speaker.label }}</strong>
            <small>{{ speaker.language || "auto" }}</small>
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
        <span class="pulse-ring" aria-hidden="true"></span>
        <p>Open Discord and start talking in the connected voice channel.</p>
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
import { computed, onMounted } from "vue";
import { useRoomManager, type RoomTranscript } from "../composables/useRoomManager";

const props = defineProps<{
  sessionId: string;
}>();

const room = useRoomManager();
const ACTIVE_WINDOW_MS = 5000;

onMounted(async () => {
  await room.joinRoom(props.sessionId, undefined, "Caption viewer");
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
        language: entry.language,
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
  const raw = entry.peerLabel || entry.peerId || entry.channelId || "Speaker";
  const mention = raw.match(/^<@!?(\d+)>$/);
  if (mention) return `Discord ${mention[1].slice(-4)}`;
  return raw;
}

function initialsFor(label: string): string {
  const parts = label.replace(/[^a-zA-Z0-9\s]/g, " ").trim().split(/\s+/);
  return (parts[0]?.[0] || "S").toUpperCase() + (parts[1]?.[0] || "").toUpperCase();
}
</script>
