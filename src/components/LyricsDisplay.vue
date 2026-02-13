<template>
  <div
    class="spotify-lyrics-display"
    ref="containerRef"
    :style="containerStyle"
  >
    <!-- All sentences with typing animation -->
    <div
      v-for="(sentence, idx) in allSentences"
      :key="`sentence-${idx}`"
      :class="[
        'lyrics-line',
        {
          completed:
            Number(idx) < allSentences.length - 1 &&
            sentence.displayed === sentence.full,
          current:
            Number(idx) === allSentences.length - 1 ||
            sentence.displayed !== sentence.full,
        },
      ]"
    >
      <!-- Speaker inline avatar (prefer sentence-level speaker; fallback to component-level) -->
      <div
        v-if="
          showInitials &&
          (sentence.speaker || props.speaker) &&
          sentence.displayed
        "
        class="speaker-inline"
      >
        <div
          class="speaker-avatar inline"
          :style="{
            backgroundColor: getSpeakerColor(sentence.speaker || props.speaker),
          }"
        >
          {{ getSpeakerInitials(sentence.speaker || props.speaker) }}
        </div>
      </div>
      <span class="sentence-text">{{ sentence.displayed }}</span>
      <!-- Typing cursor on sentences that are animating -->
      <span
        v-if="sentence.isAnimating && isStreaming && isFinal === false"
        class="typing-cursor"
      ></span>
    </div>
    <div v-if="props.speaker && !hasPerSentenceSpeakers" class="speaker-chip">
      <div
        v-if="showInitials"
        class="speaker-avatar"
        :style="{ backgroundColor: getSpeakerColor(props.speaker) }"
      >
        {{ getSpeakerInitials(props.speaker) }}
      </div>
      <span class="speaker-name">{{ props.speaker }}</span>
    </div>

    <!-- Bottom section with speaker chip and original text (fixed at bottom) -->
  <div class="bottom-info" ref="bottomInfoRef">
      <div
        class="full-screen-participants-bar"
        v-if="props.speakingChannels.length"
      >
        <ParticipantChip
          v-for="p in speakingChannels"
          :key="p.id"
          :label="p.label"
        />
      </div>

      <!-- Speaker chip if provided (component-level speaker; hidden when using per-sentence speakers) -->
      <div v-if="props.speaker && !hasPerSentenceSpeakers" class="speaker-chip">
        <div
          v-if="showInitials"
          class="speaker-avatar"
          :style="{ backgroundColor: getSpeakerColor(props.speaker) }"
        >
          {{ getSpeakerInitials(props.speaker) }}
        </div>
        <span class="speaker-name">{{ props.speaker }}</span>
      </div>

      <!-- Original text -->
      <div
        v-if="currentOriginalText"
        class="original-overlay"
        aria-live="polite"
      >
        <div class="original-text">
          <span class="original-label">Original:</span>
          {{ currentOriginalText }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
  import {
    ref,
    computed,
    watch,
    nextTick,
    onUnmounted,
    onMounted,
  } from "vue";
  import ParticipantChip from "./ParticipantChip.vue";

  interface SegmentInput {
    id?: string;
    text: string;
    originalText?: string;
    speaker?: string;
    timestamp?: number | string;
  }

  interface Props {
    text?: string;
    originalText?: string; // legacy single overlay; replaced by current sentence original
    speaker?: string;
    streaming?: boolean;
    speed?: number;
    isFinal?: boolean;
    centerOnStable?: boolean;
    segments?: SegmentInput[]; // New API: interleaved segments with per-sentence metadata
    speakingChannels?: Array<{ id: string; label: string }>; // Currently speaking participants
    // When true, always show speaker initials/avatar regardless of unique speaker count
    alwaysShowInitials?: boolean;
    // Top padding as viewport height percentage to bias for more history visibility
    topPadVh?: number;
    // Auto-scroll only when near bottom; set false to disable auto pinning
    autoscroll?: boolean;
    // Distance from bottom in pixels considered "pinned"
    pinThresholdPx?: number;
  }

  const props = withDefaults(defineProps<Props>(), {
    text: "",
    originalText: "",
    speaker: "",
    streaming: true,
    speed: 30, // milliseconds per character
    isFinal: false,
    centerOnStable: false,
    segments: () => [] as SegmentInput[],
    speakingChannels: () => [] as Array<{ id: string; label: string }>,
    alwaysShowInitials: false,
    topPadVh: 35, // show more history than the previous 50vh default
    autoscroll: true,
    pinThresholdPx: 120,
  });

  const containerRef = ref<HTMLElement | null>(null);
  const bottomInfoRef = ref<HTMLElement | null>(null);
  const bottomInfoHeight = ref(200); // fallback padding for bottom overlay
  const isPinned = ref(true); // whether user is near the bottom (auto-scroll enabled)
  const hasUserScrolled = ref(false);
  let isAutoScrolling = false; // flag to ignore scroll events triggered by programmatic scrolls

  // Store sentences with their displayed text
  interface SentenceDisplay {
    id: string; // stable id for diffing
    full: string;
    displayed: string;
    isAnimating: boolean; // Track if this sentence is currently animating
    speaker?: string;
    original?: string;
  }

  // Helper to get initials from speaker name
  const getSpeakerInitials = (name: string): string => {
    if (!name) return "";
    const words = name.trim().split(/\s+/);
    if (words.length === 1) {
      return words[0].substring(0, 2).toUpperCase();
    }
    return words
      .slice(0, 2)
      .map((w) => w[0])
      .join("")
      .toUpperCase();
  };

  // Generate color from speaker name
  const getSpeakerColor = (name: string): string => {
    if (!name) return "#646cff";
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
      hash = name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    return `hsl(${hue}, 65%, 55%)`;
  };

  const allSentences = ref<SentenceDisplay[]>([]);
  const typingInterval = ref<number | null>(null);
  const hasInitialized = ref(false); // Track if we've done the initial load
  const hasPerSentenceSpeakers = computed(() =>
    allSentences.value.some((s: SentenceDisplay) => !!s.speaker)
  );

  // Dynamic container padding to avoid bottom overlay overlap and allow longer history
  const containerStyle = computed(() => {
    const topPad = `${props.topPadVh}vh`;
    const sidePad = `1rem`;
    const bottomPad = `${Math.max(bottomInfoHeight.value + 32, 160)}px`;
    return {
      padding: `${topPad} ${sidePad} ${bottomPad} ${sidePad}`,
    } as Record<string, string>;
  });

  // Only show initials if there is more than one unique speaker in the recent window
  // Focus on visible/active context: last few sentences that have some displayed text
  const showInitials = computed(() => {
    if (props.alwaysShowInitials) return true;
    let recent = allSentences.value.filter(
      (s: SentenceDisplay) => (s.displayed?.length ?? 0) > 0
    );
    if (recent.length === 0 && allSentences.value.length > 0) {
      // Nothing displayed yet: consider the latest sentence (current line)
      recent = [allSentences.value[allSentences.value.length - 1]];
    }
    recent = recent.slice(-8);
    const names = new Set<string>();
    for (const s of recent) {
      const n = (s.speaker || props.speaker || "").trim();
      if (n) names.add(n);
      if (names.size > 1) return true;
    }
    return names.size > 1;
  });

  // Split text into sentences (split on . ! ? followed by space or end of string)
  // Also split long sentences without punctuation based on max word count
  const splitIntoSentences = (
    text: string,
    maxWords: number = 15
  ): string[] => {
    if (!text.trim()) return [];

    // Split by sentence-ending punctuation followed by space or end
    // This regex splits on . ! ? and keeps the punctuation with the sentence
    const sentences: string[] = [];
    let currentSentence = "";

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      currentSentence += char;

      // Check if this is a sentence-ending punctuation
      if (char === "." || char === "!" || char === "?") {
        // Look ahead to see if next char is space or end of string
        const nextChar = text[i + 1];
        if (!nextChar || nextChar === " " || nextChar === "\n") {
          // This is a sentence ending
          sentences.push(currentSentence.trim());
          currentSentence = "";
        }
      }
    }

    // Add any remaining text as incomplete sentence
    if (currentSentence.trim()) {
      sentences.push(currentSentence.trim());
    }

    // Now split long sentences without punctuation
    const finalSentences: string[] = [];

    for (const sentence of sentences) {
      // Check if sentence has punctuation ending
      const hasPunctuation = /[.!?]$/.test(sentence);

      if (!hasPunctuation) {
        // Split by word count if too long
        const words = sentence.split(/\s+/);

        if (words.length > maxWords) {
          // Split into chunks of maxWords
          for (let i = 0; i < words.length; i += maxWords) {
            const chunk = words.slice(i, i + maxWords).join(" ");
            if (chunk.trim()) {
              finalSentences.push(chunk.trim());
            }
          }
        } else {
          finalSentences.push(sentence);
        }
      } else {
        finalSentences.push(sentence);
      }
    }

    return finalSentences.filter((s) => s.length > 0);
  };

  // Check if currently streaming
  const isStreaming = computed(() => props.streaming && !props.isFinal);

  // Process text and update sentences with typing animation
  const buildSentenceUnits = (): Array<{
    id: string;
    text: string;
    speaker?: string;
    original?: string;
  }> => {
    const units: Array<{
      id: string;
      text: string;
      speaker?: string;
      original?: string;
    }> = [];
    const segs = Array.isArray(props.segments) ? props.segments : [];
    if (segs.length > 0) {
      segs.forEach((seg: SegmentInput, segIndex: number) => {
        const baseId = seg.id || `seg-${segIndex}`;
        const parts = splitIntoSentences(seg.text || "");
        if (parts.length === 0 && (seg.text || "").trim().length) {
          units.push({
            id: `${baseId}-0`,
            text: seg.text,
            speaker: seg.speaker,
            original: seg.originalText,
          });
        } else {
          parts.forEach((p, i) => {
            units.push({
              id: `${baseId}-${i}`,
              text: p,
              speaker: seg.speaker,
              original: seg.originalText,
            });
          });
        }
      });
    } else {
      const text = (props.text || "").trim();
      if (!text) return units;
      const parts = splitIntoSentences(text);
      if (parts.length === 0) {
        units.push({
          id: `single-0`,
          text,
          speaker: props.speaker,
          original: props.originalText,
        });
      } else {
        parts.forEach((p, i) =>
          units.push({
            id: `single-${i}`,
            text: p,
            speaker: props.speaker,
            original: props.originalText,
          })
        );
      }
    }
    return units;
  };

  const updateSentences = () => {
    const units = buildSentenceUnits();

    if (units.length === 0) {
      allSentences.value = [];
      hasInitialized.value = false;
      return;
    }

    // Build new sentence models
    const newModels: SentenceDisplay[] = units.map((u) => ({
      id: u.id,
      full: u.text,
      displayed: "",
      isAnimating: false,
      speaker: u.speaker,
      original: u.original,
    }));

    // Diff with existing by id to preserve progress
    const oldById = new Map<string, SentenceDisplay>(
      allSentences.value.map((s: SentenceDisplay) => [s.id, s] as [string, SentenceDisplay])
    );

    for (let i = 0; i < newModels.length; i++) {
      const nm = newModels[i];
      const prev = oldById.get(nm.id);
      if (prev) {
        // Carry over displayed text if still a prefix of new full
        if (prev.full === nm.full) {
          nm.displayed = prev.displayed;
          nm.isAnimating = prev.isAnimating;
        } else {
          // Content changed
          const wasComplete = prev.displayed === prev.full;
          if (wasComplete) {
            nm.displayed = nm.full; // keep completed lines completed
            nm.isAnimating = false;
          } else {
            nm.displayed = ""; // restart animation for current line
            nm.isAnimating = true;
          }
        }
      } else {
        // No ID match — check if an existing sentence has matching text.
        // This handles live→history transitions where the segment ID changes
        // (e.g. "l-channel" → "h-uuid") but the text content is identical,
        // preventing a visible flash/jump when the sentence re-animates.
        let textMatch: SentenceDisplay | undefined;
        for (const old of allSentences.value) {
          if (old.full === nm.full && old.displayed === old.full) {
            textMatch = old;
            break;
          }
        }
        if (textMatch) {
          nm.displayed = nm.full;
          nm.isAnimating = false;
        } else {
          // Truly new sentence — starts hidden (will animate in order)
          nm.displayed = "";
          nm.isAnimating = false;
        }
      }
    }

    // Always instantly reveal every sentence except the very last one.
    // Only the last (current) sentence gets the typing animation.
    // This prevents a visual backlog where many queued sentences must
    // animate sequentially before the user can see the latest text.
    for (let i = 0; i < newModels.length - 1; i++) {
      if (newModels[i].displayed !== newModels[i].full) {
        newModels[i].displayed = newModels[i].full;
        newModels[i].isAnimating = false;
      }
    }

    // Kick off animation for the last sentence if it isn't complete yet
    if (newModels.length > 0) {
      const last = newModels[newModels.length - 1];
      if (last.displayed !== last.full && !last.isAnimating) {
        last.isAnimating = true;
      }
    }

    if (!hasInitialized.value) {
      hasInitialized.value = true;
    }

    allSentences.value = newModels;
  };

  // Animate typing for the current (last) sentence.
  // Uses adaptive speed: large text gaps catch up quickly so completed
  // translations appear almost instantly, while small incremental arrivals
  // keep the smooth typewriter feel.
  let lastAnimScrollTime = 0;

  // Lightweight scroll nudge during animation — uses instant scrollTop
  // assignment so it never conflicts with ongoing smooth scrolls.
  const nudgeScroll = () => {
    if (!props.autoscroll || !isPinned.value) return;
    const now = Date.now();
    if (now - lastAnimScrollTime < 200) return; // throttle to ~5 Hz
    lastAnimScrollTime = now;
    const el = containerRef.value;
    if (!el) return;
    const allLines = el.querySelectorAll(".lyrics-line");
    const lastLine = allLines[allLines.length - 1] as HTMLElement | undefined;
    if (!lastLine) return;
    isAutoScrolling = true;
    lastLine.scrollIntoView({ behavior: "auto", block: "center" });
    requestAnimationFrame(() => {
      isAutoScrolling = false;
    });
  };

  const animateTyping = () => {
    if (allSentences.value.length === 0) return;

    // Only the last sentence should ever be animating (earlier ones are
    // force-revealed in updateSentences), but scan defensively.
    for (let i = 0; i < allSentences.value.length; i++) {
      const sentence = allSentences.value[i];

      if (sentence.isAnimating && sentence.displayed !== sentence.full) {
        const gap = sentence.full.length - sentence.displayed.length;
        // Adaptive: large gap → reveal in ~8 ticks (~240 ms); small gap → 1 char
        const charsPerTick = gap > 30 ? Math.ceil(gap / 8) : 1;
        const nextLength = Math.min(
          sentence.displayed.length + charsPerTick,
          sentence.full.length
        );
        sentence.displayed = sentence.full.substring(0, nextLength);

        if (sentence.displayed === sentence.full) {
          sentence.isAnimating = false;
        }
        nudgeScroll();
        return;
      }
    }

    // No sentence currently animating — start the first incomplete one
    for (let i = 0; i < allSentences.value.length; i++) {
      const sentence = allSentences.value[i];
      if (sentence.displayed !== sentence.full) {
        sentence.isAnimating = true;

        const gap = sentence.full.length - sentence.displayed.length;
        const charsPerTick = gap > 30 ? Math.ceil(gap / 8) : 1;
        const nextLength = Math.min(
          sentence.displayed.length + charsPerTick,
          sentence.full.length
        );
        sentence.displayed = sentence.full.substring(0, nextLength);

        if (sentence.displayed === sentence.full) {
          sentence.isAnimating = false;
        }
        nudgeScroll();
        return;
      }
    }
  };

  // Watch for text or segments changes
  watch(
    () => [props.text, props.segments] as const,
    () => {
      updateSentences();
    },
    { immediate: true, deep: true }
  );

  // Start typing animation interval
  if (props.streaming && !props.isFinal) {
    typingInterval.value = window.setInterval(() => {
      animateTyping();
    }, props.speed);
  }

  // Watch for streaming state changes
  watch(
    () => [props.streaming, props.isFinal] as const,
    ([streaming, isFinal]: [boolean, boolean]) => {
      // Clear existing interval
      if (typingInterval.value !== null) {
        clearInterval(typingInterval.value);
        typingInterval.value = null;
      }

      if (streaming && !isFinal) {
        // Start typing animation
        typingInterval.value = window.setInterval(() => {
          animateTyping();
        }, props.speed);
      } else {
        // Show all text immediately
        for (const sentence of allSentences.value) {
          sentence.displayed = sentence.full;
        }
      }
    }
  );

  // Clean up interval on unmount
  onUnmounted(() => {
    if (typingInterval.value !== null) {
      clearInterval(typingInterval.value);
      typingInterval.value = null;
    }
    if (isAutoScrollTimer) {
      clearTimeout(isAutoScrollTimer);
      isAutoScrollTimer = null;
    }
    if (resizeObs) {
      resizeObs.disconnect();
      resizeObs = null;
    }
    // detach scroll listener
    if (containerRef.value && onScrollBound) {
      containerRef.value.removeEventListener("scroll", onScrollBound);
      onScrollBound = null;
    }
  });

  // Auto-scroll current sentence to keep it centered
  let isAutoScrollTimer: ReturnType<typeof setTimeout> | null = null;
  const scrollToCurrent = (behavior: ScrollBehavior = "smooth") => {
    if (!containerRef.value || allSentences.value.length === 0) return;
    const allLines = containerRef.value.querySelectorAll(".lyrics-line");
    const lastLine = allLines[allLines.length - 1] as HTMLElement | undefined;
    if (!lastLine) return;
    // Flag so the onScroll handler ignores scroll events from this call.
    // We also force isPinned = true because we are explicitly scrolling to
    // the latest content — prevents a stale false from blocking future scrolls.
    isAutoScrolling = true;
    isPinned.value = true;
    lastLine.scrollIntoView({ behavior, block: "center", inline: "center" });
    // Smooth scrolling takes 300-500 ms — keep the guard up long enough.
    if (isAutoScrollTimer) clearTimeout(isAutoScrollTimer);
    const clearDelay = behavior === "smooth" ? 500 : 60;
    isAutoScrollTimer = setTimeout(() => {
      isAutoScrolling = false;
      isAutoScrollTimer = null;
    }, clearDelay);
  };

  watch(
    () => allSentences.value.length,
    async (newLen: number, oldLen: number | undefined) => {
      await nextTick();
      // Only auto-scroll when pinned or on first initialize
      if (!props.autoscroll) return;
      if (!hasInitialized.value || isPinned.value) {
        scrollToCurrent(newLen - (oldLen ?? 0) > 1 ? "auto" : "smooth");
      }
    }
  );

  // Also auto-scroll when segment content changes (not just sentence count).
  // This covers cases where the live segment text grows without adding new
  // sentences, e.g. words appended to the last line.
  watch(
    () => {
      const segs = Array.isArray(props.segments) ? props.segments : [];
      return segs.map((s: SegmentInput) => s.text).join("|");
    },
    async () => {
      if (!props.autoscroll || !isPinned.value) return;
      await nextTick();
      scrollToCurrent("smooth");
    }
  );

  // Track whether user is near the bottom to control auto-scroll
  let onScrollBound: ((e: Event) => void) | null = null;
  const onScroll = () => {
    // Skip scroll events triggered by programmatic scrollToCurrent calls
    if (isAutoScrolling) return;
    const el = containerRef.value;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    const pinned = distanceFromBottom <= (props.pinThresholdPx ?? 120);
    hasUserScrolled.value = true;
    isPinned.value = pinned;
  };

  // Observe bottom overlay height to set bottom padding dynamically
  let resizeObs: ResizeObserver | null = null;
  onMounted(() => {
    // Attach scroll listener
    if (containerRef.value) {
      onScrollBound = onScroll.bind(null);
      containerRef.value.addEventListener("scroll", onScrollBound);
      // Initial pin state
      onScroll();
    }
    // Measure bottom overlay
    if (bottomInfoRef.value && "ResizeObserver" in window) {
      resizeObs = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const box = entry.contentRect;
          bottomInfoHeight.value = Math.round(box.height);
        }
      });
      resizeObs.observe(bottomInfoRef.value);
      // Initial measure
      bottomInfoHeight.value = Math.round(
        bottomInfoRef.value.getBoundingClientRect().height
      );
    }
    // Ensure current line is visible on first mount
    nextTick(() => scrollToCurrent("auto"));
  });

  // Current sentence's original text for overlay
  const currentOriginalText = computed(() => {
    // Prefer the most recent sentence that has any displayed characters and an original
    for (let i = allSentences.value.length - 1; i >= 0; i--) {
      const s = allSentences.value[i];
      if (s.original && (s.displayed?.length ?? 0) > 0) return s.original;
    }
    // If nothing displayed yet on the newest, but it has an original, show that
    if (allSentences.value.length) {
      const last = allSentences.value[allSentences.value.length - 1];
      if (last.original) return last.original;
    }
    // Fallback to legacy prop if provided and no per-sentence original
    return props.originalText || "";
  });
</script>

<style scoped>
  .spotify-lyrics-display {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    width: 100%;
    max-width: 900px;
    min-height: 100%;
    margin: 0 auto;
    padding: 50vh 1rem 200px 1rem;
    gap: 1.5rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      sans-serif;
    overflow-y: auto;
    overflow-x: hidden;
    /* Hide scrollbar */
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none; /* IE and Edge */
  }

  /* Hide scrollbar for Chrome, Safari and Opera */
  .spotify-lyrics-display::-webkit-scrollbar {
    display: none;
    width: 0;
    height: 0;
  }

  /* Bottom info section - fixed at bottom with speaker chip above original text */
  .bottom-info {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: min(900px, 90vw);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 1.5rem;
    z-index: 1000;
    pointer-events: auto;
  }

  /* Original text - positioned at bottom */
  .original-overlay {
    width: 100%;
    max-width: 100%;
    padding: 0;
  }

  /* Original text - small and dimmed */
  .original-text {
    width: 100%;
    font-size: clamp(0.75rem, 1.8vw, 0.9rem);
    line-height: 1.4;
    font-weight: 400;
    color: #aaa;
    text-align: center;
    opacity: 0.9;
    font-style: italic;
    padding: 0.5rem 1rem;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
  }

  .original-label {
    font-weight: 600;
    color: #888;
    font-style: normal;
    text-transform: uppercase;
    font-size: 0.7em;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
  }

  /* Speaker chip - above original text */
  .speaker-chip {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    opacity: 0.9;
  }

  /* Speaking badges - participants currently speaking */
  .speaking-badges {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .badge-placeholder {
    font-size: clamp(0.85rem, 2vw, 0.95rem);
    font-weight: 600;
    color: #ddd;
    padding: 0.4rem 0.8rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }

  .speaker-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .speaker-name {
    font-size: clamp(0.9rem, 2.2vw, 1.1rem);
    font-weight: 600;
    color: #ccc;
  }

  /* Each line (sentence) */
  .lyrics-line {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    line-height: 1.5;
    font-weight: 600;
    word-wrap: break-word;
    overflow-wrap: break-word;
    transition: all 0.3s ease;
    text-align: center;
  }

  /* Completed sentences - dimmed */
  .lyrics-line.completed {
    color: #666;
    font-weight: 500;
    opacity: 0.7;
  }

  /* Current sentence - highlighted like Spotify */
  .lyrics-line.current {
    color: #fff;
    font-size: clamp(1.5rem, 4vw, 2.5rem);
    font-weight: 700;
    position: relative;
  }

  .speaker-inline {
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .speaker-avatar.inline {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 800;
    color: #fff;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
  }

  .sentence-text {
    display: inline;
  }
  /* Participants fixed bar at top of fullscreen */
  .focused-view.full-screen .full-screen-participants-bar {
    top: clamp(8px, 4vh, 56px);
    z-index: 3000;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    max-width: 90vw;
    padding: 0.35rem 0.5rem;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }
  .typing-cursor {
    display: inline-block;
    width: 3px;
    height: 1em;
    background: #1db954;
    animation: blink 1s step-end infinite;
    border-radius: 2px;
    margin-left: 0.2rem;
  }

  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .spotify-lyrics-display {
      padding: 50vh 0.75rem 180px 0.75rem;
      max-width: 100%;
      gap: 1.25rem;
    }

    .lyrics-line {
      line-height: 1.4;
    }

    .bottom-info {
      max-width: 100vw;
      padding: 1rem 0.75rem;
    }

    .original-text {
      font-size: clamp(0.7rem, 2.5vw, 0.85rem);
      padding: 0.4rem 0.5rem;
    }

    .speaker-avatar {
      width: 32px;
      height: 32px;
      font-size: 0.75rem;
    }
  }
</style>
