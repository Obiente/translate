<template>
  <div class="spotify-lyrics-display" ref="containerRef">
    <!-- Original text overlay (fixed, not inside scroll) shows current sentence's original -->
    <div
      v-if="currentOriginalText"
      class="original-overlay"
      aria-live="polite"
      :style="overlayTopStyle"
    >
      <div class="original-text">
        <span class="original-label">Original:</span>
        {{ currentOriginalText }}
      </div>
    </div>

    <!-- Speaker indicator if provided (component-level speaker; hidden when using per-sentence speakers) -->
    <div
      v-if="props.speaker && !hasPerSentenceSpeakers"
      class="speaker-indicator"
    >
      <div
        v-if="showInitials"
        class="speaker-avatar"
        :style="{ backgroundColor: getSpeakerColor(props.speaker) }"
      >
        {{ getSpeakerInitials(props.speaker) }}
      </div>
      <span class="speaker-name">{{ props.speaker }}</span>
    </div>

    <!-- All sentences with typing animation -->
    <div
      v-for="(sentence, idx) in allSentences"
      :key="`sentence-${idx}`"
      :class="[
        'lyrics-line',
        {
          completed:
            idx < allSentences.length - 1 &&
            sentence.displayed === sentence.full,
          current:
            idx === allSentences.length - 1 ||
            sentence.displayed !== sentence.full,
        },
      ]"
    >
      <!-- Speaker inline avatar (prefer sentence-level speaker; fallback to component-level) -->
      <div
        v-if="showInitials && (sentence.speaker || props.speaker) && sentence.displayed"
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
  </div>
</template>

<script setup lang="ts">
  import { ref, computed, watch, nextTick, onUnmounted } from "vue";

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
    overlayTop?: string | number; // Optional override for original overlay top
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
    overlayTop: undefined,
  });

  const containerRef = ref<HTMLElement | null>(null);

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
    allSentences.value.some((s) => !!s.speaker)
  );

  // Only show initials if there is more than one unique speaker in the recent window
  // Focus on visible/active context: last few sentences that have some displayed text
  const showInitials = computed(() => {
    let recent = allSentences.value.filter((s) => (s.displayed?.length ?? 0) > 0);
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
  const overlayTopStyle = computed(() => {
    if (props.overlayTop === undefined || props.overlayTop === null)
      return {} as Record<string, string>;
    const val =
      typeof props.overlayTop === "number"
        ? `${props.overlayTop}px`
        : String(props.overlayTop);
    return { top: val } as Record<string, string>;
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
      segs.forEach((seg, segIndex) => {
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
    const oldById = new Map(allSentences.value.map((s) => [s.id, s] as const));

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
        // New sentence starts hidden (will animate in order)
        nm.displayed = "";
        nm.isAnimating = false;
      }
    }

    // Initial load: reveal history, animate last
    if (!hasInitialized.value) {
      hasInitialized.value = true;
      for (let i = 0; i < newModels.length; i++) {
        newModels[i].displayed =
          i < newModels.length - 1 ? newModels[i].full : "";
        newModels[i].isAnimating = i === newModels.length - 1;
      }
    }

    allSentences.value = newModels;
  };

  // Animate typing for incomplete sentences
  const animateTyping = () => {
    if (allSentences.value.length === 0) return;

    // Find the first sentence that is currently animating or needs animation
    let foundAnimating = false;

    for (let i = 0; i < allSentences.value.length; i++) {
      const sentence = allSentences.value[i];

      // If we found a sentence that's animating or incomplete
      if (sentence.isAnimating && sentence.displayed !== sentence.full) {
        // Continue animating this sentence
        const nextLength = Math.min(
          sentence.displayed.length + 1,
          sentence.full.length
        );
        sentence.displayed = sentence.full.substring(0, nextLength);

        // If this sentence is done animating
        if (sentence.displayed === sentence.full) {
          sentence.isAnimating = false;

          // Start animating the next sentence if it exists and is not complete
          if (i + 1 < allSentences.value.length) {
            const nextSentence = allSentences.value[i + 1];
            if (nextSentence.displayed !== nextSentence.full) {
              nextSentence.isAnimating = true;
            }
          }
        }

        foundAnimating = true;
        break;
      }
    }

    // If no sentence is currently animating, start the first incomplete one
    if (!foundAnimating) {
      for (let i = 0; i < allSentences.value.length; i++) {
        const sentence = allSentences.value[i];
        if (sentence.displayed !== sentence.full) {
          sentence.isAnimating = true;

          // Animate one character
          const nextLength = Math.min(
            sentence.displayed.length + 1,
            sentence.full.length
          );
          sentence.displayed = sentence.full.substring(0, nextLength);

          // If immediately complete, mark as not animating
          if (sentence.displayed === sentence.full) {
            sentence.isAnimating = false;
          }
          break;
        }
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
    ([streaming, isFinal]) => {
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
  });

  // Auto-scroll current sentence to keep it centered
  watch(
    () => allSentences.value.length,
    async () => {
      await nextTick();
      // Always scroll to the last sentence (the current one)
      if (allSentences.value.length > 0 && containerRef.value) {
        const allLines = containerRef.value.querySelectorAll(".lyrics-line");
        const lastLine = allLines[allLines.length - 1];
        if (lastLine) {
          lastLine.scrollIntoView({
            behavior: "smooth",
            block: "center",
            inline: "center",
          });
        }
      }
    }
  );

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
    padding: 50vh 1rem;
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

  /* Fixed overlay for original text (outside inner scroll) */
  .original-overlay {
    position: fixed;
    top: clamp(12px, 6vh, 72px);
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: min(900px, 90vw);
    z-index: 2500;
    pointer-events: none; /* do not block scroll/drag */
    padding: 0 1rem;
  }

  /* Original text - small and dimmed */
  .original-text {
    width: 100%;
    max-width: 90%;
    margin: 0 auto;
    font-size: clamp(0.75rem, 1.8vw, 0.9rem);
    line-height: 1.4;
    font-weight: 400;
    color: #aaa;
    text-align: center;
    opacity: 0.9;
    font-style: italic;
    padding: 0.75rem 1rem;
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

  /* Speaker indicator */
  .speaker-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 2rem;
    opacity: 0.8;
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
      padding: 50vh 0.75rem;
      max-width: 100%;
      gap: 1.25rem;
    }

    .lyrics-line {
      line-height: 1.4;
    }

    .original-overlay {
      top: clamp(8px, 5vh, 56px);
      padding: 0 0.75rem;
    }

    .original-text {
      font-size: clamp(0.7rem, 2.5vw, 0.85rem);
      padding: 0.5rem 0.75rem;
    }

    .speaker-indicator {
      margin-bottom: 1.5rem;
    }

    .speaker-avatar {
      width: 32px;
      height: 32px;
      font-size: 0.75rem;
    }
  }
</style>
