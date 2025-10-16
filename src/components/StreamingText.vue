<template class="streaming-text">
  {{ displayedText }}
  <span class="stream-cursor" v-if="streaming && !isStreamingComplete"></span>
</template>

<script setup lang="ts">
  import { ref, watch, computed } from "vue";

  interface Props {
    text: string;
    streaming?: boolean;
    speed?: number; // milliseconds per character
    isFinal?: boolean; // Indicates when text should be finalized
  }

  const props = withDefaults(defineProps<Props>(), {
    streaming: false,
    speed: 25, // Very fast streaming - 3ms per character
    isFinal: false,
  });

  const displayedLength = ref(0);
  const streamingTimer = ref<number | null>(null);
  const isStreamingComplete = ref(false);

  const displayedText = computed(() => {
    if (!props.streaming) {
      return props.text;
    }
    return props.text.slice(0, displayedLength.value);
  });

  // Watch for text changes - only restart animation for new text or when isFinal
  watch(
    () => props.text,
    (newText, oldText) => {
      if (props.streaming && newText && newText !== oldText) {
        const shouldStartNewAnimation =
          // Text is completely new (not an extension)
          !oldText ||
          !newText.startsWith(oldText) ||
          // Initial load
          oldText === undefined;

        if (shouldStartNewAnimation) {
          // Start fresh animation
          displayedLength.value = 0;
          isStreamingComplete.value = false;

          if (streamingTimer.value) {
            clearInterval(streamingTimer.value);
          }

          streamingTimer.value = setInterval(() => {
            if (displayedLength.value < newText.length) {
              displayedLength.value++;
            } else {
              isStreamingComplete.value = props.isFinal;
              if (props.isFinal && streamingTimer.value) {
                clearInterval(streamingTimer.value);
                streamingTimer.value = null;
              }
            }
          }, props.speed);
        } else if (newText.length > displayedLength.value) {
          // Text is extending, continue existing animation from current position
          if (!streamingTimer.value) {
            isStreamingComplete.value = false;
            streamingTimer.value = setInterval(() => {
              if (displayedLength.value < newText.length) {
                displayedLength.value++;
              } else {
                isStreamingComplete.value = props.isFinal;
                if (props.isFinal && streamingTimer.value) {
                  clearInterval(streamingTimer.value);
                  streamingTimer.value = null;
                }
              }
            }, props.speed);
          }
        }
      } else if (!props.streaming) {
        // Show all text immediately when not streaming
        displayedLength.value = newText.length;
        isStreamingComplete.value = true;
      }
    },
    { immediate: true }
  );

  // Watch for isFinal changes to complete animation
  watch(
    () => props.isFinal,
    (isFinal) => {
      if (isFinal && displayedLength.value >= props.text.length) {
        isStreamingComplete.value = true;
        if (streamingTimer.value) {
          clearInterval(streamingTimer.value);
          streamingTimer.value = null;
        }
      }
    }
  );

  // Cleanup timer on unmount
  watch(
    () => props.streaming,
    (isStreaming) => {
      if (!isStreaming && streamingTimer.value) {
        clearInterval(streamingTimer.value);
        streamingTimer.value = null;
        displayedLength.value = props.text.length;
        isStreamingComplete.value = true;
      }
    }
  );
</script>

<style scoped>
  .streaming-text {
    position: relative;
    /* Allow streaming partials to wrap naturally inside their parent line.
     In some layouts the parent may constrain width — ensure the streaming
     span itself doesn't force a single unbreakable box. */
    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
    display: inline-block;
    text-align: inherit;
    width: 100%;
  }

  /* Removed unused char animations - using actual text appending now */

  .stream-cursor {
    display: inline-block;
    width: 2px;
    height: 1.2em; /* use em so it scales with font-size in full-screen */
    background: rgba(29, 185, 84, 0.9);
    margin-left: 1px;
    animation: stream-blink 0.8s infinite;
    vertical-align: middle;
    border-radius: 1px;
  }

  @keyframes stream-blink {
    0%,
    65% {
      opacity: 1;
    }
    66%,
    100% {
      opacity: 0.2;
    }
  }
</style>
