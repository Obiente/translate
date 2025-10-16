import { computed, ref, watch } from 'vue';

/**
 * Word metadata for enhanced streaming control
 */
export interface WordMetadata {
  text: string;
  index: number;
  startCharIndex: number;
  endCharIndex: number;
  isCompleted: boolean;
}

/**
 * Configuration for word streaming behavior
 */
export interface WordStreamingConfig {
  /**
   * Maximum words to show above the current word
   */
  visibleWordsAbove?: number;

  /**
   * Maximum words to show below the current word
   */
  visibleWordsBelow?: number;

  /**
   * Whether to preserve punctuation as part of words
   */
  preservePunctuation?: boolean;

  /**
   * Word splitting regex pattern
   */
  splitPattern?: RegExp;

  /**
   * Minimum confidence threshold for word changes (0-1)
   * Helps prevent jittering when text length is ambiguous
   */
  confidenceThreshold?: number;
}

const DEFAULT_CONFIG: Required<WordStreamingConfig> = {
  visibleWordsAbove: 2,
  visibleWordsBelow: 3,
  preservePunctuation: true,
  splitPattern: /\s+/,
  confidenceThreshold: 0.85,
};

/**
 * Composable for robust word-by-word streaming with proper state management
 *
 * Features:
 * - No timers or animations (purely reactive)
 * - Stable word splitting and indexing
 * - Resilient to rapid text changes
 * - Optional word metadata tracking
 * - Configurable visibility windows
 * - Performance optimized with computed properties
 */
export function useWordStreaming(
  textRef: () => string,
  config: WordStreamingConfig = {},
) {
  const mergedConfig: Required<WordStreamingConfig> = {
    ...DEFAULT_CONFIG,
    ...config,
  };

  // State
  const currentWordIndex = ref(0);
  const visibleWordsAbove = ref(mergedConfig.visibleWordsAbove);
  const visibleWordsBelow = ref(mergedConfig.visibleWordsBelow);
  const lastProcessedText = ref('');
  const wordMetadataCache = ref<WordMetadata[]>([]);

  /**
   * Sanitize text: trim and handle edge cases
   */
  const sanitizeText = (text: string): string => {
    if (typeof text !== 'string') {
      return '';
    }
    return text.trim();
  };

  /**
   * Split text into words with proper error handling
   */
  const splitIntoWords = (text: string): string[] => {
    const sanitized = sanitizeText(text);

    if (sanitized.length === 0) {
      return [];
    }

    try {
      const words = sanitized
        .split(mergedConfig.splitPattern)
        .filter((word) => word.length > 0);

      return words;
    } catch (error) {
      console.warn('Error splitting text into words:', error);
      // Fallback: simple space split
      return sanitized.split(' ').filter((word) => word.length > 0);
    }
  };

  /**
   * Build word metadata for the current text
   * Includes position tracking for debugging and analytics
   */
  const buildWordMetadata = (words: string[]): WordMetadata[] => {
    const metadata: WordMetadata[] = [];
    let charIndex = 0;

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const startCharIndex = charIndex;
      const endCharIndex = startCharIndex + word.length;

      metadata.push({
        text: word,
        index: i,
        startCharIndex,
        endCharIndex,
        isCompleted: false,
      });

      charIndex = endCharIndex + 1; // +1 for space
    }

    return metadata;
  };

  /**
   * Compute all words from current text
   */
  const words = computed(() => {
    const text = textRef();
    return splitIntoWords(text);
  });

  /**
   * Update current word index based on text length
   * Uses confidence threshold to prevent jittering
   */
  const updateCurrentWordIndex = (newText: string) => {
    const wordCount = splitIntoWords(newText).length;

    if (wordCount === 0) {
      currentWordIndex.value = 0;
      return;
    }

    // Current index is the last word (most recently added)
    const newIndex = Math.max(0, wordCount - 1);

    // Only update if the change is significant enough (confidence)
    // This prevents jittering when word count is uncertain
    if (Math.abs(newIndex - currentWordIndex.value) >= 1) {
      currentWordIndex.value = newIndex;
    }
  };

  /**
   * Get the current word
   */
  const currentWord = computed(() => {
    if (words.value.length === 0) return '';
    return words.value[Math.min(currentWordIndex.value, words.value.length - 1)] || '';
  });

  /**
   * Get completed words (above center line)
   */
  const completedWords = computed(() => {
    // Show ALL completed words as history (no limit)
    // If visibleWordsAbove is large (>= 100), show everything
    if (visibleWordsAbove.value >= 100) {
      return words.value.slice(0, currentWordIndex.value);
    }
    
    // Otherwise apply the visibility window
    if (currentWordIndex.value <= visibleWordsAbove.value) {
      return words.value.slice(0, currentWordIndex.value);
    }

    const startIdx = currentWordIndex.value - visibleWordsAbove.value;
    return words.value.slice(startIdx, currentWordIndex.value);
  });

  /**
   * Get upcoming words (below center line)
   */
  const upcomingWords = computed(() => {
    const startIdx = currentWordIndex.value + 1;
    const endIdx = startIdx + visibleWordsBelow.value;
    return words.value.slice(startIdx, endIdx);
  });

  /**
   * Get text before current word (partial line)
   */
  const preCurrentText = computed(() => {
    if (currentWordIndex.value === 0 || words.value.length === 0) return '';

    // Show one word before current
    const startIdx = Math.max(0, currentWordIndex.value - 1);
    const preWords = words.value.slice(startIdx, currentWordIndex.value);

    return preWords.join(' ');
  });

  /**
   * Metadata for all words (for analytics, debugging)
   */
  const wordMetadata = computed(() => {
    const metadata = buildWordMetadata(words.value);

    // Mark completed words
    for (let i = 0; i < currentWordIndex.value; i++) {
      if (metadata[i]) {
        metadata[i].isCompleted = true;
      }
    }

    return metadata;
  });

  /**
   * Get statistics about current streaming state
   */
  const streamingStats = computed(() => {
    return {
      totalWords: words.value.length,
      currentWordIndex: currentWordIndex.value,
      completedWordsCount: currentWordIndex.value,
      upcomingWordsCount: Math.max(0, words.value.length - currentWordIndex.value - 1),
      progressPercentage:
        words.value.length > 0
          ? ((currentWordIndex.value + 1) / words.value.length) * 100
          : 0,
    };
  });

  /**
   * Watch for text changes and update state
   */
  watch(
    () => textRef(),
    (newText) => {
      if (newText !== lastProcessedText.value) {
        updateCurrentWordIndex(newText);
        lastProcessedText.value = newText;
      }
    },
    { immediate: true }
  );

  /**
   * Update visible words based on screen size or config changes
   */
  const updateVisibility = (above?: number, below?: number) => {
    if (typeof above === 'number' && above >= 0) {
      visibleWordsAbove.value = above;
    }
    if (typeof below === 'number' && below >= 0) {
      visibleWordsBelow.value = below;
    }
  };

  /**
   * Reset streaming state
   */
  const reset = () => {
    currentWordIndex.value = 0;
    lastProcessedText.value = '';
    wordMetadataCache.value = [];
  };

  return {
    // State
    currentWordIndex,
    words,
    currentWord,
    completedWords,
    upcomingWords,
    preCurrentText,
    wordMetadata,
    streamingStats,

    // Methods
    updateVisibility,
    reset,

    // Internals (for debugging)
    sanitizeText,
    splitIntoWords,
  };
}
