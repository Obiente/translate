import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { DEFAULT_LANGUAGES } from "../constants/languages";
import {
  CHANNEL_STORAGE_KEY,
  HISTORY_LIMIT,
  HISTORY_STORAGE_KEY,
  SELECTED_CHANNEL_KEY,
  SHOW_TRANSLATION_ALTERNATIVES_KEY,
  TRANSLATION_ALTERNATIVE_LIMIT_KEY,
  TRANSCRIPTION_MODE_KEY,
} from "../constants/storage";
import { useAudioCapture } from "../composables/useAudioCapture";
import { useSpeechSynthesis } from "../composables/useSpeechSynthesis";
import { useTranslationService } from "../composables/useTranslationService";
import { useMicrophoneManager } from "../composables/liveTranslation/useMicrophoneManager";
import { useWhisperTranscriber } from "./liveTranslation/useWhisperTranscriber";
import type { WhisperTranscriberManager } from "./liveTranslation/useWhisperTranscriber";
import { useStreamingWhisperTranscriber } from "./liveTranslation/useStreamingWhisperTranscriber";
import type {
  ConversationChannel,
  ConversationHistoryEntry,
  LanguageOption,
  StatusMessage,
  StatusType,
  TranslationEntry,
} from "../types/conversation";
import { normalizeTranslationMap } from "../utils/translation";

interface FinalTranscriptPayload {
  fullText?: string;
  deltaText: string;
  isFinal: boolean;
  translations?: Record<string, TranslationEntry>;
  detectedLanguage?: string | null;
}

const DEFAULT_WHISPER_ENDPOINT = "https://whisper.obiente.cloud/transcribe";
const WHISPER_ENDPOINT =
  (import.meta.env.VITE_WHISPER_ENDPOINT as string | undefined)?.trim() ||
  DEFAULT_WHISPER_ENDPOINT;
const DEFAULT_WHISPER_STREAMING_ENDPOINT =
  "wss://whisper.obiente.cloud/ws/transcribe";
const WHISPER_STREAMING_ENDPOINT =
  (import.meta.env.VITE_WHISPER_STREAMING_ENDPOINT as string | undefined)
    ?.trim() ||
  DEFAULT_WHISPER_STREAMING_ENDPOINT;

type TranscriptionMode = "chunked" | "streaming";

const createId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 10);

const createChannel = (
  sourceType: ConversationChannel["sourceType"],
  label: string,
  sourceLanguage: string,
  targetLanguages: string[],
  autoSpeak = true,
  id?: string,
  microphoneDeviceId?: string | null,
  microphoneDeviceLabel?: string | null,
): ConversationChannel => ({
  id: id ?? createId(),
  label,
  sourceType,
  sourceLanguage,
  targetLanguages,
  liveTranscript: "",
  liveTranslations: {},
  lastFinalTranscript: "",
  translations: {},
  detectedLanguage: null,
  isActive: false,
  isTranslating: false,
  status: null,
  autoSpeak,
  microphoneDeviceId: microphoneDeviceId ?? null,
  microphoneDeviceLabel: microphoneDeviceLabel ?? null,
  microphoneStream: null,
});

export const useLiveTranslation = () => {
  const languages = ref<LanguageOption[]>(DEFAULT_LANGUAGES);
  const channels = ref<ConversationChannel[]>([]);
  const selectedChannelId = ref<string | null>(null);
  const conversationHistory = ref<ConversationHistoryEntry[]>([]);
  const globalStatus = ref<StatusMessage | null>(null);
  const showTranslationAlternatives = ref<boolean>(false);
  const clampAlternativeLimit = (value: number): number =>
    Math.max(0, Math.min(5, Number.isFinite(value) ? Math.round(value) : 0));
  const translationAlternativesLimit = ref<number>(clampAlternativeLimit(2));
  const resolveAlternativeLimit = (): number =>
    clampAlternativeLimit(
      showTranslationAlternatives.value ? translationAlternativesLimit.value : 0,
    );
  const transcriptionMode = ref<TranscriptionMode>("streaming");
  const speechSupported = ref<boolean>(
    typeof window !== "undefined" &&
      typeof MediaRecorder !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia,
  );

  const LIVE_TRANSLATION_DEBOUNCE_MS = 450;
  const liveTranslationTimers = new Map<string, ReturnType<typeof setTimeout>>();
  const liveTranslationRequests = new Map<string, symbol>();

  const sanitizeTranslationEntry = (
    entry: TranslationEntry | undefined | null,
  ): TranslationEntry | null => {
    if (!entry) {
      return null;
    }

    let primary = typeof entry.primary === "string"
      ? entry.primary.trim()
      : "";
    let alternatives = Array.isArray(entry.alternatives)
      ? Array.from(
        new Set(
          entry.alternatives
            .map((option) =>
              typeof option === "string" ? option.trim() : "",
            )
            .filter((option) => option.length > 0),
        ),
      )
      : [];

    alternatives = alternatives.filter((option) => option !== primary);

    if (!primary && alternatives.length > 0) {
      primary = alternatives[0];
      alternatives = alternatives.slice(1);
    }

    if (!primary && alternatives.length === 0) {
      return null;
    }

    return { primary, alternatives };
  };

  const filterTranslationEntries = (
    translations: Record<string, TranslationEntry> | undefined,
    targetLanguages: string[],
  ): Record<string, TranslationEntry> => {
    if (!translations || typeof translations !== "object") {
      return {};
    }

    const whitelist = new Set(
      targetLanguages.map((code) => code.trim()).filter((code) => code.length > 0),
    );
    const allowAll = whitelist.size === 0;

    return Object.entries(translations).reduce<Record<string, TranslationEntry>>(
      (acc, [code, entry]) => {
        if (!allowAll && !whitelist.has(code)) {
          return acc;
        }

        const sanitized = sanitizeTranslationEntry(entry);
        if (sanitized) {
          acc[code] = sanitized;
        }
        return acc;
      },
      {},
    );
  };

  const { translateText, getSupportedLanguages } = useTranslationService();
  const {
    ensureMicrophonePermission,
    startDesktopCapture,
    resetDesktopCapture,
    enumerateAudioInputs,
    support,
  } = useAudioCapture();
  const { speak, isSpeaking } = useSpeechSynthesis();

  const microphoneManager = useMicrophoneManager({
    channels,
    enumerateAudioInputs,
  });

  const setChannelStatus = (
    channel: ConversationChannel,
    message: string,
    type: StatusType,
  ) => {
    channel.status = { message, type };
  };
  const resolveTranscriptionStream = async (
    channel: ConversationChannel,
  ): Promise<MediaStream> => {
    if (channel.sourceType === "microphone") {
      await microphoneManager.ensureStream(channel);
      if (!channel.microphoneStream) {
        throw new Error("Microphone stream unavailable");
      }
      return channel.microphoneStream;
    }

    if (channel.sourceType === "system") {
      if (!channel.systemStream) {
        throw new Error("System audio stream unavailable");
      }
      return channel.systemStream;
    }

    throw new Error("Unsupported channel type");
  };

  const chunkedTranscriptionManager = useWhisperTranscriber({
    ensureStream: resolveTranscriptionStream,
    updateStatus: setChannelStatus,
    endpoint: WHISPER_ENDPOINT,
    getTranslationAlternativeLimit: resolveAlternativeLimit,
    onTranscription: async (
      channel,
      { text, language, fullText, isFinal, translations },
    ) => {
      if (language) {
        channel.detectedLanguage = language;
      }
      if (fullText && fullText.length > 0) {
        channel.liveTranscript = fullText;
      } else if (text.length > 0) {
        channel.liveTranscript = channel.liveTranscript
          ? `${channel.liveTranscript} ${text}`.trim()
          : text;
      }
      const trimmedFullText = fullText?.trim() ?? "";
      const primaryPreview = trimmedFullText.length > 0
        ? trimmedFullText
        : channel.liveTranscript;
      const normalizedTranslations = translations ?? {};
      const filteredServerTranslations = filterTranslationEntries(
        normalizedTranslations,
        channel.targetLanguages,
      );
      const hasServerTranslations =
        Object.keys(filteredServerTranslations).length > 0;

      if (isFinal) {
        clearLiveTranslationSchedule(channel.id);
        channel.liveTranslations = {};
        if (!hasServerTranslations) {
          resetLiveTranslations(channel);
        }
      } else if (hasServerTranslations) {
        clearLiveTranslationSchedule(channel.id);
        channel.liveTranslations = filteredServerTranslations;
      } else if (primaryPreview) {
        scheduleLiveTranslation(channel, primaryPreview);
      } else {
        resetLiveTranslations(channel);
      }

      await handleFinalTranscript(channel, {
        fullText,
        deltaText: text,
        isFinal: isFinal ?? true,
        translations: filteredServerTranslations,
        detectedLanguage: language ?? null,
      });
    },
  });

  const streamingTranscriptionManager = useStreamingWhisperTranscriber({
    ensureStream: resolveTranscriptionStream,
    updateStatus: setChannelStatus,
    endpoint: WHISPER_STREAMING_ENDPOINT,
    getTranslationAlternativeLimit: resolveAlternativeLimit,
    onTranscription: async (
      channel,
      { text, language, fullText, isFinal, translations },
    ) => {
      if (language) {
        channel.detectedLanguage = language;
      }
      if (fullText && fullText.length > 0) {
        channel.liveTranscript = fullText;
      } else if (text.length > 0) {
        channel.liveTranscript = channel.liveTranscript
          ? `${channel.liveTranscript} ${text}`.trim()
          : text;
      }

      const trimmedFullText = fullText?.trim() ?? "";
      const primaryPreview = trimmedFullText.length > 0
        ? trimmedFullText
        : channel.liveTranscript;
      const normalizedTranslations = translations ?? {};
      const filteredServerTranslations = filterTranslationEntries(
        normalizedTranslations,
        channel.targetLanguages,
      );
      const hasServerTranslations =
        Object.keys(filteredServerTranslations).length > 0;

      if (isFinal) {
        clearLiveTranslationSchedule(channel.id);
        channel.liveTranslations = {};
        if (!hasServerTranslations) {
          resetLiveTranslations(channel);
        }
      } else if (hasServerTranslations) {
        clearLiveTranslationSchedule(channel.id);
        channel.liveTranslations = filteredServerTranslations;
      } else if (primaryPreview) {
        scheduleLiveTranslation(channel, primaryPreview);
      } else {
        resetLiveTranslations(channel);
      }

      await handleFinalTranscript(channel, {
        fullText,
        deltaText: text,
        isFinal: isFinal ?? false,
        translations: filteredServerTranslations,
        detectedLanguage: language ?? null,
      });
    },
  });

  const transcriptionManagers: Record<
    TranscriptionMode,
    WhisperTranscriberManager
  > = {
    chunked: chunkedTranscriptionManager,
    streaming: streamingTranscriptionManager,
  };

  const activeTranscriptionModes = new Map<string, TranscriptionMode>();

  let statusTimeout: number | undefined;
  let channelPersistenceReady = false;
  let deviceChangeCleanup: (() => void) | null = null;

  const hasSystemChannel = computed(() =>
    channels.value.some((channel) => channel.sourceType === "system")
  );
  const desktopCaptureSupported = computed(
    () => support.value.getDisplayMedia,
  );
  const recentHistory = computed(() =>
    [...conversationHistory.value].slice(-10).reverse()
  );
  const selectedChannel = computed(() =>
    channels.value.find((channel) => channel.id === selectedChannelId.value) ??
      null
  );
  const isAnyChannelActive = computed(() =>
    channels.value.some((channel) => channel.isActive)
  );

  const setGlobalStatus = (message: string, type: StatusType): void => {
    globalStatus.value = { message, type };
    if (statusTimeout) {
      window.clearTimeout(statusTimeout);
    }
    statusTimeout = window.setTimeout(() => {
      globalStatus.value = null;
    }, 4000);
  };

  const getLanguageName = (code: string): string => {
    const match = languages.value.find((lang) => lang.code === code);
    return match ? match.name : code;
  };

  const ensureTargets = (channel: ConversationChannel): void => {
    if (channel.targetLanguages.length === 0) {
      const fallback = languages.value.find(
        (lang) => lang.code !== channel.sourceLanguage,
      );
      if (fallback) {
        channel.targetLanguages = [fallback.code];
      }
    }

    channel.liveTranslations = filterTranslationEntries(
      channel.liveTranslations,
      channel.targetLanguages,
    );
    channel.translations = filterTranslationEntries(
      channel.translations,
      channel.targetLanguages,
    );
  };

  const clearLiveTranslationSchedule = (channelId: string): void => {
    const timer = liveTranslationTimers.get(channelId);
    if (timer !== undefined) {
      clearTimeout(timer);
      liveTranslationTimers.delete(channelId);
    }
    liveTranslationRequests.delete(channelId);
  };

  const resetLiveTranslations = (channel: ConversationChannel): void => {
    clearLiveTranslationSchedule(channel.id);
    if (Object.keys(channel.liveTranslations).length > 0) {
      channel.liveTranslations = {};
    }
  };

  const scheduleLiveTranslation = (
    channel: ConversationChannel,
    text: string,
  ): void => {
    const trimmed = text.trim();
    const channelId = channel.id;

    if (!trimmed || channel.targetLanguages.length === 0) {
      resetLiveTranslations(channel);
      return;
    }

    const alternativeLimit = resolveAlternativeLimit();

    const existingTimer = liveTranslationTimers.get(channelId);
    if (existingTimer !== undefined) {
      clearTimeout(existingTimer);
    }

    liveTranslationTimers.set(
      channelId,
      setTimeout(async () => {
        liveTranslationTimers.delete(channelId);
        const requestToken = Symbol(channelId);
        liveTranslationRequests.set(channelId, requestToken);

        try {
          ensureTargets(channel);
          if (channel.targetLanguages.length === 0) {
            resetLiveTranslations(channel);
            return;
          }

          const translationResults = await Promise.all(
            channel.targetLanguages.map(async (target) => {
              const result = await translateText(
                trimmed,
                channel.sourceLanguage === "auto"
                  ? "auto"
                  : channel.sourceLanguage,
                target,
                alternativeLimit,
              );
              return {
                target,
                entry: sanitizeTranslationEntry({
                  primary: result.translatedText,
                  alternatives: Array.isArray(result.alternatives)
                    ? result.alternatives
                    : [],
                }),
              };
            }),
          );

          if (liveTranslationRequests.get(channelId) !== requestToken) {
            return;
          }

          const liveUpdates: Record<string, TranslationEntry> = {};
          translationResults.forEach(({ target, entry }) => {
            if (entry) {
              liveUpdates[target] = entry;
            }
          });

          if (Object.keys(liveUpdates).length === 0) {
            resetLiveTranslations(channel);
            return;
          }

          channel.liveTranslations = filterTranslationEntries(
            {
              ...channel.liveTranslations,
              ...liveUpdates,
            },
            channel.targetLanguages,
          );
        } catch (error) {
          console.warn("Live translation update failed", error);
        } finally {
          if (liveTranslationRequests.get(channelId) === requestToken) {
            liveTranslationRequests.delete(channelId);
          }
        }
      }, LIVE_TRANSLATION_DEBOUNCE_MS),
    );
  };

  const ensureSelectedChannel = (): void => {
    if (channels.value.length === 0) {
      selectedChannelId.value = null;
      return;
    }
    const exists = channels.value.some(
      (channel) => channel.id === selectedChannelId.value,
    );
    if (!exists) {
      selectedChannelId.value = channels.value[0].id;
    }
  };

  const serializeChannels = () =>
    channels.value.map((channel) => ({
      id: channel.id,
      label: channel.label,
      sourceType: channel.sourceType,
      sourceLanguage: channel.sourceLanguage,
      targetLanguages: [...new Set(channel.targetLanguages)],
      autoSpeak: channel.autoSpeak,
      microphoneDeviceId: channel.microphoneDeviceId ?? null,
      microphoneDeviceLabel: channel.microphoneDeviceLabel ?? null,
    }));

  const persistChannels = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const payload = serializeChannels();
      localStorage.setItem(CHANNEL_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("Failed to persist channel preferences", error);
    }
  };

  const persistSelectedChannel = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      if (!selectedChannelId.value) {
        localStorage.removeItem(SELECTED_CHANNEL_KEY);
        return;
      }
      localStorage.setItem(SELECTED_CHANNEL_KEY, selectedChannelId.value);
    } catch (error) {
      console.warn("Failed to persist selected channel", error);
    }
  };

  const persistAlternativePreference = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      localStorage.setItem(
        SHOW_TRANSLATION_ALTERNATIVES_KEY,
        showTranslationAlternatives.value ? "true" : "false",
      );
    } catch (error) {
      console.warn(
        "Failed to persist translation alternative preference",
        error,
      );
    }
  };

  const persistAlternativeLimit = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const sanitized = clampAlternativeLimit(
        translationAlternativesLimit.value,
      );
      if (sanitized !== translationAlternativesLimit.value) {
        translationAlternativesLimit.value = sanitized;
        return;
      }
      localStorage.setItem(
        TRANSLATION_ALTERNATIVE_LIMIT_KEY,
        String(sanitized),
      );
    } catch (error) {
      console.warn("Failed to persist translation alternative limit", error);
    }
  };

  const persistTranscriptionMode = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      localStorage.setItem(TRANSCRIPTION_MODE_KEY, transcriptionMode.value);
    } catch (error) {
      console.warn("Failed to persist transcription mode preference", error);
    }
  };

  const restoreSelectedChannel = (): boolean => {
    if (typeof window === "undefined") {
      return false;
    }

    try {
      const stored = localStorage.getItem(SELECTED_CHANNEL_KEY);
      if (!stored) {
        return false;
      }
      selectedChannelId.value = stored;
      return true;
    } catch (error) {
      console.warn("Failed to restore selected channel", error);
      return false;
    }
  };

  const restoreAlternativePreference = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const raw = localStorage.getItem(SHOW_TRANSLATION_ALTERNATIVES_KEY);
      if (raw === null) {
        return;
      }
      showTranslationAlternatives.value = raw === "true";
    } catch (error) {
      console.warn(
        "Failed to restore translation alternative preference",
        error,
      );
    }
  };

  const restoreAlternativeLimit = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const raw = localStorage.getItem(TRANSLATION_ALTERNATIVE_LIMIT_KEY);
      if (raw === null) {
        return;
      }

      const parsed = Number.parseInt(raw, 10);
      translationAlternativesLimit.value = clampAlternativeLimit(
        Number.isFinite(parsed) ? parsed : 0,
      );
    } catch (error) {
      console.warn("Failed to restore translation alternative limit", error);
    }
  };

  const restoreTranscriptionMode = (): void => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const raw = localStorage.getItem(TRANSCRIPTION_MODE_KEY);
      if (!raw) {
        return;
      }

      if (raw === "chunked" || raw === "streaming") {
        transcriptionMode.value = raw;
      }
    } catch (error) {
      console.warn("Failed to restore transcription mode preference", error);
    }
  };

  const restoreChannels = (): boolean => {
    if (typeof window === "undefined") {
      return false;
    }

    try {
      const raw = localStorage.getItem(CHANNEL_STORAGE_KEY);
      if (!raw) {
        return false;
      }

      const parsed = JSON.parse(raw) as Array<
        ReturnType<typeof serializeChannels>[number]
      >;
      if (!Array.isArray(parsed) || parsed.length === 0) {
        return false;
      }

      const restored = parsed
        .map((item) => {
          if (!item || typeof item !== "object") {
            return null;
          }

          const targetLanguages = Array.isArray(item.targetLanguages)
            ? item.targetLanguages.filter(
              (code): code is string =>
                typeof code === "string" && code.trim().length > 0,
            )
            : [];

          const sourceType = item.sourceType === "system"
            ? "system"
            : "microphone";

          return createChannel(
            sourceType,
            item.label || "Participant",
            item.sourceLanguage || "auto",
            targetLanguages,
            item.autoSpeak ?? true,
            item.id,
            item.microphoneDeviceId ?? null,
            item.microphoneDeviceLabel ?? null,
          );
        })
        .filter((channel): channel is ConversationChannel => channel !== null);

      if (restored.length === 0) {
        return false;
      }

      channels.value = restored;
      return true;
    } catch (error) {
      console.warn("Failed to restore channel preferences", error);
      return false;
    }
  };

  const persistHistory = () => {
    try {
      localStorage.setItem(
        HISTORY_STORAGE_KEY,
        JSON.stringify(conversationHistory.value),
      );
    } catch (error) {
      console.warn("Failed to persist history", error);
    }
  };

  const restoreHistory = () => {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Array<
        Partial<ConversationHistoryEntry>
      >;
      if (!Array.isArray(parsed)) {
        conversationHistory.value = [];
        return;
      }

      const normalised = parsed
        .map((item) => {
          if (!item || typeof item !== "object") {
            return null;
          }

          const translations = normalizeTranslationMap(
            (item as { translations?: unknown }).translations,
          );

          return {
            id: item.id ?? createId(),
            channelId: item.channelId ?? "",
            channelLabel: item.channelLabel ?? "Participant",
            sourceLanguage: item.sourceLanguage ?? "auto",
            detectedLanguage: item.detectedLanguage ?? null,
            transcript: item.transcript ?? "",
            translations,
            timestamp: item.timestamp ?? new Date().toISOString(),
          } as ConversationHistoryEntry;
        })
        .filter((entry): entry is ConversationHistoryEntry => entry !== null);

      conversationHistory.value = normalised;
    } catch (error) {
      console.warn("Failed to restore history", error);
    }
  };

  const conversationHistoryPush = (entry: ConversationHistoryEntry): void => {
    conversationHistory.value.push({
      ...entry,
      translations: normalizeTranslationMap(entry.translations),
    });
    if (conversationHistory.value.length > HISTORY_LIMIT) {
      conversationHistory.value.splice(
        0,
        conversationHistory.value.length - HISTORY_LIMIT,
      );
    }
  };

  const formatTimestamp = (iso: string): string => {
    const time = new Date(iso);
    return time.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const selectChannel = (id: string): void => {
    selectedChannelId.value = id;
  };

  const addChannel = (channel: ConversationChannel): void => {
    channels.value.push(channel);
    selectedChannelId.value = channel.id;
  };

  const addMicrophoneChannel = (): void => {
    const defaultTarget = languages.value.find((lang) => lang.code !== "en");
    const newChannel = createChannel(
      "microphone",
      `Participant ${channels.value.length + 1}`,
      channels.value.length === 0 ? "en" : "auto",
      defaultTarget ? [defaultTarget.code] : ["es"],
    );
    const preferredDeviceId = microphoneManager.pickUnusedDeviceId();
    if (preferredDeviceId) {
      newChannel.microphoneDeviceId = preferredDeviceId;
      microphoneManager.handleSelectionChange(newChannel);
    }
    addChannel(newChannel);
  };

  const addSystemChannel = async (): Promise<void> => {
    if (hasSystemChannel.value) {
      setGlobalStatus(
        "Only one system audio capture is supported at a time.",
        "info",
      );
      return;
    }

    addChannel(
      createChannel(
        "system",
        "Shared System Audio",
        "auto",
        ["en", "es"],
        false,
      ),
    );
  };

  const removeChannel = async (id: string): Promise<void> => {
    const channel = channels.value.find((item) => item.id === id);
    if (!channel) return;
    microphoneManager.stopStream(channel);
    if (channel.isActive) {
      await pauseChannel(channel);
    }
    await Promise.all([
      chunkedTranscriptionManager.dispose(id),
      streamingTranscriptionManager.dispose(id),
    ]);
    activeTranscriptionModes.delete(id);
    channels.value = channels.value.filter((item) => item.id !== id);
    ensureSelectedChannel();
  };

  const handleFinalTranscript = async (
    channel: ConversationChannel,
    payload: FinalTranscriptPayload,
  ): Promise<void> => {
    if (!payload.isFinal) {
      return;
    }

    const previousFull = channel.lastFinalTranscript?.trim() ?? "";
    const fullUtterance = (payload.fullText ?? "").trim();
    const deltaUtterance = payload.deltaText.trim();
    const effectiveUtterance = fullUtterance || deltaUtterance;

    if (!effectiveUtterance) {
      return;
    }

    if (fullUtterance && fullUtterance === previousFull) {
      return;
    }

    ensureTargets(channel);
    channel.lastFinalTranscript = fullUtterance || effectiveUtterance;

    const filteredServerTranslations = filterTranslationEntries(
      payload.translations,
      channel.targetLanguages,
    );

    if (Object.keys(filteredServerTranslations).length > 0) {
      channel.translations = {
        ...channel.translations,
        ...filteredServerTranslations,
      };
      channel.detectedLanguage =
        payload.detectedLanguage ?? channel.detectedLanguage;
      channel.liveTranscript = "";
      channel.isTranslating = false;
      setChannelStatus(channel, "Translations updated", "success");

      if (channel.autoSpeak) {
        Object.entries(filteredServerTranslations)
          .filter(([, entry]) => entry.primary.length > 0)
          .forEach(([target, entry]) => speak(entry.primary, target));
      }

      conversationHistoryPush({
        id: createId(),
        channelId: channel.id,
        channelLabel: channel.label,
        sourceLanguage: channel.sourceLanguage,
        detectedLanguage: channel.detectedLanguage,
        transcript: channel.lastFinalTranscript,
        translations: filteredServerTranslations,
        timestamp: new Date().toISOString(),
      });
      return;
    }

    if (channel.targetLanguages.length === 0) {
      channel.isTranslating = false;
      setChannelStatus(channel, "No target languages selected", "info");
      return;
    }

    setChannelStatus(channel, "Translating…", "processing");
    channel.isTranslating = true;

    const translationSource = fullUtterance || deltaUtterance;
    const alternativeLimit = resolveAlternativeLimit();

    try {
      const translationResults = await Promise.all(
        channel.targetLanguages.map(async (target) => {
          const result = await translateText(
            translationSource,
            channel.sourceLanguage === "auto"
              ? "auto"
              : channel.sourceLanguage,
            target,
            alternativeLimit,
          );
          return {
            target,
            entry: sanitizeTranslationEntry({
              primary: result.translatedText,
              alternatives: Array.isArray(result.alternatives)
                ? result.alternatives
                : [],
            }),
            detectedLanguage: result.detectedLanguage,
          };
        }),
      );

      const translationMap: Record<string, TranslationEntry> = {};
      translationResults.forEach((entry) => {
        if (entry.entry) {
          translationMap[entry.target] = entry.entry;
        }
      });

      if (Object.keys(translationMap).length === 0) {
        setChannelStatus(
          channel,
          "No translations were generated for the latest speech segment.",
          "info",
        );
        channel.liveTranscript = "";
        channel.isTranslating = false;
        return;
      }

      channel.translations = {
        ...channel.translations,
        ...translationMap,
      };
      const detected = translationResults.find(
        (entry) => entry.detectedLanguage,
      )?.detectedLanguage;
      channel.detectedLanguage = detected ?? channel.detectedLanguage;
      setChannelStatus(channel, "Translations updated", "success");
      channel.liveTranscript = "";

      if (channel.autoSpeak) {
        Object.entries(translationMap)
          .filter(([, entry]) => entry.primary.length > 0)
          .forEach(([target, entry]) => speak(entry.primary, target));
      }

      conversationHistoryPush({
        id: createId(),
        channelId: channel.id,
        channelLabel: channel.label,
        sourceLanguage: channel.sourceLanguage,
        detectedLanguage: channel.detectedLanguage,
        transcript: channel.lastFinalTranscript,
        translations: translationMap,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Translation error", error);
      setChannelStatus(
        channel,
        "Failed to translate the latest speech segment.",
        "error",
      );
    } finally {
      channel.isTranslating = false;
    }
  };

  const startChannel = async (channel: ConversationChannel): Promise<void> => {
    if (!speechSupported.value) {
      setGlobalStatus(
        "Audio capture is not supported in this browser.",
        "error",
      );
      return;
    }

    try {
      if (channel.sourceType === "microphone") {
        const permission = await ensureMicrophonePermission();
        if (permission === "denied") {
          setChannelStatus(channel, "Microphone permission denied.", "error");
          return;
        }
        await microphoneManager.refresh();
        microphoneManager.handleSelectionChange(channel);
      } else {
        if (!channel.systemStream) {
          try {
            channel.systemStream = await startDesktopCapture();
          } catch (captureError) {
            const message = captureError instanceof Error
              ? captureError.message
              : "Desktop capture permission denied.";
            setChannelStatus(channel, message, "error");
            return;
          }
        }
      }

      ensureTargets(channel);
      channel.isActive = true;

      setChannelStatus(
        channel,
        channel.sourceType === "microphone"
          ? "Connecting to microphone…"
          : "Preparing system capture…",
        "processing",
      );

      const mode = transcriptionMode.value;
      activeTranscriptionModes.set(channel.id, mode);
      const manager = transcriptionManagers[mode];

      await manager.start(channel);
    } catch (error) {
      console.error("Failed to start channel", error);
      setChannelStatus(channel, "Unable to start listening.", "error");
      channel.isActive = false;
      activeTranscriptionModes.delete(channel.id);
      microphoneManager.stopStream(channel);
      if (channel.sourceType === "system" && channel.systemStream) {
        resetDesktopCapture();
        channel.systemStream = undefined;
      }
    }
  };

  const pauseChannel = async (
    channel: ConversationChannel,
  ): Promise<void> => {
    const mode = activeTranscriptionModes.get(channel.id);
    const manager = mode ? transcriptionManagers[mode] : chunkedTranscriptionManager;
    activeTranscriptionModes.delete(channel.id);

    await manager.stop(channel.id);

    channel.isActive = false;

    if (channel.sourceType === "system" && channel.systemStream) {
      resetDesktopCapture();
      channel.systemStream = undefined;
    }

    microphoneManager.stopStream(channel);
    setChannelStatus(channel, "Channel paused.", "info");
  };

  const toggleChannel = async (id: string): Promise<void> => {
    const channel = channels.value.find((item) => item.id === id);
    if (!channel) return;
    selectChannel(id);
    if (channel.isActive) {
      await pauseChannel(channel);
    } else {
      await startChannel(channel);
    }
  };

  watch(
    channels,
    () => {
      ensureSelectedChannel();
      if (!channelPersistenceReady) {
        return;
      }
      persistChannels();
    },
    { deep: true },
  );

  watch(selectedChannelId, () => {
    if (!channelPersistenceReady) {
      return;
    }
    persistSelectedChannel();
  });

  watch(
    conversationHistory,
    () => {
      persistHistory();
    },
    { deep: true },
  );

  watch(showTranslationAlternatives, () => {
    persistAlternativePreference();
  });

  watch(translationAlternativesLimit, (value) => {
    const clamped = clampAlternativeLimit(value);
    if (clamped !== value) {
      translationAlternativesLimit.value = clamped;
      return;
    }
    persistAlternativeLimit();
  });

  watch(
    transcriptionMode,
    (newMode, oldMode) => {
      if (!channelPersistenceReady) {
        return;
      }

      persistTranscriptionMode();

      if (newMode !== oldMode && isAnyChannelActive.value) {
        setGlobalStatus(
          "Transcription mode changes apply to newly started channels.",
          "info",
        );
      }
    },
  );

  onMounted(async () => {
    restoreHistory();
    restoreAlternativePreference();
    restoreAlternativeLimit();
    restoreTranscriptionMode();

    const restoredFromStorage = restoreChannels();
    restoreSelectedChannel();
    ensureSelectedChannel();

    await microphoneManager.refresh();

    try {
      const supported = await getSupportedLanguages();
      if (supported.length) {
        languages.value = supported;
      }
    } catch (error) {
      console.warn("Failed to fetch supported languages", error);
    }

    if (!restoredFromStorage && channels.value.length === 0) {
      addChannel(createChannel("microphone", "Participant A", "en", ["es"]));
      addChannel(createChannel("microphone", "Participant B", "es", ["en"]));
      ensureSelectedChannel();
    }

    channelPersistenceReady = true;
    persistChannels();
    persistSelectedChannel();
    persistAlternativePreference();
    persistAlternativeLimit();
    persistTranscriptionMode();

    if (typeof navigator !== "undefined" && navigator.mediaDevices) {
      const handler = () => {
        void microphoneManager.refresh();
      };

      if (typeof navigator.mediaDevices.addEventListener === "function") {
        navigator.mediaDevices.addEventListener("devicechange", handler);
        deviceChangeCleanup = () => {
          navigator.mediaDevices.removeEventListener("devicechange", handler);
        };
      } else if ("ondevicechange" in navigator.mediaDevices) {
        const original = navigator.mediaDevices.ondevicechange;
        navigator.mediaDevices.ondevicechange = handler;
        deviceChangeCleanup = () => {
          navigator.mediaDevices.ondevicechange = original;
        };
      }
    }
  });

  onUnmounted(() => {
    deviceChangeCleanup?.();
    deviceChangeCleanup = null;
    activeTranscriptionModes.clear();
    void chunkedTranscriptionManager.stopAll();
    void streamingTranscriptionManager.stopAll();
    channels.value.forEach((channel) => {
      microphoneManager.stopStream(channel);
      if (channel.systemStream) {
        channel.systemStream.getTracks().forEach((track) => track.stop());
        channel.systemStream = undefined;
      }
    });
  });

  return {
    languages,
    channels,
    selectedChannelId,
    selectedChannel,
    conversationHistory,
    recentHistory,
    globalStatus,
    showTranslationAlternatives,
    translationAlternativesLimit,
    transcriptionMode,
    speechSupported,
    desktopCaptureSupported,
    hasSystemChannel,
    isAnyChannelActive,
    isSpeaking,
    audioInputOptions: microphoneManager.audioInputOptions,
    isRefreshingAudioInputs: microphoneManager.isRefreshing,
    addMicrophoneChannel,
    addSystemChannel,
    removeChannel,
    toggleChannel,
    selectChannel,
    getLanguageName,
    formatTimestamp,
    handleMicrophoneSelectionChange: microphoneManager.handleSelectionChange,
    refreshAudioInputs: microphoneManager.refresh,
    resolveMicrophoneLabel: microphoneManager.resolveLabel,
  };
};
