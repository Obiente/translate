export type StatusType = "info" | "success" | "error" | "processing" | "listening";

export interface StatusMessage {
  message: string;
  type: StatusType;
}

export interface LanguageOption {
  code: string;
  name: string;
}

export type ChannelType = "microphone" | "system";

export interface TranslationEntry {
  primary: string;
  alternatives: string[];
}

export interface ConversationChannel {
  id: string;
  label: string;
  sourceType: ChannelType;
  sourceLanguage: string;
  targetLanguages: string[];
  liveTranscript: string;
  liveTranslations: Record<string, string>;
  lastFinalTranscript: string;
  translations: Record<string, TranslationEntry>;
  detectedLanguage: string | null;
  isActive: boolean;
  isTranslating: boolean;
  status: StatusMessage | null;
  autoSpeak: boolean;
  systemStream?: MediaStream;
  microphoneDeviceId?: string | null;
  microphoneDeviceLabel?: string | null;
  microphoneStream?: MediaStream | null;
}

export interface StoredChannelConfig {
  id: string;
  label: string;
  sourceType: ChannelType;
  sourceLanguage: string;
  targetLanguages: string[];
  autoSpeak: boolean;
  microphoneDeviceId?: string | null;
  microphoneDeviceLabel?: string | null;
}

export interface ConversationHistoryEntry {
  id: string;
  channelId: string;
  channelLabel: string;
  sourceLanguage: string;
  detectedLanguage: string | null;
  transcript: string;
  translations: Record<string, TranslationEntry>;
  timestamp: string;
}
