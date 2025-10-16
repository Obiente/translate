import { computed, type Ref, ref } from "vue";
import type { ConversationChannel } from "../../types/conversation";

export interface MicrophoneOption {
  deviceId: string;
  label: string;
}

export interface MicrophoneManager {
  audioInputs: Ref<MediaDeviceInfo[]>;
  audioInputOptions: Ref<MicrophoneOption[]>;
  isRefreshing: Ref<boolean>;
  refresh: () => Promise<void>;
  resolveLabel: (channel: ConversationChannel) => string;
  handleSelectionChange: (channel: ConversationChannel) => void;
  pickUnusedDeviceId: () => string | null;
  stopStream: (channel: ConversationChannel) => void;
  ensureStream: (channel: ConversationChannel) => Promise<void>;
}

interface UseMicrophoneManagerOptions {
  channels: Ref<ConversationChannel[]>;
  enumerateAudioInputs: () => Promise<MediaDeviceInfo[]>;
}

export const useMicrophoneManager = (
  options: UseMicrophoneManagerOptions,
): MicrophoneManager => {
  const { channels, enumerateAudioInputs } = options;

  const audioInputs = ref<MediaDeviceInfo[]>([]);
  const isRefreshing = ref<boolean>(false);

  const audioInputOptions = computed<MicrophoneOption[]>(() =>
    audioInputs.value
      .filter((device) => device.deviceId !== "default")
      .map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label && device.label.trim().length > 0
          ? device.label
          : device.deviceId === "communications"
          ? "Communications microphone"
          : `Microphone ${index + 1}`,
      }))
      .sort((a, b) => a.label.localeCompare(b.label))
  );

  const refresh = async (): Promise<void> => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) {
      audioInputs.value = [];
      return;
    }
    if (isRefreshing.value) {
      return;
    }
    isRefreshing.value = true;

    try {
      const devices = await enumerateAudioInputs();
      audioInputs.value = devices;

      const labelLookup = new Map<string, string>();
      devices.forEach((device) => {
        if (device.deviceId) {
          labelLookup.set(device.deviceId, device.label);
        }
      });

      channels.value.forEach((channel) => {
        if (channel.sourceType !== "microphone") {
          return;
        }
        if (!channel.microphoneDeviceId) {
          const defaultLabel = labelLookup.get("default") ||
            labelLookup.get("") || null;
          if (defaultLabel && defaultLabel.trim().length > 0) {
            channel.microphoneDeviceLabel = defaultLabel;
          }
          return;
        }
        const label = labelLookup.get(channel.microphoneDeviceId);
        if (label && label.trim().length > 0) {
          channel.microphoneDeviceLabel = label;
        }
      });
    } catch (error) {
      console.warn("Failed to enumerate audio inputs", error);
    } finally {
      isRefreshing.value = false;
    }
  };

  const resolveLabel = (channel: ConversationChannel): string => {
    if (channel.sourceType !== "microphone") {
      return "System audio capture";
    }

    if (channel.microphoneDeviceId) {
      const match = audioInputOptions.value.find(
        (option) => option.deviceId === channel.microphoneDeviceId,
      );
      const label = match?.label ?? channel.microphoneDeviceLabel;
      if (label && label.trim().length > 0) {
        return label;
      }
      if (channel.microphoneDeviceId === "default") {
        return "System default microphone";
      }
      return "Microphone in use";
    }

    if (
      channel.microphoneDeviceLabel &&
      channel.microphoneDeviceLabel.trim().length > 0
    ) {
      return channel.microphoneDeviceLabel;
    }

    return "System default microphone";
  };

  const handleSelectionChange = (channel: ConversationChannel): void => {
    if (channel.sourceType !== "microphone") {
      return;
    }
    if (!channel.microphoneDeviceId) {
      channel.microphoneDeviceLabel = "System default microphone";
      return;
    }
    const match = audioInputOptions.value.find(
      (option) => option.deviceId === channel.microphoneDeviceId,
    );
    if (match) {
      channel.microphoneDeviceLabel = match.label;
      return;
    }
    const liveTrackLabel = channel.microphoneStream?.getAudioTracks()[0]?.label;
    if (liveTrackLabel && liveTrackLabel.trim().length > 0) {
      channel.microphoneDeviceLabel = liveTrackLabel;
    }
  };

  const pickUnusedDeviceId = (): string | null => {
    const usedDeviceIds = new Set(
      channels.value
        .filter(
          (channel) =>
            channel.sourceType === "microphone" &&
            channel.microphoneDeviceId &&
            channel.microphoneDeviceId !== "default",
        )
        .map((channel) => channel.microphoneDeviceId as string),
    );

    const candidate = audioInputOptions.value.find(
      (option) =>
        option.deviceId &&
        option.deviceId !== "default" &&
        !usedDeviceIds.has(option.deviceId),
    );
    return candidate?.deviceId ?? null;
  };

  const stopStream = (channel: ConversationChannel): void => {
    if (!channel.microphoneStream) {
      return;
    }
    channel.microphoneStream.getTracks().forEach((track) => track.stop());
    channel.microphoneStream = null;
  };

  const ensureStream = async (
    channel: ConversationChannel,
  ): Promise<void> => {
    if (channel.sourceType !== "microphone") {
      return;
    }

    const existing = channel.microphoneStream;
    const hasLiveTrack = existing?.getAudioTracks().some(
      (track) => track.readyState === "live",
    );
    if (existing && hasLiveTrack) {
      return;
    }

    stopStream(channel);

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      throw new Error("Microphone capture is not supported in this browser.");
    }

    const trackConstraints: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };

    const constraints: MediaStreamConstraints = channel.microphoneDeviceId
      ? {
        audio: {
          ...trackConstraints,
          deviceId: { exact: channel.microphoneDeviceId },
        },
      }
      : {
        audio: trackConstraints,
      };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    channel.microphoneStream = stream;

    const trackLabel = stream.getAudioTracks()[0]?.label;
    if (trackLabel && trackLabel.trim().length > 0) {
      channel.microphoneDeviceLabel = trackLabel;
    }
  };

  return {
    audioInputs,
    audioInputOptions,
    isRefreshing,
    refresh,
    resolveLabel,
    handleSelectionChange,
    pickUnusedDeviceId,
    stopStream,
    ensureStream,
  };
};
