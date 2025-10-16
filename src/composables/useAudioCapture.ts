import { computed, type Ref, ref } from "vue";

interface BrowserSupport {
  mediaDevices: boolean;
  getDisplayMedia: boolean;
  getUserMedia: boolean;
  isSecureContext: boolean;
}

type ExtendedPermissionState = "granted" | "denied" | "prompt" | "unknown";

interface ActiveDesktopCapture {
  stream: MediaStream;
  audioElement: HTMLAudioElement;
}

const getBrowserSupport = (): BrowserSupport => ({
  mediaDevices: typeof navigator !== "undefined"
    ? !!navigator.mediaDevices
    : false,
  getDisplayMedia: typeof navigator !== "undefined"
    ? !!navigator.mediaDevices?.getDisplayMedia
    : false,
  getUserMedia: typeof navigator !== "undefined"
    ? !!navigator.mediaDevices?.getUserMedia
    : false,
  isSecureContext: typeof window !== "undefined"
    ? window.isSecureContext
    : false,
});

const stopStreamTracks = (stream: MediaStream) => {
  stream.getTracks().forEach((track) => track.stop());
};

export function useAudioCapture() {
  const support = computed(getBrowserSupport);
  const isCapturingDesktop: Ref<boolean> = ref(false);
  const desktopCapture: Ref<ActiveDesktopCapture | null> = ref(null);

  const ensureMicrophonePermission = async (): Promise<
    ExtendedPermissionState
  > => {
    if (!support.value.isSecureContext) {
      throw new Error(
        "Microphone access requires HTTPS. Please load the app over https:// or localhost.",
      );
    }

    if (!support.value.getUserMedia) {
      throw new Error("Microphone capture is not supported in this browser.");
    }

    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const state = stream.getAudioTracks().length > 0 ? "granted" : "unknown";
      return state;
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "NotAllowedError") {
        return "denied";
      }
      throw error instanceof Error
        ? error
        : new Error("Failed to request microphone access");
    } finally {
      if (stream) {
        stopStreamTracks(stream);
      }
    }
  };

  const ensureDesktopAudioPermission = async (): Promise<
    ExtendedPermissionState
  > => {
    if (!support.value.isSecureContext) {
      throw new Error(
        "Desktop capture requires HTTPS. Please load the app over https:// or localhost.",
      );
    }

    if (!support.value.getDisplayMedia) {
      throw new Error(
        "Desktop capture is not supported in this browser. Try Chrome, Edge, or Firefox 100+.",
      );
    }

    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: 1,
          width: { ideal: 640 },
          height: { ideal: 360 },
        },
        audio: true,
      });

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        stopStreamTracks(stream);
        return "denied";
      }

      return "granted";
    } catch (error: unknown) {
      if (
        error instanceof DOMException &&
        (error.name === "NotAllowedError" ||
          error.name === "PermissionDeniedError")
      ) {
        return "denied";
      }
      throw error instanceof Error
        ? error
        : new Error("Failed to request desktop capture access");
    } finally {
      if (stream) {
        stopStreamTracks(stream);
      }
    }
  };

  const enumerateAudioInputs = async (): Promise<MediaDeviceInfo[]> => {
    if (!support.value.mediaDevices) {
      throw new Error("Device enumeration is not supported in this browser.");
    }

    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      console.warn(
        "Unable to pre-authorise microphone before enumeration",
        error,
      );
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === "audioinput");
  };

  const startDesktopCapture = async (): Promise<MediaStream> => {
    let stream: MediaStream | null = null;
    let audioElement: HTMLAudioElement | null = null;
    try {
      const isFirefox = typeof navigator !== "undefined" &&
        /firefox/i.test(navigator.userAgent);
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: 1,
          width: { ideal: 640 },
          height: { ideal: 360 },
        },
        audio: true,
      });

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        stopStreamTracks(stream);
        throw new Error(
          'No audio track provided. Ensure "Share audio" is enabled in the share dialog.',
        );
      }

      if (!isFirefox) {
        const [track] = audioTracks;
        const supported = typeof navigator !== "undefined" &&
            navigator.mediaDevices?.getSupportedConstraints
          ? navigator.mediaDevices.getSupportedConstraints()
          : {};
        const desiredConstraints: MediaTrackConstraints = {};
        if ("echoCancellation" in supported) {
          desiredConstraints.echoCancellation = false;
        }
        if ("noiseSuppression" in supported) {
          desiredConstraints.noiseSuppression = false;
        }
        if ("autoGainControl" in supported) {
          desiredConstraints.autoGainControl = false;
        }
        if (Object.keys(desiredConstraints).length > 0) {
          try {
            // Apply post-capture to keep Chrome's audio share toggle visible while still disabling processing.
            await track.applyConstraints(desiredConstraints);
          } catch (constraintError) {
            console.warn(
              "Unable to apply desktop audio constraints",
              constraintError,
            );
          }
        }
      }

      audioElement = document.createElement("audio");
      audioElement.srcObject = stream;
      audioElement.autoplay = true;
      audioElement.muted = true;
      audioElement.setAttribute("playsinline", "true");
      audioElement.controls = false;
      audioElement.volume = 0;
      audioElement.style.position = "fixed";
      audioElement.style.top = "-9999px";
      audioElement.style.left = "-9999px";

      document.body.appendChild(audioElement);

      try {
        await audioElement.play();
      } catch (playError) {
        console.warn(
          "Desktop audio playback blocked, retrying with muted element",
          playError,
        );
        audioElement.muted = true;
        try {
          await audioElement.play();
        } catch (retryError) {
          console.warn(
            "Desktop audio playback still blocked after muting",
            retryError,
          );
        }
      }

      desktopCapture.value = { stream, audioElement };
      isCapturingDesktop.value = true;
      return stream;
    } catch (error: unknown) {
      isCapturingDesktop.value = false;
      if (audioElement) {
        audioElement.pause();
        audioElement.srcObject = null;
        if (audioElement.parentElement) {
          audioElement.parentElement.removeChild(audioElement);
        }
      }
      if (stream) {
        stopStreamTracks(stream);
      }
      if (error instanceof DOMException) {
        if (error.name === "NotAllowedError") {
          throw new Error(
            'Desktop capture permission denied. Allow screen sharing and enable "Share audio".',
          );
        }
        if (error.name === "AbortError") {
          throw new Error(
            "Desktop capture request cancelled. Try again when ready to share.",
          );
        }
        if (error.name === "NotSupportedError") {
          throw new Error(
            "Desktop audio capture is not supported in this browser.",
          );
        }
      }
      throw error instanceof Error
        ? error
        : new Error("Failed to start desktop capture");
    }
  };

  const disposeDesktopCapture = () => {
    if (!desktopCapture.value) {
      return;
    }

    const { stream, audioElement } = desktopCapture.value;
    audioElement.pause();
    audioElement.srcObject = null;
    if (audioElement.parentElement) {
      audioElement.parentElement.removeChild(audioElement);
    }
    stopStreamTracks(stream);
    desktopCapture.value = null;
    isCapturingDesktop.value = false;
  };

  const stopDesktopCapture = (): void => {
    disposeDesktopCapture();
  };

  const resetDesktopCapture = () => {
    disposeDesktopCapture();
  };

  const checkMicrophonePermission = async (): Promise<
    ExtendedPermissionState
  > => {
    if (!("permissions" in navigator)) {
      return "unknown";
    }
    try {
      const permission = await navigator.permissions.query({
        name: "microphone" as PermissionName,
      });
      return permission.state as ExtendedPermissionState;
    } catch (error) {
      console.warn("Unable to query microphone permission", error);
      return "unknown";
    }
  };

  return {
    support,
    isCapturingDesktop,
    ensureMicrophonePermission,
    ensureDesktopAudioPermission,
    enumerateAudioInputs,
    startDesktopCapture,
    stopDesktopCapture,
    resetDesktopCapture,
    checkMicrophonePermission,
  };
}
