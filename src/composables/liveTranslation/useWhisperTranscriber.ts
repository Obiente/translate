import type { ConversationChannel, StatusType } from "../../types/conversation";

interface TranscriptionResult {
    text: string;
    language?: string;
    fullText?: string;
    isFinal?: boolean;
    sequence?: number;
}

interface UseWhisperTranscriberOptions {
    ensureStream: (channel: ConversationChannel) => Promise<MediaStream>;
    onTranscription: (
        channel: ConversationChannel,
        result: TranscriptionResult,
    ) => Promise<void> | void;
    updateStatus: (
        channel: ConversationChannel,
        message: string,
        type: StatusType,
    ) => void;
    endpoint: string;
    chunkIntervalMs?: number;
}

interface RecorderEntry {
    recorder: MediaRecorder;
    stream: MediaStream;
    channel: ConversationChannel;
}


const resolveMimeType = (): string | undefined => {
    if (typeof MediaRecorder === "undefined") {
        return undefined;
    }

    const preferred = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
        "audio/wav",
    ];

    return preferred.find((type) => MediaRecorder.isTypeSupported(type));
};

export interface WhisperTranscriberManager {
    start: (channel: ConversationChannel) => Promise<void>;
    stop: (channelId: string) => Promise<void>;
    dispose: (channelId: string) => Promise<void>;
    stopAll: () => Promise<void>;
}

export const useWhisperTranscriber = (
    options: UseWhisperTranscriberOptions,
): WhisperTranscriberManager => {
    const {
        ensureStream,
        onTranscription,
        updateStatus,
        endpoint,
        chunkIntervalMs = 4000,
    } = options;

    const recorders = new Map<string, RecorderEntry>();
    const processingQueues = new Map<string, Promise<void>>();
    const activeChannels = new Set<string>();
    const chunkTimers = new Map<string, ReturnType<typeof setTimeout>>();
    const mimeType = resolveMimeType();
    const previousTranscripts = new Map<string, string>();

    const enqueue = async (channel: ConversationChannel, blob: Blob) => {
        if (!blob || blob.size === 0) {
            return;
        }

        const previousQueue = processingQueues.get(channel.id) ??
            Promise.resolve();

        const task = previousQueue
            .catch(() => undefined)
            .then(async () => {
                if (!activeChannels.has(channel.id)) {
                    return;
                }
                updateStatus(channel, "Processing audio…", "processing");
                try {
                    const formData = new FormData();
                    const type = blob.type ?? "";
                    const extension = type.includes("ogg")
                        ? "ogg"
                        : type.includes("mp4")
                        ? "m4a"
                        : "webm";
                    const filename = `${channel.id}-${Date.now()}.${extension}`;
                    formData.append("channel_id", channel.id);
                    if (
                        channel.sourceLanguage &&
                        channel.sourceLanguage !== "auto"
                    ) {
                        formData.append("language", channel.sourceLanguage);
                    }
                    formData.append("file", blob, filename);

                    const response = await fetch(endpoint, {
                        method: "POST",
                        body: formData,
                    });

                    if (!response.ok) {
                        const errorText = await response.text().catch(() =>
                            response.statusText
                        );
                        throw new Error(
                            `Transcription request failed: ${errorText}`,
                        );
                    }

                    const payload = (await response.json()) as {
                        text?: string;
                        language?: string;
                    };
                    const fullText = (payload.text ?? "").trim();
                    const language = payload.language;

                    if (fullText.length === 0) {
                        updateStatus(channel, "Listening…", "listening");
                        return;
                    }

                    const previousFull = previousTranscripts.get(channel.id) ??
                        "";
                    let delta = fullText;

                    if (previousFull.length > 0) {
                        if (
                            fullText.length >= previousFull.length &&
                            fullText.startsWith(previousFull)
                        ) {
                            delta = fullText.slice(previousFull.length).replace(
                                /^\s+/,
                                "",
                            );
                        } else if (fullText.length >= previousFull.length) {
                            const max = Math.min(
                                previousFull.length,
                                fullText.length,
                            );
                            let idx = 0;
                            while (
                                idx < max && previousFull[idx] === fullText[idx]
                            ) {
                                idx += 1;
                            }
                            const slicePoint = idx > 0 ? idx : 0;
                            delta = fullText.slice(slicePoint).replace(
                                /^\s+/,
                                "",
                            );
                            if (delta.length === 0) {
                                delta = fullText;
                            }
                        }

                        if (fullText.length < previousFull.length) {
                            delta = fullText;
                        }
                    }

                    previousTranscripts.set(channel.id, fullText);

                    const normalizedDelta = delta.trim();

                    if (normalizedDelta.length > 0) {
                        await onTranscription(channel, {
                            text: normalizedDelta,
                            fullText,
                            language,
                        });
                        updateStatus(channel, "Listening…", "listening");
                    } else {
                        updateStatus(channel, "Listening…", "listening");
                    }
                } catch (error) {
                    console.error("Failed to transcribe audio chunk", error);
                    updateStatus(
                        channel,
                        "Failed to transcribe the latest audio chunk.",
                        "error",
                    );
                }
            });

        processingQueues.set(channel.id, task);
    };

    const createRecorderInstance = (
        channel: ConversationChannel,
        stream: MediaStream,
    ): MediaRecorder => {
        let recorder: MediaRecorder;

        try {
            recorder = mimeType
                ? new MediaRecorder(stream, { mimeType })
                : new MediaRecorder(stream);
        } catch (error) {
            console.warn(
                "Falling back to default MediaRecorder options",
                error,
            );
            recorder = new MediaRecorder(stream);
        }

        recorder.addEventListener("dataavailable", (event: BlobEvent) => {
            if (!activeChannels.has(channel.id)) {
                return;
            }
            enqueue(channel, event.data);
        });

        recorder.addEventListener("error", (event) => {
            console.error("MediaRecorder error", event);
            updateStatus(channel, "Audio capture interrupted.", "error");
        });

        return recorder;
    };

    const clearChunkTimer = (channelId: string) => {
        const timer = chunkTimers.get(channelId);
        if (timer !== undefined) {
            clearTimeout(timer);
            chunkTimers.delete(channelId);
        }
    };

    const scheduleNextChunk = (channelId: string) => {
        if (chunkIntervalMs <= 0) {
            return;
        }

        clearChunkTimer(channelId);

        const timer = setTimeout(() => {
            chunkTimers.delete(channelId);
            requestChunk(channelId);
        }, chunkIntervalMs);

        chunkTimers.set(channelId, timer);
    };

    const requestChunk = (channelId: string) => {
        const entry = recorders.get(channelId);
        if (!entry) {
            return;
        }

        clearChunkTimer(channelId);

        const { recorder, stream, channel } = entry;

        if (recorder.state === "inactive") {
            if (!activeChannels.has(channelId)) {
                return;
            }

            try {
                const nextRecorder = createRecorderInstance(channel, stream);
                entry.recorder = nextRecorder;
                nextRecorder.start();
                scheduleNextChunk(channelId);
            } catch (error) {
                console.error("Failed to restart MediaRecorder", error);
                updateStatus(channel, "Audio capture interrupted.", "error");
            }
            return;
        }

        recorder.addEventListener(
            "stop",
            () => {
                if (!activeChannels.has(channelId)) {
                    return;
                }

                try {
                    const nextRecorder = createRecorderInstance(
                        channel,
                        stream,
                    );
                    entry.recorder = nextRecorder;
                    nextRecorder.start();
                    scheduleNextChunk(channelId);
                } catch (error) {
                    console.error("Failed to restart MediaRecorder", error);
                    updateStatus(
                        channel,
                        "Audio capture interrupted.",
                        "error",
                    );
                }
            },
            { once: true },
        );

        try {
            recorder.stop();
        } catch (error) {
            console.warn("Failed to stop MediaRecorder for chunk", error);
            scheduleNextChunk(channelId);
        }
    };

    const createRecorder = async (
        channel: ConversationChannel,
    ): Promise<RecorderEntry> => {
        const sourceStream = await ensureStream(channel);

        if (!sourceStream) {
            throw new Error("No media stream available for transcription");
        }

        const audioTracks = sourceStream.getAudioTracks();

        if (audioTracks.length === 0) {
            throw new Error("Media stream does not contain an audio track");
        }

        const audioStream = new MediaStream();
        audioTracks.forEach((track) => {
            const clone = track.clone();
            audioStream.addTrack(clone);
        });

        const recorder = createRecorderInstance(channel, audioStream);

        return { recorder, stream: audioStream, channel };
    };

    const stopRecorder = async (channelId: string) => {
        const entry = recorders.get(channelId);
        if (!entry) {
            return;
        }

        const { recorder } = entry;

        if (recorder.state !== "inactive") {
            try {
                recorder.stop();
            } catch (error) {
                console.warn("Failed to stop MediaRecorder", error);
            }
        }

        recorders.delete(channelId);
    };

    const flushQueue = async (channelId: string) => {
        const queue = processingQueues.get(channelId);
        try {
            await queue;
        } catch (error) {
            console.warn("Transcription queue terminated with error", error);
        }
        processingQueues.delete(channelId);
    };

    const start = async (channel: ConversationChannel): Promise<void> => {
        if (recorders.has(channel.id)) {
            return;
        }

        const entry = await createRecorder(channel);
        recorders.set(channel.id, entry);
        activeChannels.add(channel.id);
        previousTranscripts.delete(channel.id);

        updateStatus(channel, "Listening…", "listening");
        channel.liveTranscript = "";

        try {
            entry.recorder.start();
            scheduleNextChunk(channel.id);
        } catch (error) {
            recorders.delete(channel.id);
            activeChannels.delete(channel.id);
            clearChunkTimer(channel.id);
            throw error;
        }
    };

    const stop = async (channelId: string): Promise<void> => {
        activeChannels.delete(channelId);
        clearChunkTimer(channelId);
        previousTranscripts.delete(channelId);
        await stopRecorder(channelId);
        await flushQueue(channelId);
    };

    const dispose = async (channelId: string): Promise<void> => {
        const entry = recorders.get(channelId);
        await stop(channelId);
        entry?.stream.getTracks().forEach((track) => track.stop());
        recorders.delete(channelId);
        previousTranscripts.delete(channelId);
    };

    const stopAll = async (): Promise<void> => {
        const ids = Array.from(recorders.keys());
        await Promise.all(ids.map((id) => stop(id)));
        recorders.clear();
        processingQueues.clear();
        activeChannels.clear();
        chunkTimers.forEach((timer) => clearTimeout(timer));
        chunkTimers.clear();
        previousTranscripts.clear();
    };

    return {
        start,
        stop,
        dispose,
        stopAll,
    };
};
