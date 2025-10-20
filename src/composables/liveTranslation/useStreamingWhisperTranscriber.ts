import type {
    ConversationChannel,
    StatusType,
    TranslationEntry,
} from "../../types/conversation";
import type { WhisperTranscriberManager } from "./useWhisperTranscriber";
import { watch } from "vue";
import { normalizeTranslationMap } from "../../utils/translation";
import { useRoomManager } from "../useRoomManager";
import { filterAudioChunk, getDefaultFilterConfig } from "../../utils/audioFilters";
const DEFAULT_CHUNK_INTERVAL_MS = 300;
const DEFAULT_KEEPALIVE_INTERVAL_MS = 5000;
const DEFAULT_KEEPALIVE_TIMEOUT_MS = 45000;
const RECONNECT_BASE_DELAY_MS = 500;
const RECONNECT_MAX_DELAY_MS = 15000;

type EnsureStreamFn = (channel: ConversationChannel) => Promise<MediaStream>;

interface TranscriptionResult {
    text: string;
    language?: string;
    fullText?: string;
    isFinal?: boolean;
    sequence?: number;
    translations?: Record<string, TranslationEntry>;
}

interface UseStreamingWhisperTranscriberOptions {
    ensureStream: EnsureStreamFn;
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
    getTranslationAlternativeLimit?: () => number;
}

interface StreamingState {
    channel: ConversationChannel;
    stream: MediaStream;
    audioContext: AudioContext | null;
    processor: ScriptProcessorNode | null;
    websocket: WebSocket | null;
    sequence: number;
    pending: Promise<void>;
    isClosing: boolean;
    finalChunkSent: boolean;
    finalResolvers: Array<() => void>;
    cleanedUp: boolean;
    chunkTimer: number | null;
    audioBuffer: Float32Array[];
    sampleRate: number;
    keepaliveTimer: number | null;
    lastKeepaliveAt: number;
    reconnectAttempts: number;
    reconnectTimer: number | null;
}

// Removed resampleTo16kHz - server will handle resampling with ffmpeg for better quality

const float32ToInt16 = (buffer: Float32Array): Int16Array => {
    const length = buffer.length;
    const result = new Int16Array(length);
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, buffer[i]));
        result[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    return result;
};

const createWavChunk = (
    pcmData: Int16Array,
    sampleRate: number,
): ArrayBuffer => {
    const buffer = new ArrayBuffer(44 + pcmData.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    const byteRate = sampleRate * 1 * 2; // sampleRate * channels * bytesPerSample

    // WAV header
    writeString(0, "RIFF");
    view.setUint32(4, 36 + pcmData.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, 1, true); // AudioFormat (PCM)
    view.setUint16(22, 1, true); // NumChannels (mono)
    view.setUint32(24, sampleRate, true); // SampleRate (use original sample rate)
    view.setUint32(28, byteRate, true); // ByteRate (sampleRate * channels * bytesPerSample)
    view.setUint16(32, 2, true); // BlockAlign (channels * bytesPerSample)
    view.setUint16(34, 16, true); // BitsPerSample
    writeString(36, "data");
    view.setUint32(40, pcmData.length * 2, true);

    // PCM data
    let offset = 44;
    for (let i = 0; i < pcmData.length; i++) {
        view.setInt16(offset, pcmData[i], true);
        offset += 2;
    }

    return buffer;
};

const blobToBase64 = async (blob: Blob): Promise<string> => {
    const reader = new FileReader();
    const promise = new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
            const result = typeof reader.result === "string"
                ? reader.result
                : "";
            const base64 = result.includes(",") ? result.split(",")[1] : result;
            resolve(base64 ?? "");
        };
        reader.onerror = () =>
            reject(reader.error ?? new Error("Failed to read audio blob"));
    });

    reader.readAsDataURL(blob);
    return promise;
};

const cloneAudioStream = (source: MediaStream): MediaStream => {
    const audioStream = new MediaStream();
    source.getAudioTracks().forEach((track) => {
        audioStream.addTrack(track.clone());
    });
    return audioStream;
};

const sendJson = (socket: WebSocket | null, payload: unknown): void => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }
    socket.send(JSON.stringify(payload));
};

const resolvePayloadText = (value: unknown): string =>
    typeof value === "string" ? value : "";

const resolvePayloadNumber = (value: unknown): number | undefined =>
    typeof value === "number" && Number.isFinite(value) ? value : undefined;

const resolveTargetLanguages = (targets: string[]): string[] =>
    Array.from(
        new Set(
            targets
                .map((code) => (typeof code === "string" ? code.trim() : ""))
                .filter((code) => code.length > 0),
        ),
    );

const clampAlternativeLimit = (value: number): number =>
    Math.max(0, Math.min(5, Number.isFinite(value) ? Math.trunc(value) : 0));

export const useStreamingWhisperTranscriber = (
    options: UseStreamingWhisperTranscriberOptions,
): WhisperTranscriberManager => {
    const {
        ensureStream,
        onTranscription,
        updateStatus,
        endpoint,
        chunkIntervalMs = DEFAULT_CHUNK_INTERVAL_MS,
        getTranslationAlternativeLimit,
    } = options;

    const states = new Map<string, StreamingState>();
    // Room integration: reuse the room websocket if present
    const room = useRoomManager();

    // Keep streaming sockets joined to the current room even if the user joins later
    // or if the streaming connection is separate from the room manager socket.
    watch(
        () => room.roomId.value,
        (newRoomId, oldRoomId) => {
            // If leaving a room, notify all active streaming sockets as well
            if (!newRoomId && oldRoomId) {
                states.forEach((state) => {
                    const socket = state.websocket;
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        sendJson(socket, { type: "leave_room" });
                    }
                });
                return;
            }

            // If joining or switching rooms, send join to every active streaming socket
            if (newRoomId) {
                states.forEach((state) => {
                    // Ensure we have a socket (this may reuse the room socket)
                    const socket = state.websocket ?? room.ensureSocket();
                    if (socket.readyState === WebSocket.OPEN) {
                        sendJson(socket, {
                            type: "join_room",
                            room_id: newRoomId,
                            peer_id: room.peerId.value ?? undefined,
                            peer_label: room.peerLabel.value ?? undefined,
                        });
                    } else {
                        socket.addEventListener(
                            "open",
                            () =>
                                sendJson(socket, {
                                    type: "join_room",
                                    room_id: newRoomId,
                                    peer_id: room.peerId.value ?? undefined,
                                    peer_label: room.peerLabel.value ?? undefined,
                                }),
                            { once: true },
                        );
                    }
                });
            }
        },
        { immediate: false },
    );

    // Keep peer identity synchronized across sockets; if peerId changes, re-send join to update server-side identity
    watch(
        () => room.peerId.value,
        (newPeerId) => {
            if (!room.roomId.value) return;
            states.forEach((state) => {
                const socket = state.websocket ?? room.ensureSocket();
                const payload: Record<string, unknown> = {
                    type: "join_room",
                    room_id: room.roomId.value,
                    peer_id: newPeerId ?? undefined,
                    peer_label: room.peerLabel.value ?? undefined,
                };
                if (socket.readyState === WebSocket.OPEN) {
                    sendJson(socket, payload);
                } else {
                    socket.addEventListener("open", () => sendJson(socket, payload), { once: true });
                }
            });
        },
        { immediate: false },
    );

    const clearKeepalive = (state: StreamingState): void => {
        if (state.keepaliveTimer !== null) {
            clearInterval(state.keepaliveTimer);
            state.keepaliveTimer = null;
        }
    };

    const clearReconnect = (state: StreamingState): void => {
        if (state.reconnectTimer !== null) {
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = null;
        }
    };

    const cleanupState = (channelId: string, state: StreamingState): void => {
        if (state.cleanedUp) {
            return;
        }
        state.cleanedUp = true;

        // Clear chunk timer
        if (state.chunkTimer !== null) {
            clearInterval(state.chunkTimer);
            state.chunkTimer = null;
        }

        clearKeepalive(state);
        clearReconnect(state);

        // Stop audio processing
        if (state.processor) {
            try {
                state.processor.disconnect();
            } catch (error) {
                console.warn("Failed to disconnect audio processor", error);
            }
        }

        if (state.audioContext && state.audioContext.state !== "closed") {
            try {
                state.audioContext.close();
            } catch (error) {
                console.warn("Failed to close audio context", error);
            }
        }

        state.stream.getTracks().forEach((track) => track.stop());
        if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
            try {
                state.websocket.close();
            } catch (error) {
                console.warn("Failed to close streaming websocket", error);
            }
        }
        states.delete(channelId);
    };

    const scheduleReconnect = (state: StreamingState): void => {
        if (state.isClosing || state.cleanedUp) {
            return;
        }

        if (state.reconnectTimer !== null) {
            return;
        }

        const delay = Math.min(
            RECONNECT_BASE_DELAY_MS * 0.5 ** state.reconnectAttempts,
            RECONNECT_MAX_DELAY_MS,
        );

        state.reconnectAttempts += 1;
        state.reconnectTimer = window.setTimeout(() => {
            state.reconnectTimer = null;
            void connectWebSocket(state).catch((error) => {
                console.warn("Streaming reconnect attempt failed", error);
                scheduleReconnect(state);
            });
        }, delay);
    };

    const startKeepalive = (state: StreamingState): void => {
        clearKeepalive(state);
        state.lastKeepaliveAt = Date.now();
        state.keepaliveTimer = window.setInterval(() => {
            const socket = state.websocket;
            if (!socket || socket.readyState !== WebSocket.OPEN) {
                return;
            }

            const now = Date.now();
            if (now - state.lastKeepaliveAt >= DEFAULT_KEEPALIVE_TIMEOUT_MS) {
                console.warn(
                    "Streaming keepalive timed out; forcing reconnect",
                );
                try {
                    socket.close();
                } catch (error) {
                    console.warn("Failed to close stale websocket", error);
                }
                return;
            }

            sendJson(socket, { type: "ping", ts: now });
        }, DEFAULT_KEEPALIVE_INTERVAL_MS);
    };

    const createAudioProcessor = (
        channel: ConversationChannel,
        stream: MediaStream,
        state: StreamingState,
    ): void => {
        try {
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1); // Original buffer size for stable capture

            state.audioContext = audioContext;
            state.processor = processor;
            state.audioBuffer = [];
            state.sampleRate = audioContext.sampleRate;

            console.log(
                `Audio context sample rate: ${audioContext.sampleRate}Hz (will be sent to server for resampling)`,
            );

            processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                // Store at original sample rate - server will handle resampling
                state.audioBuffer.push(new Float32Array(inputBuffer));
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            // Set up timer to send chunks at regular intervals
            state.chunkTimer = setInterval(() => {
                if (state.audioBuffer.length > 0 && !state.isClosing) {
                    processAudioBuffer(state, false);
                }
            }, chunkIntervalMs);
        } catch (error) {
            console.error("Failed to create audio processor", error);
            updateStatus(channel, "Audio processing setup failed.", "error");
            throw error;
        }
    };

    const processAudioBuffer = (
        state: StreamingState,
        isFinal: boolean,
    ): void => {
        const socketOpen = !!state.websocket &&
            state.websocket.readyState === WebSocket.OPEN;

        if (!socketOpen && !isFinal) {
            return;
        }

        if (!socketOpen && isFinal) {
            state.audioBuffer = [];
            if (!state.finalChunkSent) {
                state.finalChunkSent = true;
                state.finalResolvers.splice(0).forEach((resolve) => resolve());
            }
            return;
        }

        if (state.audioBuffer.length === 0 && !isFinal) {
            return;
        }

        try {
            // Combine all buffered audio
            const totalSamples = state.audioBuffer.reduce(
                (sum, buffer) => sum + buffer.length,
                0,
            );
            const combinedBuffer = new Float32Array(totalSamples);
            let offset = 0;

            for (const buffer of state.audioBuffer) {
                combinedBuffer.set(buffer, offset);
                offset += buffer.length;
            }

            // Clear the buffer for next chunk
            state.audioBuffer = [];

            if (combinedBuffer.length === 0 && !isFinal) {
                return;
            }

            // Apply VAD and noise gate filtering - skip sending if no voice activity
            let filteredBuffer: Float32Array = combinedBuffer;
            if (!isFinal && combinedBuffer.length > 0) {
                const filterConfig = getDefaultFilterConfig(state.sampleRate);
                const filtered = filterAudioChunk(
                    combinedBuffer,
                    filterConfig.vad,
                    filterConfig.gate
                );
                
                // If no voice activity detected, skip sending this chunk
                if (filtered === null) {
                    console.log('[VAD] Silence detected, skipping chunk');
                    return;
                }
                
                console.log('[VAD] Voice activity detected, sending chunk');
                filteredBuffer = new Float32Array(filtered);
            }

            // Convert to WAV
            const pcmData = float32ToInt16(filteredBuffer);
            const wavBuffer = createWavChunk(pcmData, state.sampleRate);
            const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

            state.pending = state.pending
                .then(() => sendChunk(state, wavBlob, isFinal))
                .catch((error) => {
                    console.error("Failed to stream audio chunk", error);
                });
        } catch (error) {
            console.error("Failed to process audio buffer", error);
        }
    };

    const sendChunk = async (
        state: StreamingState,
        blob: Blob,
        isFinal: boolean,
    ): Promise<void> => {
        const socket = state.websocket;
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            return;
        }

        const base64 = await blobToBase64(blob);

        const targets = resolveTargetLanguages(state.channel.targetLanguages);
        const alternativeLimit = clampAlternativeLimit(
            getTranslationAlternativeLimit
                ? getTranslationAlternativeLimit()
                : 0,
        );
        const payload: Record<string, unknown> = {
            type: "chunk",
            sequence: state.sequence,
            data: base64,
            mime_type: "audio/wav",
            sample_rate: state.sampleRate,
            is_final: isFinal,
        };
        // Attach room metadata for implicit join/continuity on the server
        if (room.roomId.value) {
            payload.room_id = room.roomId.value;
            if (room.peerId.value) payload.peer_id = room.peerId.value;
            if (room.peerLabel.value) payload.peer_label = room.peerLabel.value;
        }
        if (targets.length > 0) {
            payload.target_languages = targets;
        }
        if (alternativeLimit > 0) {
            payload.translation_alternatives = alternativeLimit;
        }

        sendJson(socket, payload);

        state.sequence += 1;

        if (isFinal) {
            state.finalChunkSent = true;
            state.finalResolvers.splice(0).forEach((resolve) => resolve());
        }
    };

    const waitForFinalChunk = (state: StreamingState): Promise<void> => {
        if (state.finalChunkSent || state.sequence === 0) {
            state.finalChunkSent = true;
            return Promise.resolve();
        }

        return new Promise<void>((resolve) => {
            state.finalResolvers.push(resolve);
        });
    };

    const handleMessage = async (
        state: StreamingState,
        rawData: unknown,
    ): Promise<void> => {
        if (typeof rawData !== "string") {
            return;
        }

        let payload: any;
        try {
            payload = JSON.parse(rawData);
        } catch (error) {
            console.warn(
                "Failed to parse streaming transcription payload",
                error,
            );
            return;
        }

        if (payload?.type === "error") {
            const detail = resolvePayloadText(payload.detail) ||
                "Streaming transcription error.";
            updateStatus(state.channel, detail, "error");
            return;
        }

        if (payload?.type === "pong") {
            state.lastKeepaliveAt = Date.now();
            return;
        }

        if (payload?.type === "ping") {
            const socket = state.websocket;
            if (socket && socket.readyState === WebSocket.OPEN) {
                sendJson(socket, { type: "pong", ts: payload?.ts });
            }
            return;
        }

        if (payload?.type !== "transcript") {
            return;
        }

        const delta = resolvePayloadText(payload.text);
        const fullText = resolvePayloadText(payload.fullText);
        const language = resolvePayloadText(payload.language) || undefined;
        const isFinal = Boolean(payload.isFinal);
        const sequence = resolvePayloadNumber(payload.sequence);
        const translations = normalizeTranslationMap(payload.translations);

        try {
            await onTranscription(state.channel, {
                text: delta,
                fullText,
                language,
                isFinal,
                sequence,
                translations,
            });
        } catch (error) {
            console.error("Streaming transcription callback failed", error);
        }
    };

    const attachSocketHandlers = (state: StreamingState, socket: WebSocket) => {
        socket.addEventListener("message", (event) => {
            if (state.websocket !== socket) {
                return;
            }
            void handleMessage(state, event.data);
        });

        socket.addEventListener("close", () => {
            if (state.websocket !== socket) {
                return;
            }

            clearKeepalive(state);

            if (state.isClosing || state.cleanedUp) {
                cleanupState(state.channel.id, state);
                return;
            }

            state.websocket = null;
            updateStatus(
                state.channel,
                "Streaming connection lost. Reconnecting…",
                "processing",
            );
            scheduleReconnect(state);
        });

        socket.addEventListener("error", (event) => {
            if (state.websocket !== socket) {
                return;
            }
            console.error("Streaming websocket error", event);
            updateStatus(state.channel, "Streaming connection error.", "error");
        });
    };

    const connectWebSocket = async (state: StreamingState): Promise<void> => {
        if (state.cleanedUp || state.isClosing) {
            return;
        }

        clearReconnect(state);

        // Reuse existing socket from room if pointing to same endpoint; otherwise create new
        const canReuseRoomSocket = (() => {
            try {
                const url = new URL(endpoint);
                const configured = (import.meta.env.VITE_WHISPER_STREAMING_ENDPOINT as string | undefined)?.trim() || endpoint;
                const roomUrl = new URL(configured);
                return url.host === roomUrl.host && url.pathname === roomUrl.pathname;
            } catch { return false; }
        })();

        const websocket = canReuseRoomSocket ? room.ensureSocket() : new WebSocket(endpoint);
        state.websocket = websocket;
        state.sequence = 0;
        state.finalChunkSent = false;

        const openPromise = new Promise<void>((resolve, reject) => {
            const handleOpen = () => {
                websocket.removeEventListener("open", handleOpen);
                websocket.removeEventListener("error", handleError);
                resolve();
            };
            const handleError = () => {
                websocket.removeEventListener("open", handleOpen);
                websocket.removeEventListener("error", handleError);
                reject(
                    new Error(
                        "Failed to establish streaming transcription connection",
                    ),
                );
            };

            websocket.addEventListener("open", handleOpen);
            websocket.addEventListener("error", handleError);
        });

        attachSocketHandlers(state, websocket);

        if (!canReuseRoomSocket) {
            await openPromise;
        } else {
            // If reusing, ensure it's open
            if (websocket.readyState !== WebSocket.OPEN) {
                await openPromise;
            }
        }

        state.reconnectAttempts = 0;
        state.lastKeepaliveAt = Date.now();
        startKeepalive(state);

        const payload: Record<string, unknown> = {
            type: "start",
            channel_id: state.channel.id,
        };
        // Attach room metadata as part of start for server-side implicit join
        if (room.roomId.value) {
            payload["room_id"] = room.roomId.value;
            if (room.peerId.value) payload["peer_id"] = room.peerId.value;
            if (room.peerLabel.value) payload["peer_label"] = room.peerLabel.value;
        }

        if (
            state.channel.sourceLanguage &&
            state.channel.sourceLanguage !== "auto"
        ) {
            payload.language = state.channel.sourceLanguage;
        }

        const targets = resolveTargetLanguages(state.channel.targetLanguages);
        if (targets.length > 0) {
            payload.target_languages = targets;
        }

        const alternativeLimit = clampAlternativeLimit(
            getTranslationAlternativeLimit
                ? getTranslationAlternativeLimit()
                : 0,
        );
        if (alternativeLimit > 0) {
            payload.translation_alternatives = alternativeLimit;
        }

        // Attach to any joined room if present
        if (room.roomId.value) {
            // Ask server to join room on this socket (idempotent)
            sendJson(websocket, {
                type: "join_room",
                room_id: room.roomId.value,
                peer_id: room.peerId.value ?? undefined,
                peer_label: room.peerLabel.value ?? undefined,
            });
        }

        sendJson(websocket, payload);
        updateStatus(state.channel, "Listening…", "listening");
    };

    const start = async (channel: ConversationChannel): Promise<void> => {
        if (states.has(channel.id)) {
            return;
        }

        updateStatus(
            channel,
            "Connecting to streaming transcription…",
            "processing",
        );

        const sourceStream = await ensureStream(channel);
        if (!sourceStream) {
            throw new Error("No media stream available for transcription");
        }

        const audioTracks = sourceStream.getAudioTracks();
        if (audioTracks.length === 0) {
            throw new Error("Media stream does not contain an audio track");
        }

        const audioStream = cloneAudioStream(sourceStream);
        const state: StreamingState = {
            channel,
            stream: audioStream,
            audioContext: null,
            processor: null,
            websocket: null,
            sequence: 0,
            pending: Promise.resolve(),
            isClosing: false,
            finalChunkSent: false,
            finalResolvers: [],
            cleanedUp: false,
            chunkTimer: null,
            audioBuffer: [],
            sampleRate: 44100, // Default, will be updated when audio context is created
            keepaliveTimer: null,
            lastKeepaliveAt: Date.now(),
            reconnectAttempts: 0,
            reconnectTimer: null,
        };

        states.set(channel.id, state);

        channel.liveTranscript = "";

        try {
            createAudioProcessor(channel, audioStream, state);
        } catch (error) {
            cleanupState(channel.id, state);
            throw error;
        }

        try {
            await connectWebSocket(state);
        } catch (error) {
            console.error(
                "Initial streaming websocket connection failed",
                error,
            );
            scheduleReconnect(state);
            throw error;
        }
    };

    const stop = async (channelId: string): Promise<void> => {
        const state = states.get(channelId);
        if (!state) {
            return;
        }

        state.isClosing = true;

        // Send final chunk with remaining audio buffer
        if (state.audioBuffer.length > 0) {
            processAudioBuffer(state, true);
        } else if (!state.finalChunkSent) {
            // Send empty final chunk
            state.finalChunkSent = true;
            const emptyBlob = new Blob([], { type: "audio/wav" });
            state.pending = state.pending
                .then(() => sendChunk(state, emptyBlob, true))
                .catch((error) => {
                    console.error(
                        "Failed to stream final empty audio chunk",
                        error,
                    );
                    state.finalResolvers.splice(0).forEach((resolve) =>
                        resolve()
                    );
                });
        }

        try {
            await waitForFinalChunk(state);
            await state.pending;
        } catch (error) {
            console.warn("Streaming transcription pending tasks failed", error);
        }

        const socket = state.websocket;
        if (socket && socket.readyState === WebSocket.OPEN) {
            try {
                sendJson(socket, { type: "stop" });
                socket.close();
            } catch (error) {
                console.warn("Failed to close streaming websocket", error);
            }
        }

        cleanupState(channelId, state);
    };

    const dispose = async (channelId: string): Promise<void> => {
        await stop(channelId);
    };

    const stopAll = async (): Promise<void> => {
        const ids = Array.from(states.keys());
        await Promise.all(ids.map((id) => stop(id)));
        states.clear();
    };

    return {
        start,
        stop,
        dispose,
        stopAll,
    };
};
