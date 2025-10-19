import { ref, reactive, onUnmounted } from 'vue';
import { isHallucinationText } from '../utils/hallucinations';

export interface RoomTranscript {
  roomId: string;
  peerId: string | null;
  peerLabel: string | null;
  channelId?: string | null;
  text: string;
  fullText?: string;
  language?: string | null;
  isFinal?: boolean;
  translations?: Record<string, any>;
  sequence?: number;
  timestamp: string;
}

type RoomStatus = 'idle' | 'connecting' | 'connected' | 'error';

const DEFAULT_WHISPER_STREAMING_ENDPOINT =
  'wss://whisper.obiente.cloud/ws/transcribe';
const WHISPER_STREAMING_ENDPOINT =
  (import.meta.env.VITE_WHISPER_STREAMING_ENDPOINT as string | undefined)?.trim() ||
  DEFAULT_WHISPER_STREAMING_ENDPOINT;

// Module-level singleton state so all consumers share the same room
const _roomId = ref<string | null>(null);
const _peerId = ref<string | null>(null);
const _peerLabel = ref<string | null>(null);
const _status = ref<RoomStatus>('idle');
const _statusMessage = ref('');
const _socket = ref<WebSocket | null>(null);
const _transcripts = reactive<RoomTranscript[]>([]);
const _members = reactive<Array<{ peerId: string | null; peerLabel: string | null; channelId: string | null }>>([]);
let _keepaliveTimer: number | null = null;
let _reconnectTimer: number | null = null;
let _lastKeepaliveAt = 0;
const KEEPALIVE_INTERVAL_MS = 15000;
const KEEPALIVE_TIMEOUT_MS = 45000;
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 15000;
let _reconnectAttempts = 0;

const ROOM_PERSIST_KEY = 'obiente_room_state';

const createPeerId = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 12);

// Restore persisted room session (best-effort)
try {
  const raw = typeof window !== 'undefined' ? localStorage.getItem(ROOM_PERSIST_KEY) : null;
  if (raw) {
    const data = JSON.parse(raw);
    if (data && typeof data === 'object') {
      _roomId.value = typeof data.roomId === 'string' ? data.roomId : null;
      _peerId.value = typeof data.peerId === 'string' ? data.peerId : null;
      _peerLabel.value = typeof data.peerLabel === 'string' ? data.peerLabel : null;
    }
  }
} catch {}

export const useRoomManager = () => {
  const roomId = _roomId;
  const peerId = _peerId;
  const peerLabel = _peerLabel;
  const status = _status;
  const statusMessage = _statusMessage;
  const socket = _socket;
  const transcripts = _transcripts;
  

  const ensureSocket = (): WebSocket => {
    if (socket.value && (socket.value.readyState === WebSocket.OPEN || socket.value.readyState === WebSocket.CONNECTING)) {
      return socket.value;
    }
    const ws = new WebSocket(WHISPER_STREAMING_ENDPOINT);
    socket.value = ws;
    status.value = 'connecting';
    statusMessage.value = 'Connecting to room server…';

    ws.addEventListener('open', () => {
      status.value = 'connected';
      statusMessage.value = 'Connected';
      // reset reconnect attempts
      _reconnectAttempts = 0;
      // start keepalive
      startKeepalive(ws);
      // If we already picked a room, re-join it
      if (roomId.value) {
        // Ensure we have a stable peerId before (re)joining
        if (!peerId.value) {
          peerId.value = createPeerId();
          try { localStorage.setItem(ROOM_PERSIST_KEY, JSON.stringify({ roomId: roomId.value, peerId: peerId.value, peerLabel: peerLabel.value })); } catch {}
        }
        joinRoom(roomId.value, peerId.value, peerLabel.value ?? undefined);
      }
    });

    ws.addEventListener('close', () => {
      if (socket.value === ws) {
        status.value = 'idle';
        statusMessage.value = 'Disconnected';
        stopKeepalive();
        scheduleReconnect();
      }
    });

    ws.addEventListener('error', () => {
      status.value = 'error';
      statusMessage.value = 'Room connection error';
      scheduleReconnect();
    });

    ws.addEventListener('message', (event) => {
      try {
        const payload = JSON.parse(event.data as string);
        if (!payload || typeof payload !== 'object') return;

        if (payload.type === 'pong') {
          _lastKeepaliveAt = Date.now();
          return;
        }

        if (payload.type === 'room_joined') {
          // Ack from server
          roomId.value = payload.room_id || payload.roomId || roomId.value;
          peerId.value = payload.peer_id || payload.peerId || peerId.value;
          peerLabel.value = payload.peer_label || payload.peerLabel || peerLabel.value;
          // Persist
          try {
            localStorage.setItem(ROOM_PERSIST_KEY, JSON.stringify({ roomId: roomId.value, peerId: peerId.value, peerLabel: peerLabel.value }));
          } catch {}
          return;
        }

        if (payload.type === 'room_left') {
          return;
        }

        if (payload.type === 'room_transcript' || (payload.type === 'transcript' && (payload.room_id || payload.roomId))) {
          // Ignore any room-bound message that originated from our own peer_id to avoid self-echo duplicates
          const pid = payload.peer_id || payload.peerId || null;
          if (typeof pid === 'string' && _peerId.value && pid === _peerId.value) {
            return;
          }
          const candidate = String(payload.fullText || payload.text || '').trim();
          if (!candidate || isHallucinationText(candidate)) {
            return;
          }
          const entry: RoomTranscript = {
            roomId: payload.room_id || payload.roomId,
            peerId: payload.peer_id || payload.peerId || null,
            peerLabel: payload.peer_label || payload.peerLabel || null,
            channelId: payload.channel_id || payload.channelId || null,
            text: typeof payload.text === 'string' ? payload.text : '',
            fullText: typeof payload.fullText === 'string' ? payload.fullText : undefined,
            language: typeof payload.language === 'string' ? payload.language : null,
            isFinal: Boolean(payload.isFinal),
            translations: (payload.translations && typeof payload.translations === 'object') ? payload.translations : undefined,
            sequence: typeof payload.sequence === 'number' ? payload.sequence : undefined,
            timestamp: new Date().toISOString(),
          };
          transcripts.push(entry);
          return;
        }

        if (payload.type === 'room_roster') {
          const list = Array.isArray(payload.members) ? payload.members : [];
          _members.splice(0, _members.length, ...list.map((m: any) => ({
            peerId: typeof m?.peer_id === 'string' ? m.peer_id : (typeof m?.peerId === 'string' ? m.peerId : null),
            peerLabel: typeof m?.peer_label === 'string' ? m.peer_label : (typeof m?.peerLabel === 'string' ? m.peerLabel : null),
            channelId: typeof m?.channel_id === 'string' ? m.channel_id : (typeof m?.channelId === 'string' ? m.channelId : null),
          })));
          return;
        }
      } catch (e) {
        // ignore
      }
    });

    return ws;
  };

  const startKeepalive = (ws: WebSocket) => {
    stopKeepalive();
    _lastKeepaliveAt = Date.now();
    _keepaliveTimer = window.setInterval(() => {
      if (!socket.value || socket.value !== ws) return;
      if (ws.readyState !== WebSocket.OPEN) return;
      const now = Date.now();
      // timeout detection
      if (now - _lastKeepaliveAt > KEEPALIVE_TIMEOUT_MS) {
        try { ws.close(); } catch {}
        return;
      }
      try { ws.send(JSON.stringify({ type: 'ping', ts: now })); } catch {}
    }, KEEPALIVE_INTERVAL_MS) as unknown as number;
  };

  const stopKeepalive = () => {
    if (_keepaliveTimer !== null) {
      clearInterval(_keepaliveTimer);
      _keepaliveTimer = null;
    }
  };

  const scheduleReconnect = () => {
    // Only auto-reconnect when a room is set; otherwise, keep idle until explicitly used
    if (!roomId.value) return;
    if (_reconnectTimer !== null) return;
    const delay = Math.min(RECONNECT_BASE_DELAY_MS * Math.pow(2, _reconnectAttempts), RECONNECT_MAX_DELAY_MS);
    _reconnectAttempts += 1;
    _reconnectTimer = window.setTimeout(() => {
      _reconnectTimer = null;
      try { ensureSocket(); } catch {}
    }, delay) as unknown as number;
  };

  const sendJson = (data: unknown) => {
    const ws = ensureSocket();
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    } else {
      ws.addEventListener('open', () => ws.send(JSON.stringify(data)), { once: true });
    }
  };

  const createRoom = async (label?: string): Promise<string> => {
    const id = generateRoomId();
    await joinRoom(id, undefined, label);
    return id;
  };

  const joinRoom = async (id: string, pid?: string, plabel?: string): Promise<void> => {
    roomId.value = id;
    // Ensure a stable peerId exists
    if (!pid && !peerId.value) {
      pid = createPeerId();
    }
    if (pid) peerId.value = pid;
    if (plabel) peerLabel.value = plabel;
    sendJson({ type: 'join_room', room_id: id, peer_id: peerId.value, peer_label: peerLabel.value });
    try { localStorage.setItem(ROOM_PERSIST_KEY, JSON.stringify({ roomId: roomId.value, peerId: peerId.value, peerLabel: peerLabel.value })); } catch {}
  };

  const leaveRoom = (): void => {
    if (roomId.value) {
      try { sendJson({ type: 'leave_room' }); } catch {}
    }
    roomId.value = null;
    transcripts.length = 0;
    _members.splice(0, _members.length);
    try { localStorage.removeItem(ROOM_PERSIST_KEY); } catch {}
  };

  const annotateLocalTranscript = (
    payload: Omit<RoomTranscript, 'timestamp' | 'roomId'>,
  ): void => {
    if (!roomId.value) return;
    const entry: RoomTranscript = {
      ...payload,
      roomId: roomId.value,
      timestamp: new Date().toISOString(),
    } as RoomTranscript;
    transcripts.push(entry);
  };

  const generateRoomId = (): string => Math.random().toString(36).slice(2, 8).toUpperCase();

  // Do not force-close the socket on component unmount; keep room persistent across views
  onUnmounted(() => {
    // No-op; consumers can call leaveRoom explicitly if needed
    stopKeepalive();
    if (_reconnectTimer !== null) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
  });

  // If a persisted roomId exists, ensure the socket connects immediately (after helpers exist)
  if (typeof window !== 'undefined' && roomId.value && !_socket.value) {
    try { /* fire-and-forget */ ensureSocket(); } catch {}
  }

  return {
    roomId,
    peerId,
    peerLabel,
    status,
    statusMessage,
    transcripts,
    members: _members,
    isInRoom: () => !!roomId.value,
    isConnected: () => status.value === 'connected' && !!roomId.value,
    ensureSocket, // expose for transcriber to reuse
    createRoom,
    joinRoom,
    leaveRoom,
    annotateLocalTranscript,
  };
};