import { ref, type Ref } from 'vue'

interface SpeechRecognitionAlternative {
  readonly transcript: string
  readonly confidence: number
}

interface SpeechRecognitionResult extends ArrayLike<SpeechRecognitionAlternative> {
  readonly isFinal: boolean
  [index: number]: SpeechRecognitionAlternative
}

interface SpeechRecognitionResultList extends ArrayLike<SpeechRecognitionResult> {
  item(index: number): SpeechRecognitionResult
  [index: number]: SpeechRecognitionResult
}

interface RecognitionResultEvent extends Event {
  readonly results: SpeechRecognitionResultList
  readonly resultIndex: number
}

interface RecognitionErrorEvent extends Event {
  readonly error: string
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  onstart: ((this: SpeechRecognitionInstance, ev: Event) => unknown) | null
  onend: ((this: SpeechRecognitionInstance, ev: Event) => unknown) | null
  onerror: ((this: SpeechRecognitionInstance, ev: RecognitionErrorEvent) => unknown) | null
  onresult: ((this: SpeechRecognitionInstance, ev: RecognitionResultEvent) => unknown) | null
  start(): void
  stop(): void
  abort(): void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance

export interface SpeechRecognizerCallbacks {
  onPartial?: (text: string) => void
  onFinal?: (text: string) => void
  onDisconnect?: (reason: 'ended' | 'error', detail?: string) => void
}

export interface SpeechRecognizer {
  isListening: Ref<boolean>
  transcript: Ref<string>
  start: (language: string, callbacks?: SpeechRecognizerCallbacks) => Promise<void>
  stop: () => Promise<string>
  abort: () => void
}

const getSpeechRecognitionCtor = (): SpeechRecognitionConstructor | null => {
  const win = window as typeof window & {
    SpeechRecognition?: SpeechRecognitionConstructor
    webkitSpeechRecognition?: SpeechRecognitionConstructor
  }

  return win.SpeechRecognition || win.webkitSpeechRecognition || null
}

export const isSpeechRecognitionSupported = (): boolean => {
  if (typeof window === 'undefined') {
    return false
  }
  return getSpeechRecognitionCtor() !== null
}

export const createSpeechRecognizer = (): SpeechRecognizer => {
  const isListening = ref<boolean>(false)
  const transcript = ref<string>('')
  let recognition: SpeechRecognitionInstance | null = null

  const start = async (language: string, callbacks?: SpeechRecognizerCallbacks): Promise<void> => {
    return new Promise((resolve, reject) => {
      const Ctor = getSpeechRecognitionCtor()
      if (!Ctor) {
        reject(new Error('Speech recognition not supported in this browser'))
        return
      }

      recognition = new Ctor()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = language

      transcript.value = ''
      let hasStarted = false

      recognition.onstart = () => {
        hasStarted = true
        isListening.value = true
        resolve()
      }

      recognition.onresult = (event: RecognitionResultEvent) => {
        let finalTranscript = ''
        let interimTranscript = ''

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i]
          const text = result[0].transcript
          if (result.isFinal) {
            finalTranscript += text
          } else {
            interimTranscript += text
          }
        }

        const combined = finalTranscript + interimTranscript
        transcript.value = combined
        if (callbacks?.onFinal && finalTranscript.trim()) {
          callbacks.onFinal(finalTranscript.trim())
        }
        if (callbacks?.onPartial) {
          callbacks.onPartial(combined)
        }
      }

      recognition.onend = () => {
        isListening.value = false
        if (hasStarted) {
          callbacks?.onDisconnect?.('ended')
        }
      }

      recognition.onerror = (event: RecognitionErrorEvent) => {
        isListening.value = false
        console.error('Speech recognition error:', event.error)
        const message = `Speech recognition error: ${event.error}`
        if (!hasStarted) {
          reject(new Error(message))
          return
        }
        callbacks?.onDisconnect?.('error', event.error)
      }

      try {
        recognition.start()
      } catch (error: unknown) {
        isListening.value = false
        reject(error instanceof Error ? error : new Error('Failed to start speech recognition'))
      }
    })
  }

  const stop = async (): Promise<string> => {
    return new Promise((resolve) => {
      if (!recognition) {
        resolve(transcript.value)
        return
      }

      recognition.onend = () => {
        isListening.value = false
        resolve(transcript.value)
      }

      recognition.stop()
    })
  }

  const abort = (): void => {
    if (!recognition) return
    recognition.onend = null
    try {
      recognition.abort()
    } catch (error) {
      console.warn('Failed to abort recognition', error)
    }
    isListening.value = false
  }

  return {
    isListening,
    transcript,
    start,
    stop,
    abort
  }
}

export const useSpeechRecognitionSupport = () => {
  const supported = ref<boolean>(isSpeechRecognitionSupported())
  return {
    supported
  }
}