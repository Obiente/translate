import { onMounted, onUnmounted, ref, type Ref } from 'vue'

const MAX_TTS_SEGMENT_LENGTH = 200

const LANGUAGE_FALLBACKS: Record<string, string> = {
  en: 'en',
  'en-us': 'en',
  'en-gb': 'en',
  es: 'es',
  'es-es': 'es',
  fr: 'fr',
  de: 'de',
  it: 'it',
  pt: 'pt',
  'pt-br': 'pt-BR',
  'pt-pt': 'pt-PT',
  ru: 'ru',
  ja: 'ja',
  ko: 'ko',
  zh: 'zh-CN',
  'zh-cn': 'zh-CN',
  'zh-tw': 'zh-TW',
  ar: 'ar',
  hi: 'hi',
  nl: 'nl',
  sv: 'sv',
  pl: 'pl'
}

const chunkText = (text: string, maxLength: number): string[] => {
  const words = text.trim().split(/\s+/)
  const segments: string[] = []
  let current = ''

  for (const word of words) {
    const potential = current ? `${current} ${word}` : word
    if (potential.length > maxLength && current) {
      segments.push(current)
      current = word
    } else {
      current = potential
    }
  }

  if (current) {
    segments.push(current)
  }

  return segments
}

const normaliseLanguage = (language: string): string => {
  if (!language) {
    return 'en'
  }

  const lower = language.toLowerCase()
  if (LANGUAGE_FALLBACKS[lower]) {
    return LANGUAGE_FALLBACKS[lower]
  }

  const base = lower.split('-')[0]
  return LANGUAGE_FALLBACKS[base] ?? base
}

export function useSpeechSynthesis() {
  const isSpeaking: Ref<boolean> = ref(false)
  const synthesis: SpeechSynthesis | null =
    typeof window !== 'undefined' && 'speechSynthesis' in window
      ? window.speechSynthesis
      : null
  const voices = ref<SpeechSynthesisVoice[]>([])

  const updateVoices = (): void => {
    if (!synthesis) {
      return
    }
    const available = synthesis.getVoices()
    if (available && available.length) {
      voices.value = available
    }
  }

  const handleVoicesChanged = (): void => {
    updateVoices()
  }

  onMounted(() => {
    updateVoices()
    if (!synthesis) {
      return
    }
    if (typeof synthesis.addEventListener === 'function') {
      synthesis.addEventListener('voiceschanged', handleVoicesChanged)
    } else {
      synthesis.onvoiceschanged = handleVoicesChanged
    }
  })

  onUnmounted(() => {
    if (!synthesis) {
      return
    }
    if (typeof synthesis.removeEventListener === 'function') {
      synthesis.removeEventListener('voiceschanged', handleVoicesChanged)
    } else if (synthesis.onvoiceschanged === handleVoicesChanged) {
      synthesis.onvoiceschanged = null
    }
  })

  const cancelPlayback = (): void => {
    if (!synthesis) {
      return
    }
    synthesis.cancel()
    isSpeaking.value = false
  }

  const resolveVoice = (language: string): SpeechSynthesisVoice | null => {
    const normalised = normaliseLanguage(language)
    const lower = normalised.toLowerCase()
    const directMatch = voices.value.find(voice => voice.lang?.toLowerCase() === lower)
    if (directMatch) {
      return directMatch
    }

    const base = lower.split('-')[0]
    return (
      voices.value.find(voice => voice.lang?.toLowerCase() === base) ??
      voices.value.find(voice => voice.lang?.toLowerCase().startsWith(base)) ??
      null
    )
  }

  const speak = async (text: string, language: string = 'en'): Promise<void> => {
    const trimmed = text.trim()
    if (!trimmed) {
      return
    }

    if (!synthesis) {
      console.warn('Speech synthesis API is unavailable in this environment')
      return
    }

    cancelPlayback()

    const segments = chunkText(trimmed, MAX_TTS_SEGMENT_LENGTH)
    if (segments.length === 0) {
      return
    }

    const voice = resolveVoice(language)
    let remaining = segments.length
    isSpeaking.value = true

    for (const segment of segments) {
      const utterance = new SpeechSynthesisUtterance(segment)
      utterance.lang = normaliseLanguage(language)
      if (voice) {
        utterance.voice = voice
      }

      utterance.onend = () => {
        remaining -= 1
        if (remaining === 0) {
          isSpeaking.value = false
        }
      }

      utterance.onerror = event => {
        console.error('Local speech synthesis failed', event)
        cancelPlayback()
      }

      synthesis.speak(utterance)
    }
  }

  const stopSpeaking = (): void => {
    cancelPlayback()
  }

  return {
    isSpeaking,
    speak,
    stopSpeaking
  }
}