import axios from 'axios'

type LanguageCode = string

interface TranslationResult {
  translatedText: string
  detectedLanguage: LanguageCode | null
  alternatives: string[]
  raw: LibreTranslateResponse
}

interface LanguageOption {
  code: LanguageCode
  name: string
}

interface LibreTranslateResponse {
  translatedText: string
  detectedLanguage?: string | null
  alternatives?: string[]
}

interface LibreLanguageEntry {
  code: string
  name: string
}

interface LibreDetectResponse {
  language: string
  confidence: number
}

const DEFAULT_LIBRE_BASE = 'https://libretranslate.obiente.cloud'
const DEFAULT_ALTERNATIVES = 3

const libreBase = (import.meta.env.VITE_LIBRE_TRANSLATE_BASE as string | undefined)?.trim() || DEFAULT_LIBRE_BASE
const rawAlternativesRequested = Number((import.meta.env.VITE_LIBRE_TRANSLATE_ALTERNATIVES as string | undefined) ?? DEFAULT_ALTERNATIVES)

const clampAlternativeLimit = (value: number | undefined): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 0
  }

  return Math.max(0, Math.min(5, Math.trunc(value)))
}

const defaultAlternativesRequested = clampAlternativeLimit(rawAlternativesRequested)

const libreClient = axios.create({
  baseURL: libreBase,
  headers: {
    'Content-Type': 'application/json'
  }
})

const FALLBACK_LANGUAGES: LanguageOption[] = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese (Simplified)' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
  { code: 'nl', name: 'Dutch' },
  { code: 'sv', name: 'Swedish' },
  { code: 'pl', name: 'Polish' }
]

export function useTranslationService() {
  const translateText = async (
    text: string,
    sourceLanguage: LanguageCode = 'auto',
    targetLanguage: LanguageCode = 'en',
    alternativeLimit?: number
  ): Promise<TranslationResult> => {
    if (!text.trim()) {
      const defaultLang = sourceLanguage === 'auto' ? null : sourceLanguage
      return {
        translatedText: '',
        detectedLanguage: defaultLang,
        alternatives: [],
        raw: { translatedText: '', detectedLanguage: defaultLang, alternatives: [] }
      }
    }

    try {
      const payload: Record<string, unknown> = {
        q: text,
        source: sourceLanguage,
        target: targetLanguage,
        format: 'text'
      }

      const requestedFromParam = clampAlternativeLimit(alternativeLimit)
      const fallbackAlternatives = defaultAlternativesRequested
      const effectiveAlternatives = requestedFromParam > 0 ? requestedFromParam : fallbackAlternatives

      if (effectiveAlternatives > 0) {
        payload.alternatives = effectiveAlternatives
      }

      const response = await libreClient.post<LibreTranslateResponse>('/translate', payload)

      const translatedText = response.data?.translatedText ?? ''
      const alternatives = Array.isArray(response.data?.alternatives)
        ? response.data.alternatives.filter((value): value is string => typeof value === 'string')
        : []
      const detectedLanguage = sourceLanguage === 'auto'
        ? (typeof response.data?.detectedLanguage === 'string' ? response.data.detectedLanguage : null)
        : sourceLanguage

      return {
        translatedText,
        detectedLanguage,
        alternatives,
        raw: response.data
      }
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data ?? error.message
        throw new Error(`Translation service error: ${JSON.stringify(message)}`)
      }

      throw new Error('Failed to translate text. Please try again.')
    }
  }

  const detectLanguage = async (text: string): Promise<LanguageCode | null> => {
    if (!text.trim()) {
      return null
    }

    try {
      const response = await libreClient.post<LibreDetectResponse[]>('/detect', {
        q: text
      })

      const bestMatch = Array.isArray(response.data) ? response.data[0] : null
      return bestMatch?.language ?? null
    } catch (error: unknown) {
      console.error('Language detection error:', error)
      return null
    }
  }

  const getSupportedLanguages = async (): Promise<LanguageOption[]> => {
    try {
      const response = await libreClient.get<LibreLanguageEntry[]>('/languages')
      if (!Array.isArray(response.data)) {
        return FALLBACK_LANGUAGES
      }
      return response.data.map((entry) => ({ code: entry.code, name: entry.name }))
    } catch (error) {
      console.warn('Failed to fetch supported languages from LibreTranslate', error)
      return FALLBACK_LANGUAGES
    }
  }

  return {
    translateText,
    detectLanguage,
    getSupportedLanguages
  }
}