// Centralized helpers to filter out common hallucinated/garbage transcripts
// Examples observed: single dot "." and subtitle watermarks like "Subtitles made by DimaTorzok"

export const HALLUCINATION_DICTIONARY: readonly string[] = [
  ".",
  "·",
  "•",
  "…",
  "...",
  "—",
  "–",
  "Subtitles made by DimaTorzok",
  "Subtitles by DimaTorzok",
  "DimaTorzok",
  // Add more if we see recurring artifacts
] as const;

// Normalize for comparison: trim, collapse whitespace, lowercase
const normalize = (s: string): string => s
  .replace(/\s+/g, " ")
  .trim()
  .toLowerCase();

export const isHallucinationText = (text: string | undefined | null): boolean => {
  if (!text) return true; // treat empty strings as hallucinations for skip logic
  const t = normalize(text);
  if (!t) return true;
  // Single character punctuation or repeated dots
  if (t === "." || t === "…" || t === "..." || t === "·" || t === "•") return true;
  // Very short token that is only punctuation
  if (t.length <= 2 && /^[^\w\p{L}]+$/u.test(t)) return true;
  return HALLUCINATION_DICTIONARY.map(normalize).some((h) => h === t);
};

// For cases where we have both translated and original, favor dropping the pair if both look bogus
export const sanitizeTranscriptPair = (
  transcript: { text?: string; fullText?: string; },
  original?: string,
): boolean => {
  const main = transcript.fullText?.trim() || transcript.text?.trim() || "";
  const isMainBad = isHallucinationText(main);
  const isOriginalBad = isHallucinationText(original ?? "");
  // If both are bad, skip entirely
  if (isMainBad && isOriginalBad) return false;
  // If main is bad but original is not, still skip from display since we only show translated text
  if (isMainBad) return false;
  return true;
};
