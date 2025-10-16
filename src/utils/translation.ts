import type { TranslationEntry } from "../types/conversation";

export const normalizeTranslationMap = (
  input: unknown,
): Record<string, TranslationEntry> => {
  if (!input || typeof input !== "object") {
    return {};
  }

  const normalizeAlternative = (value: unknown): string | null => {
    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed.length > 0 ? trimmed : null;
    }
    if (value && typeof value === "object") {
      const candidate = (value as Record<string, unknown>).translatedText ??
        (value as Record<string, unknown>).translation ??
        (value as Record<string, unknown>).text ??
        (value as Record<string, unknown>).value;
      if (typeof candidate === "string") {
        const trimmed = candidate.trim();
        return trimmed.length > 0 ? trimmed : null;
      }
    }
    return null;
  };

  const entries = Object.entries(input as Record<string, unknown>);
  return entries.reduce<Record<string, TranslationEntry>>(
    (acc, [code, value]) => {
      if (typeof value === "string") {
        const primary = value.trim();
        if (primary.length > 0) {
          acc[code] = { primary, alternatives: [] };
        }
        return acc;
      }

      if (value && typeof value === "object") {
        const primaryValue = (value as { primary?: unknown }).primary;
        const alternativesValue =
          (value as { alternatives?: unknown }).alternatives;

        let primary = typeof primaryValue === "string"
          ? primaryValue.trim()
          : "";
        const alternatives = Array.isArray(alternativesValue)
          ? alternativesValue
            .map((entry) => normalizeAlternative(entry))
            .filter((entry): entry is string => entry !== null)
          : [];

        const uniqueAlternatives = alternatives
          .filter((option) => option.length > 0)
          .filter((entry, index, arr) =>
            arr.indexOf(entry) === index && entry !== primary
          );

        const normalizedAlternatives = !primary && uniqueAlternatives.length > 0
          ? uniqueAlternatives.slice(1)
          : uniqueAlternatives;

        if (!primary && uniqueAlternatives.length > 0) {
          primary = uniqueAlternatives[0];
        }

        if (primary.length > 0 || normalizedAlternatives.length > 0) {
          acc[code] = {
            primary,
            alternatives: normalizedAlternatives,
          };
        }
      }

      return acc;
    },
    {},
  );
};
