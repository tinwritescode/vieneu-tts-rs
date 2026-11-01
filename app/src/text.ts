import type { VoiceId, VoiceInfo } from "./schema";

const SPOKEN_CHARS_PER_SECOND = 15;
export const RECENT_VOICES = 4;

export const countWords = (text: string): number => text.split(/\s+/u).filter(Boolean).length;

export const estimateSeconds = ({ text, speed }: { text: string; speed: number }): number =>
  text.trim().length / SPOKEN_CHARS_PER_SECOND / speed;

export function formatTime(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Lowercase without Vietnamese accents, so "thu ha" finds "Thu Hà". */
export const fold = (text: string): string =>
  text.normalize("NFD").replace(/\p{M}/gu, "").replaceAll("đ", "d").replaceAll("Đ", "D").toLowerCase();

export type VoiceFilter = { query: string; region: string; gender: string };

export function filterVoices(voices: readonly VoiceInfo[], filter: VoiceFilter): VoiceInfo[] {
  const query = fold(filter.query.trim());
  return voices.filter(
    (voice) =>
      (filter.region === "" || voice.region === filter.region) &&
      (filter.gender === "" || voice.gender === filter.gender) &&
      fold(`${voice.name} ${voice.description}`).includes(query),
  );
}

export const remember = (recent: readonly VoiceId[], voice: VoiceId): VoiceId[] =>
  [voice, ...recent.filter((id) => id !== voice)].slice(0, RECENT_VOICES);

export const distinctValues = (values: readonly string[]): string[] =>
  [...new Set(values.filter((value) => value !== ""))].sort();

export function insertAt(input: { text: string; start: number; end: number; insert: string }): {
  text: string;
  caret: number;
} {
  const { text, start, end, insert } = input;
  const before = text.slice(0, start);
  const spaced = before === "" || /\s$/u.test(before) ? insert : ` ${insert}`;
  return { text: `${before}${spaced}${text.slice(end)}`, caret: start + spaced.length };
}
