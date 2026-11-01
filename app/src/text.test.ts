import { describe, expect, it } from "vitest";
import { modelStatus, settings, voiceId, type VoiceId, type VoiceInfo } from "./schema";
import { countWords, distinctValues, estimateSeconds, filterVoices, fold, formatTime, insertAt, remember } from "./text";

const id = (value: string): VoiceId => voiceId.parse(value);
const voice = (name: string, region: string, gender: string): VoiceInfo => ({
  id: id(name), name, kind: "preset", region, gender, description: "",
});

describe("text helpers", () => {
  it("counts words across any whitespace", () => {
    expect(countWords("  một hai\n\tba ")).toBe(3);
    expect(countWords("   ")).toBe(0);
  });

  it("estimates duration from trimmed length and speed", () => {
    expect(estimateSeconds({ text: ` ${"a".repeat(30)} `, speed: 2 })).toBe(1);
  });

  it("formats minutes and seconds", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(65.4)).toBe("1:05");
    expect(formatTime(59.6)).toBe("1:00");
    expect(formatTime(-3)).toBe("0:00");
  });

  it("folds Vietnamese accents and đ", () => {
    expect(fold("Đoàn Thu Hà")).toBe("doan thu ha");
  });

  it("filters by accent-insensitive query, region and gender", () => {
    const voices = [voice("Thu Hà", "Bắc", "Nữ"), voice("Nam", "Nam", "Nam")];
    const names = (query: string, region: string, gender: string) =>
      filterVoices(voices, { query, region, gender }).map((v) => v.name);
    expect(names("thu ha", "", "")).toEqual(["Thu Hà"]);
    expect(names("", "Nam", "")).toEqual(["Nam"]);
    expect(names("", "", "Nữ")).toEqual(["Thu Hà"]);
    expect(names("thu", "Nam", "")).toEqual([]);
  });

  it("moves a voice to the front of recents without repeats, capped", () => {
    const recent = ["a", "b", "c", "d"].map(id);
    expect(remember(recent, id("c"))).toEqual(["c", "a", "b", "d"].map(id));
    expect(remember(recent, id("e"))).toEqual(["e", "a", "b", "c"].map(id));
  });

  it("drops blanks and repeats from filter values", () => {
    expect(distinctValues(["b", "a", "", "b"])).toEqual(["a", "b"]);
  });

  it("inserts over a selection and spaces after a word", () => {
    expect(insertAt({ text: "ab cd", start: 2, end: 2, insert: "[x]" })).toEqual({ text: "ab [x] cd", caret: 6 });
    expect(insertAt({ text: "ab ", start: 3, end: 3, insert: "[x]" })).toEqual({ text: "ab [x]", caret: 6 });
    expect(insertAt({ text: "abc", start: 0, end: 3, insert: "[x]" })).toEqual({ text: "[x]", caret: 3 });
  });
});

describe("boundary schemas", () => {
  it("falls back field by field on damaged stored settings", () => {
    expect(settings.parse({ language: "fr", speed: 9, pause: 0.5, recent: "x", live: 1, threads: 0, voice: "" })).toEqual({
      language: "vi", threads: null, voice: null, recent: [], speed: 1, pause: 0.5, live: false,
    });
  });

  it("rejects a malformed backend reply", () => {
    expect(modelStatus.safeParse({ installed: "yes" }).success).toBe(false);
  });
});
