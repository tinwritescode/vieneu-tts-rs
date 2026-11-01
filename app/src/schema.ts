import { z } from "zod";

export const voiceId = z.string().min(1).brand<"VoiceId">();
export type VoiceId = z.infer<typeof voiceId>;

export const voiceInfo = z.object({
  id: voiceId,
  name: z.string(),
  kind: z.enum(["preset", "clone"]),
  gender: z.string(),
  region: z.string(),
  description: z.string(),
});
export type VoiceInfo = z.infer<typeof voiceInfo>;

export const modelStatus = z.object({
  installed: z.boolean(),
  totalBytes: z.number(),
  defaultThreads: z.number(),
  defaultVoice: voiceId,
  version: z.string(),
});
export type ModelStatus = z.infer<typeof modelStatus>;

export const playerStatus = z.object({
  available: z.boolean(),
  transport: z.enum(["idle", "playing", "paused"]),
  position: z.number().nullable(),
});
export type PlayerStatus = z.infer<typeof playerStatus>;

export const jobEvent = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("loading") }),
  z.object({ kind: z.literal("planned"), chunks: z.number() }),
  z.object({ kind: z.literal("chunk"), index: z.number() }),
  z.object({ kind: z.literal("finished"), seconds: z.number(), peaks: z.array(z.number()) }),
  z.object({ kind: z.literal("cancelled") }),
  z.object({ kind: z.literal("auditioned") }),
  z.object({ kind: z.literal("enrolled"), voice: voiceInfo }),
  z.object({ kind: z.literal("failed"), message: z.string() }),
]);
export type JobEvent = z.infer<typeof jobEvent>;

export const installEvent = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("progress"), bytes: z.number() }),
  z.object({ kind: z.literal("finished") }),
  z.object({ kind: z.literal("failed"), message: z.string() }),
]);
export type InstallEvent = z.infer<typeof installEvent>;

export const settings = z.object({
  language: z.enum(["vi", "en"]).catch("vi"),
  threads: z.number().int().min(1).max(32).nullable().catch(null),
  voice: voiceId.nullable().catch(null),
  recent: z.array(voiceId).catch([]),
  speed: z.number().min(0.5).max(2).catch(1),
  pause: z.number().min(0).max(2).catch(0),
  live: z.boolean().catch(false),
});
export type Settings = z.infer<typeof settings>;
