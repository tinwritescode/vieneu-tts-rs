import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { z } from "zod";
import { installEvent, jobEvent, modelStatus, playerStatus, voiceInfo, type VoiceId } from "./schema";

export const getModelStatus = async () => modelStatus.parse(await invoke("model_status"));
export const listVoices = async () => z.array(voiceInfo).parse(await invoke("list_voices"));
export const getPlayerStatus = async () => playerStatus.parse(await invoke("player_status"));
export const importDocument = async (path: string) => z.string().parse(await invoke("import_document", { path }));

export const installModels = () => invoke<void>("install_models");
export const cancelJob = () => invoke<void>("cancel_job");
export const playerPlay = (from: number) => invoke<void>("player_play", { from });
export const playerPause = (paused: boolean) => invoke<void>("player_pause", { paused });
export const playerStop = () => invoke<void>("player_stop");
export const exportTake = (destination: string) => invoke<void>("export_take", { destination });
export const removeVoice = (id: VoiceId) => invoke<void>("remove_voice", { id });
export const audition = (args: { voice: VoiceId; threads: number }) => invoke<void>("audition", args);

export const speak = (args: {
  text: string; voice: VoiceId; speed: number; pause: number; live: boolean; threads: number;
}) => invoke<void>("speak", { args });

export const enrollVoice = (args: { name: string; clip: string; denoise: boolean; threads: number }) =>
  invoke<void>("enroll_voice", args);

export const onJob = (handler: (event: z.infer<typeof jobEvent>) => void) =>
  listen("job", ({ payload }) => handler(jobEvent.parse(payload)));

export const onInstall = (handler: (event: z.infer<typeof installEvent>) => void) =>
  listen("install", ({ payload }) => handler(installEvent.parse(payload)));
