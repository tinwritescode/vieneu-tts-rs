import { ask, open, save } from "@tauri-apps/plugin-dialog";
import * as api from "./api";
import { applyLanguage, t, type Language } from "./i18n";
import { settings, type JobEvent, type ModelStatus, type Settings, type VoiceId, type VoiceInfo } from "./schema";
import { countWords, distinctValues, estimateSeconds, filterVoices, formatTime, insertAt, remember } from "./text";

const SETTINGS_KEY = "settings";
const POLL_MS = 100;
const DOCUMENT_EXTENSIONS = ["txt", "md", "docx", "pdf", "srt"];
const AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "m4a"];

type Activity =
  | { kind: "idle" }
  | { kind: "speaking" }
  | { kind: "auditioning" }
  | { kind: "enrolling" };

type Take = { seconds: number; peaks: number[] };
type Source = "take" | "audition";

function need<T extends HTMLElement>(id: string, type: new () => T): T {
  const element = document.getElementById(id);
  if (element instanceof type) return element;
  throw new Error(`index.html has no ${type.name} #${id}`);
}

const ui = {
  text: need("text", HTMLTextAreaElement), counts: need("counts", HTMLElement),
  generate: need("generate", HTMLButtonElement), play: need("play", HTMLButtonElement),
  stop: need("stop", HTMLButtonElement), export: need("export", HTMLButtonElement),
  audition: need("audition", HTMLButtonElement), wave: need("wave", HTMLCanvasElement),
  clock: need("clock", HTMLElement), status: need("status", HTMLElement), version: need("version", HTMLElement),
  avatar: need("avatar", HTMLElement), voiceName: need("voice-name", HTMLElement),
  voiceMeta: need("voice-meta", HTMLElement), recent: need("recent", HTMLElement),
  speed: need("speed", HTMLInputElement), speedOut: need("speed-out", HTMLOutputElement),
  pause: need("pause", HTMLInputElement), pauseOut: need("pause-out", HTMLOutputElement),
  live: need("live", HTMLInputElement), grid: need("voice-grid", HTMLElement),
  search: need("search", HTMLInputElement), region: need("region", HTMLSelectElement),
  gender: need("gender", HTMLSelectElement), cloneName: need("clone-name", HTMLInputElement),
  cloneClip: need("clone-clip", HTMLButtonElement), cloneFile: need("clone-file", HTMLElement),
  cloneDenoise: need("clone-denoise", HTMLInputElement), cloneGo: need("clone-go", HTMLButtonElement),
  language: need("language", HTMLSelectElement), threads: need("threads", HTMLInputElement),
  setup: need("setup", HTMLElement), setupText: need("setup-text", HTMLElement),
  setupProgress: need("setup-progress", HTMLProgressElement), setupGo: need("setup-go", HTMLButtonElement),
  import: need("import", HTMLButtonElement),
};

function loadSettings(): Settings {
  let stored: unknown = {};
  try {
    stored = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}");
  } catch {
    // Damaged settings fall back to defaults.
  }
  return settings.parse(stored);
}

let config = loadSettings();
let model: ModelStatus;
let voices: VoiceInfo[] = [];
let activity: Activity = { kind: "idle" };
let take: Take | null = null;
let source: Source = "take";
let transport: "idle" | "playing" | "paused" = "idle";
let position = 0;
let polling = false;
let chunkTotal = 0;
let clonePath: string | null = null;

const threads = (): number => config.threads ?? model.defaultThreads;
const currentVoice = (): VoiceId => config.voice ?? model.defaultVoice;

function update(change: Partial<Settings>): void {
  config = { ...config, ...change };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(config));
}

function setStatus(message: string, isError = false): void {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", isError);
}

const reportFailure = (error: unknown): void => {
  activity = { kind: "idle" };
  setStatus(t("failed", { message: String(error) }), true);
  refreshControls();
};

/** Run a UI action, surfacing any failure in the status bar. */
const guarded = (action: () => Promise<void>) => (): void => {
  action().catch(reportFailure);
};

function refreshControls(): void {
  const busy = activity.kind !== "idle";
  const hasText = ui.text.value.trim() !== "";
  ui.generate.disabled = busy || !hasText;
  ui.audition.disabled = busy;
  ui.import.disabled = busy;
  ui.cloneGo.disabled = busy || clonePath === null || ui.cloneName.value.trim() === "";
  ui.play.disabled = activity.kind === "enrolling" || (take === null && transport === "idle");
  ui.stop.disabled = !busy && transport === "idle";
  ui.export.disabled = take === null || activity.kind === "speaking";
  ui.play.textContent = t(transport === "playing" ? "pauseAudio" : transport === "paused" ? "resume" : "play");
}

function refreshCounts(): void {
  const text = ui.text.value;
  ui.counts.textContent = t("counts", {
    words: countWords(text),
    chars: text.length,
    time: formatTime(estimateSeconds({ text, speed: config.speed })),
  });
  refreshControls();
}

function drawWave(): void {
  const canvas = ui.wave;
  const scale = window.devicePixelRatio;
  canvas.width = canvas.clientWidth * scale;
  canvas.height = canvas.clientHeight * scale;
  const context = canvas.getContext("2d");
  if (context === null) return;
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  const seconds = take?.seconds ?? 0;
  const shown = source === "take" ? position : 0;
  ui.clock.textContent = `${formatTime(shown)} / ${formatTime(seconds)}`;
  if (take === null) return;
  const step = 3 * scale;
  const bars = Math.floor(width / step);
  const playhead = seconds > 0 ? shown / seconds : 0;
  for (let bar = 0; bar < bars; bar += 1) {
    const peak = take.peaks[Math.floor((bar / bars) * take.peaks.length)] ?? 0;
    const tall = Math.max(scale, peak * height);
    context.fillStyle = bar / bars < playhead ? "#2dd4bf" : "#3a4356";
    context.fillRect(bar * step, (height - tall) / 2, 2 * scale, tall);
  }
}

async function poll(): Promise<void> {
  const status = await api.getPlayerStatus();
  transport = status.transport;
  if (status.position !== null) position = status.position;
  if (transport === "idle" && activity.kind !== "speaking") polling = false;
  if (transport === "idle" && source === "take") position = 0;
  drawWave();
  refreshControls();
  if (polling) setTimeout(guarded(poll), POLL_MS);
}

function startPolling(): void {
  if (polling) return;
  polling = true;
  void poll().catch(reportFailure);
}

function initials(name: string): string {
  return name.trim().slice(0, 1).toUpperCase();
}

function genderLabel(gender: string): string {
  if (gender === "male") return t("male");
  return gender === "female" ? t("female") : gender;
}

const describeVoice = (voice: VoiceInfo): string =>
  [genderLabel(voice.gender), voice.region].filter((part) => part !== "").join(" · ");

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  fill: { className?: string; text?: string; onClick?: () => void },
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (fill.className !== undefined) node.className = fill.className;
  if (fill.text !== undefined) node.textContent = fill.text;
  if (fill.onClick !== undefined) node.addEventListener("click", fill.onClick);
  return node;
}

function selectVoice(id: VoiceId): void {
  update({ voice: id, recent: remember(config.recent, id) });
  renderVoice();
  renderGrid();
}

function renderVoice(): void {
  const voice = voices.find((item) => item.id === currentVoice());
  if (voice === undefined) {
    // A stored clone may have been deleted outside the app.
    if (config.voice !== null) {
      update({ voice: null });
      renderVoice();
    }
    return;
  }
  ui.avatar.textContent = initials(voice.name);
  ui.voiceName.textContent = voice.name;
  ui.voiceMeta.textContent = describeVoice(voice);
  ui.recent.replaceChildren(
    ...config.recent.flatMap((id) => {
      const recent = voices.find((item) => item.id === id);
      if (recent === undefined || id === voice.id) return [];
      return [element("button", { className: "chip", text: recent.name, onClick: () => selectVoice(id) })];
    }),
  );
}

function renderGrid(): void {
  const shown = filterVoices(voices, { query: ui.search.value, region: ui.region.value, gender: ui.gender.value });
  ui.grid.replaceChildren(...shown.map(voiceCard));
}

function voiceCard(voice: VoiceInfo): HTMLElement {
  const selected = voice.id === currentVoice();
  const card = element("div", { className: `card voice${selected ? " selected" : ""}` });
  const head = element("div", { className: "voice-head" });
  const about = element("div", {});
  about.append(element("strong", { text: voice.name }), element("small", { text: voice.description || describeVoice(voice) }));
  head.append(element("span", { className: "avatar", text: initials(voice.name) }), about);
  const actions = element("div", { className: "row" });
  actions.append(
    element("button", { className: "ghost", text: t("useVoice"), onClick: () => selectVoice(voice.id) }),
    element("button", { className: "ghost", text: t("listen"), onClick: guarded(() => startAudition(voice.id)) }),
  );
  if (voice.kind === "clone") {
    actions.append(element("button", { className: "link danger", text: t("remove"), onClick: guarded(() => deleteVoice(voice)) }));
  }
  card.append(head, actions);
  return card;
}

function fillFilter(input: { select: HTMLSelectElement; all: string; values: string[]; label?: (value: string) => string }): void {
  const { select, all, values, label = (value) => value } = input;
  const chosen = select.value;
  select.replaceChildren(new Option(all, ""), ...values.map((value) => new Option(label(value), value)));
  select.value = values.includes(chosen) ? chosen : "";
}

async function reloadVoices(): Promise<void> {
  voices = await api.listVoices();
  fillFilter({ select: ui.region, all: t("allRegions"), values: distinctValues(voices.map((voice) => voice.region)) });
  fillFilter({
    select: ui.gender, all: t("allGenders"), values: distinctValues(voices.map((voice) => voice.gender)), label: genderLabel,
  });
  renderVoice();
  renderGrid();
}

async function deleteVoice(voice: VoiceInfo): Promise<void> {
  if (!(await ask(t("confirmRemove", { name: voice.name }), { kind: "warning" }))) return;
  await api.removeVoice(voice.id);
  await reloadVoices();
}

async function startAudition(voice: VoiceId): Promise<void> {
  if (activity.kind !== "idle") return;
  activity = { kind: "auditioning" };
  source = "audition";
  setStatus(t("auditioning"));
  refreshControls();
  await api.audition({ voice, threads: threads() });
}

async function generate(): Promise<void> {
  const text = ui.text.value;
  activity = { kind: "speaking" };
  source = "take";
  take = null;
  position = 0;
  chunkTotal = 0;
  drawWave();
  setStatus(t("loading"));
  refreshControls();
  await api.speak({
    text, voice: currentVoice(), speed: config.speed, pause: config.pause, live: config.live, threads: threads(),
  });
  startPolling();
}

async function togglePlay(): Promise<void> {
  if (transport === "idle") {
    source = "take";
    await api.playerPlay(position);
  } else {
    await api.playerPause(transport === "playing");
  }
  startPolling();
}

async function stopAll(): Promise<void> {
  if (activity.kind !== "idle") await api.cancelJob();
  await api.playerStop();
  position = 0;
  startPolling();
}

async function seek(event: MouseEvent): Promise<void> {
  if (take === null || activity.kind === "speaking") return;
  source = "take";
  position = (event.offsetX / ui.wave.clientWidth) * take.seconds;
  await api.playerPlay(position);
  startPolling();
}

async function exportWav(): Promise<void> {
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/gu, "");
  const destination = await save({
    defaultPath: `vieneu-${stamp}.wav`,
    filters: [{ name: "WAV", extensions: ["wav"] }],
  });
  if (destination === null) return;
  await api.exportTake(destination);
  setStatus(t("exported", { path: destination }));
}

async function importFile(): Promise<void> {
  const path = await open({ filters: [{ name: t("documents"), extensions: DOCUMENT_EXTENSIONS }] });
  if (path === null) return;
  ui.text.value = await api.importDocument(path);
  refreshCounts();
}

async function chooseClip(): Promise<void> {
  const path = await open({ filters: [{ name: t("audioFiles"), extensions: AUDIO_EXTENSIONS }] });
  if (path === null) return;
  clonePath = path;
  ui.cloneFile.textContent = path;
  refreshControls();
}

async function enroll(): Promise<void> {
  if (clonePath === null) return;
  activity = { kind: "enrolling" };
  setStatus(t("cloning"));
  refreshControls();
  await api.enrollVoice({
    name: ui.cloneName.value.trim(), clip: clonePath, denoise: ui.cloneDenoise.checked, threads: threads(),
  });
}

function onJob(event: JobEvent): void {
  switch (event.kind) {
    case "loading":
      setStatus(t("loading"));
      return;
    case "planned":
      chunkTotal = event.chunks;
      return;
    case "chunk":
      if (activity.kind === "speaking") setStatus(t("planned", { index: event.index + 1, total: chunkTotal }));
      return;
    case "finished":
      take = { seconds: event.seconds, peaks: event.peaks };
      setStatus(t("finished", { time: formatTime(event.seconds) }));
      break;
    case "cancelled":
      setStatus(t("cancelled"));
      break;
    case "auditioned":
      setStatus(t("ready"));
      break;
    case "enrolled":
      setStatus(t("cloned", { name: event.voice.name }));
      ui.cloneName.value = "";
      ui.cloneFile.textContent = "";
      clonePath = null;
      reloadVoices().then(() => selectVoice(event.voice.id), reportFailure);
      break;
    case "failed":
      setStatus(t("failed", { message: event.message }), true);
      break;
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
  activity = { kind: "idle" };
  startPolling();
  drawWave();
  refreshControls();
}

function showView(view: string): void {
  for (const section of document.querySelectorAll<HTMLElement>(".view")) {
    section.hidden = section.id !== `view-${view}`;
  }
  for (const button of document.querySelectorAll<HTMLElement>(".nav")) {
    button.classList.toggle("active", button.dataset["view"] === view);
  }
  if (view === "create") drawWave();
}

const isLanguage = (value: string): value is Language => value === "vi" || value === "en";

function retranslate(): void {
  applyLanguage(config.language);
  setStatus(t("ready"));
  refreshCounts();
  void reloadVoices().catch(reportFailure);
}

function renderSliders(): void {
  ui.speedOut.textContent = `${config.speed.toFixed(2)}×`;
  ui.pauseOut.textContent = `${config.pause.toFixed(2)} s`;
}

function bind(): void {
  for (const button of document.querySelectorAll<HTMLElement>("[data-view]")) {
    button.addEventListener("click", () => showView(button.dataset["view"] ?? "create"));
  }
  for (const chip of document.querySelectorAll<HTMLElement>("[data-cue]")) {
    chip.addEventListener("click", () => {
      const { value, selectionStart, selectionEnd } = ui.text;
      const edit = insertAt({ text: value, start: selectionStart, end: selectionEnd, insert: chip.dataset["cue"] ?? "" });
      ui.text.value = edit.text;
      ui.text.focus();
      ui.text.setSelectionRange(edit.caret, edit.caret);
      refreshCounts();
    });
  }
  ui.text.addEventListener("input", refreshCounts);
  need("clear", HTMLButtonElement).addEventListener("click", () => {
    ui.text.value = "";
    refreshCounts();
  });
  ui.generate.addEventListener("click", guarded(generate));
  ui.play.addEventListener("click", guarded(togglePlay));
  ui.stop.addEventListener("click", guarded(stopAll));
  ui.export.addEventListener("click", guarded(exportWav));
  ui.import.addEventListener("click", guarded(importFile));
  ui.audition.addEventListener("click", guarded(() => startAudition(currentVoice())));
  ui.wave.addEventListener("click", (event) => void seek(event).catch(reportFailure));
  ui.speed.addEventListener("input", () => {
    update({ speed: ui.speed.valueAsNumber });
    renderSliders();
    refreshCounts();
  });
  ui.pause.addEventListener("input", () => {
    update({ pause: ui.pause.valueAsNumber });
    renderSliders();
  });
  ui.live.addEventListener("change", () => update({ live: ui.live.checked }));
  for (const filter of [ui.search, ui.region, ui.gender]) filter.addEventListener("input", renderGrid);
  ui.cloneName.addEventListener("input", refreshControls);
  ui.cloneClip.addEventListener("click", guarded(chooseClip));
  ui.cloneGo.addEventListener("click", guarded(enroll));
  ui.language.addEventListener("change", () => {
    if (isLanguage(ui.language.value)) update({ language: ui.language.value });
    retranslate();
  });
  ui.threads.addEventListener("change", () => {
    const parsed = settings.shape.threads.parse(ui.threads.valueAsNumber);
    update({ threads: parsed });
    ui.threads.value = String(threads());
  });
  ui.setupGo.addEventListener("click", () => {
    ui.setupGo.disabled = true;
    api.installModels().catch(reportFailure);
  });
  window.addEventListener("resize", drawWave);
}

function trackInstall(): Promise<unknown> {
  return api.onInstall((event) => {
    switch (event.kind) {
      case "progress":
        ui.setupProgress.value = event.bytes / model.totalBytes;
        return;
      case "finished":
        ui.setup.hidden = true;
        return;
      case "failed":
        ui.setupText.textContent = t("failed", { message: event.message });
        ui.setupGo.disabled = false;
        return;
      default: {
        const _exhaustive: never = event;
        return _exhaustive;
      }
    }
  });
}

async function start(): Promise<void> {
  model = await api.getModelStatus();
  bind();
  await Promise.all([api.onJob(onJob), trackInstall()]);
  ui.version.textContent = `v${model.version}`;
  ui.language.value = config.language;
  ui.threads.value = String(threads());
  ui.speed.value = String(config.speed);
  ui.pause.value = String(config.pause);
  ui.live.checked = config.live;
  renderSliders();
  showView("create");
  retranslate();
  const status = await api.getPlayerStatus();
  if (!status.available) setStatus(t("noAudio"), true);
  if (!model.installed) {
    ui.setup.hidden = false;
    ui.setupText.textContent = t("setupText", { size: `${Math.round(model.totalBytes / 1e6)} MB` });
  }
}

start().catch(reportFailure);
