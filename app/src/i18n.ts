const vi = {
  navCreate: "Tạo giọng đọc", navVoices: "Giọng đọc", navSettings: "Cài đặt",
  importFile: "Nhập tệp...", cues: "Biểu cảm", cueLaugh: "Cười", cueSigh: "Thở dài", cueThroat: "Hắng giọng",
  textPlaceholder: "Nhập văn bản cần đọc...", clear: "Xóa", voice: "Giọng đọc", listen: "Nghe thử",
  changeVoice: "Đổi giọng", thisTake: "Lần tạo này", speed: "Tốc độ", pause: "Ngắt giữa câu",
  live: "Phát trực tiếp", generate: "Tạo âm thanh", play: "Phát", pauseAudio: "Tạm dừng", resume: "Tiếp tục",
  stop: "Dừng", exportWav: "Xuất WAV", cloneTitle: "Nhân bản giọng",
  cloneHelp: "Chọn bản ghi 3–30 giây, một người nói, ít tiếng ồn. Giọng được xử lý ngay trên máy này.",
  cloneName: "Tên giọng", chooseClip: "Chọn tệp âm thanh...", denoise: "Khử nhiễu", cloneGo: "Tạo giọng",
  search: "Tìm giọng...", allRegions: "Mọi vùng miền", allGenders: "Mọi giới tính", useVoice: "Dùng",
  remove: "Xóa", language: "Ngôn ngữ", threads: "Số luồng CPU",
  threadsHelp: "Nhiều luồng đọc nhanh hơn nhưng máy sẽ bận hơn.", setupTitle: "Tải mô hình giọng đọc",
  setupGo: "Tải xuống", setupText: "Cần tải {size} một lần. Sau đó ứng dụng chạy hoàn toàn ngoại tuyến.",
  ready: "Sẵn sàng", loading: "Đang nạp mô hình...", planned: "Đang tạo đoạn {index}/{total}...",
  cancelled: "Đã dừng", finished: "Hoàn tất: {time}", exported: "Đã xuất {path}", cloned: "Đã tạo giọng {name}",
  noAudio: "Không tìm thấy thiết bị âm thanh. Vẫn có thể xuất WAV.", cloning: "Đang tạo giọng...",
  auditioning: "Đang tạo mẫu nghe thử...", confirmRemove: "Xóa giọng {name}?", failed: "Lỗi: {message}",
  male: "Nam", female: "Nữ",
  counts: "{words} từ · {chars} ký tự · khoảng {time}", audioFiles: "Âm thanh", documents: "Tài liệu",
};

type Key = keyof typeof vi;

const en: Record<Key, string> = {
  navCreate: "Create speech", navVoices: "Voices", navSettings: "Settings",
  importFile: "Import file...", cues: "Emotion", cueLaugh: "Laugh", cueSigh: "Sigh", cueThroat: "Clear throat",
  textPlaceholder: "Type the text to read...", clear: "Clear", voice: "Voice", listen: "Preview",
  changeVoice: "Change voice", thisTake: "This take", speed: "Speed", pause: "Pause between sentences",
  live: "Play while generating", generate: "Generate audio", play: "Play", pauseAudio: "Pause", resume: "Resume",
  stop: "Stop", exportWav: "Export WAV", cloneTitle: "Clone a voice",
  cloneHelp: "Pick a 3–30 second recording of one speaker with little noise. It is processed on this machine.",
  cloneName: "Voice name", chooseClip: "Choose audio file...", denoise: "Remove noise", cloneGo: "Create voice",
  search: "Search voices...", allRegions: "All regions", allGenders: "All genders", useVoice: "Use",
  remove: "Delete", language: "Language", threads: "CPU threads",
  threadsHelp: "More threads read faster but keep the machine busier.", setupTitle: "Download the voice models",
  setupGo: "Download", setupText: "A one-time {size} download is needed. After that the app runs fully offline.",
  ready: "Ready", loading: "Loading models...", planned: "Generating part {index}/{total}...",
  cancelled: "Stopped", finished: "Done: {time}", exported: "Exported {path}", cloned: "Created voice {name}",
  noAudio: "No audio device found. WAV export still works.", cloning: "Creating voice...",
  auditioning: "Generating preview...", confirmRemove: "Delete voice {name}?", failed: "Error: {message}",
  male: "Male", female: "Female",
  counts: "{words} words · {chars} characters · about {time}", audioFiles: "Audio", documents: "Documents",
};

export type Language = "vi" | "en";
let table: Record<Key, string> = vi;

export function t(key: Key, values: Record<string, string | number> = {}): string {
  return table[key].replace(/\{(\w+)\}/gu, (whole, name: string) => String(values[name] ?? whole));
}

const hasKey = (key: string): key is Key => key in vi;

/** Apply a language to every tagged element. */
export function applyLanguage(language: Language): void {
  table = language === "en" ? en : vi;
  document.documentElement.lang = language;
  for (const element of document.querySelectorAll<HTMLElement>("[data-i18n]")) {
    const key = element.dataset["i18n"] ?? "";
    if (hasKey(key)) element.textContent = t(key);
  }
  for (const element of document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-i18n-placeholder]")) {
    const key = element.dataset["i18nPlaceholder"] ?? "";
    if (hasKey(key)) element.placeholder = t(key);
  }
}
