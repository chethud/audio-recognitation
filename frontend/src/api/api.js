import axios from "axios";

const base =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) || "";

/** Must match backend `data.max_audio_length_sec` in config.yaml */
const MAX_ANALYZE_SEC = 30;
/** Skip browser decode for very large files (server trims instead). */
const MAX_CLIENT_TRIM_BYTES = 12 * 1024 * 1024;

export const AUTH_STORAGE_KEY = "alm_auth_token";

const client = axios.create({
  baseURL: base || undefined,
  timeout: 3_600_000,
  headers: { Accept: "application/json" },
});

export function setAuthToken(token) {
  if (token) {
    client.defaults.headers.common.Authorization = `Bearer ${token}`;
    localStorage.setItem(AUTH_STORAGE_KEY, token);
  }
}

export function clearAuthToken() {
  delete client.defaults.headers.common.Authorization;
  localStorage.removeItem(AUTH_STORAGE_KEY);
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Upload only the first few seconds so long files upload and decode faster.
 */
async function prepareUploadFile(file) {
  if (!file || file.size > MAX_CLIENT_TRIM_BYTES) {
    return file;
  }

  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    await audioCtx.close();

    const frames = Math.min(decoded.length, Math.ceil(MAX_ANALYZE_SEC * decoded.sampleRate));
    const offline = new OfflineAudioContext(1, frames, decoded.sampleRate);
    const trimmed = offline.createBuffer(1, frames, decoded.sampleRate);
    trimmed.copyToChannel(decoded.getChannelData(0).subarray(0, frames), 0);

    const source = offline.createBufferSource();
    source.buffer = trimmed;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();

    const wav = encodeWav(rendered.getChannelData(0), rendered.sampleRate);
    const baseName = file.name.replace(/\.[^.]+$/, "") || "audio";
    return new File([wav], `${baseName}-clip.wav`, { type: "audio/wav" });
  } catch {
    return file;
  }
}

export async function signupRequest(email, password) {
  const { data } = await client.post("/auth/signup", { email, password });
  return data;
}

export async function loginRequest(email, password) {
  const { data } = await client.post("/auth/login", { email, password });
  return data;
}

export async function fetchMe() {
  const { data } = await client.get("/auth/me");
  return data;
}

export async function logoutRequest() {
  try {
    await client.post("/auth/logout");
  } finally {
    clearAuthToken();
  }
}

/**
 * @param {File} file
 * @param {string} question
 */
export async function analyzeAudio(file, question, { onStatus } = {}) {
  await waitForModelsReady({ onStatus });

  const uploadFile = await prepareUploadFile(file);
  const form = new FormData();
  form.append("file", uploadFile);
  form.append("question", question || "What can be inferred from the audio?");

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { data } = await client.post("/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data;
    } catch (err) {
      if (isModelsLoadingError(err) && attempt < 2) {
        onStatus?.("loading");
        await waitForModelsReady({ onStatus });
        continue;
      }
      throw err;
    }
  }
}

export async function health() {
  const { data } = await client.get("/health", { timeout: 120_000 });
  return data;
}

/** Poll /health until model_ready or timeout (Render cold start + Whisper download). */
export async function waitForModelsReady({
  timeoutMs = 900_000,
  intervalMs = 4_000,
  onStatus,
} = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const data = await health();
      if (data?.model_ready) {
        onStatus?.("ready");
        return data;
      }
      onStatus?.("loading");
    } catch {
      onStatus?.("waking");
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(
    "AI models are still starting on the server. Wait 2–3 minutes and try again."
  );
}

function isModelsLoadingError(err) {
  const status = err?.response?.status;
  const detail = err?.response?.data?.detail;
  const msg = typeof detail === "string" ? detail : "";
  return status === 503 && msg.toLowerCase().includes("still loading");
}
