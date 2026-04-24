import axios from "axios";

const base =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) || "";

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
export async function analyzeAudio(file, question) {
  const form = new FormData();
  form.append("file", file);
  form.append("question", question || "What can be inferred from the audio?");
  const { data } = await client.post("/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function health() {
  const { data } = await client.get("/health");
  return data;
}
