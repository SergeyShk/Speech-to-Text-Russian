"use strict";

const STORAGE = "stt-ru-jobs";
const COOKIE = "stt_ru_token";
const POLL_MS = 1500;
const ACTIVE = new Set(["queued", "running"]);
const STATUS = { queued: "в очереди", running: "распознается", done: "готово", failed: "ошибка" };

const $ = (id) => document.getElementById(id);
const state = {
  info: null,
  jobs: new Map(),
  current: null,
  loaded: null,
  segments: [],
  waveform: null,
  rows: [],
  playing: [],
  frame: 0,
  polling: false,
};

function storedIds() {
  try {
    const ids = JSON.parse(localStorage.getItem(STORAGE) || "[]");
    return Array.isArray(ids) ? ids.filter((id) => typeof id === "string") : [];
  } catch {
    return [];
  }
}

function storeIds() {
  try {
    localStorage.setItem(STORAGE, JSON.stringify([...state.jobs.keys()]));
  } catch {
    // Без хранилища список задач живет до перезагрузки страницы
  }
}

async function detail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Ответ не JSON
  }
  return `Ошибка ${response.status}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (response.status === 401) {
    showToken();
    throw new Error("Нужен токен доступа");
  }
  return response;
}

function clock(seconds) {
  const tenths = Math.round(Math.max(0, seconds) * 10);
  const hours = Math.floor(tenths / 36000);
  const minutes = Math.floor((tenths % 36000) / 600);
  const rest = ((tenths % 600) / 10).toFixed(1).padStart(4, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}

function length(seconds) {
  if (seconds < 10) return `${Math.max(0, seconds).toFixed(1)} с`;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  if (hours) return minutes ? `${hours} ч ${minutes} мин` : `${hours} ч`;
  if (minutes) return rest ? `${minutes} мин ${rest} с` : `${minutes} мин`;
  return `${rest} с`;
}

function megabytes(bytes) {
  return `${(bytes / 1048576).toFixed(bytes < 10485760 ? 1 : 0)} МБ`;
}

function since(value) {
  return value ? (Date.now() - Date.parse(value)) / 1000 : 0;
}

function speakerColor(channel) {
  const css = getComputedStyle(document.documentElement);
  return css.getPropertyValue(`--speaker-${(channel ?? 0) % 4}`).trim();
}

function showToken() {
  $("token-form").hidden = false;
  $("upload").hidden = true;
}

function setCookie(value, age) {
  const secure = location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${COOKIE}=${encodeURIComponent(value)}; path=/; max-age=${age}; SameSite=Strict${secure}`;
}

async function login(event) {
  event.preventDefault();
  setCookie($("token").value, 2592000);
  let info;
  try {
    info = await (await fetch("/v1/info", { credentials: "same-origin" })).json();
  } catch {
    $("token-error").textContent = "Нет связи с сервисом";
    $("token-error").hidden = false;
    return;
  }
  if (!info.authorized) {
    $("token-error").textContent = "Неверный токен";
    setCookie("", 0);
    $("token-error").hidden = false;
    return;
  }
  $("token-error").hidden = true;
  $("token").value = "";
  state.info = info;
  start();
}

function uploadOne(file) {
  return new Promise((resolve) => {
    const error = $("upload-error");
    error.hidden = true;
    if (state.info && file.size > state.info.max_size) {
      error.textContent = `${file.name}: файл больше ${megabytes(state.info.max_size)}`;
      error.hidden = false;
      resolve();
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    const bar = $("progress-bar");
    $("progress").hidden = false;
    bar.style.width = "0";
    xhr.open("POST", "/v1/transcriptions");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) bar.style.width = `${(event.loaded / event.total) * 100}%`;
    };
    xhr.onload = () => {
      $("progress").hidden = true;
      if (xhr.status === 202 && xhr.response) {
        const job = xhr.response;
        state.jobs = new Map([[job.id, job], ...state.jobs]);
        storeIds();
        renderList();
        openJob(job.id);
      } else if (xhr.status === 401) {
        showToken();
      } else {
        const message = xhr.response && typeof xhr.response.detail === "string" ? xhr.response.detail : `Ошибка ${xhr.status}`;
        error.textContent = `${file.name}: ${message}`;
        error.hidden = false;
      }
      resolve();
    };
    xhr.onerror = () => {
      $("progress").hidden = true;
      error.textContent = `${file.name}: нет связи с сервисом`;
      error.hidden = false;
      resolve();
    };
    xhr.send(form);
  });
}

async function upload(files) {
  for (const file of files) await uploadOne(file);
}

function renderList() {
  const list = $("jobs");
  list.replaceChildren();
  $("empty").hidden = state.jobs.size > 0;
  for (const [id, job] of state.jobs) {
    if (!job) continue;
    const item = document.createElement("li");
    item.className = id === state.current ? "active" : "";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = job.filename;
    name.title = job.filename;
    const badge = document.createElement("span");
    badge.className = `badge ${job.status}`;
    badge.textContent = STATUS[job.status];
    item.append(name, badge);
    item.addEventListener("click", () => openJob(id));
    list.append(item);
  }
}

async function refresh(id) {
  try {
    const response = await api(`/v1/jobs/${encodeURIComponent(id)}`);
    if (response.status === 404) {
      state.jobs.delete(id);
      storeIds();
      if (state.current === id) closeJob();
      return;
    }
    if (response.ok) state.jobs.set(id, await response.json());
  } catch {
    // Сервис недоступен: следующий опрос повторит запрос
  }
}

async function poll() {
  const ids = [...state.jobs].filter(([, job]) => !job || ACTIVE.has(job.status)).map(([id]) => id);
  await Promise.all(ids.map(refresh));
  renderList();
  if (state.current) await renderJob();
  setTimeout(poll, POLL_MS);
}

function closeJob() {
  state.current = null;
  state.loaded = null;
  const audio = $("audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  $("result").hidden = true;
  $("placeholder").hidden = false;
  if (location.hash) history.replaceState(null, "", location.pathname);
  renderList();
}

function openJob(id) {
  if (state.current !== id) {
    state.current = id;
    state.loaded = null;
    state.segments = [];
    state.waveform = null;
    $("audio").pause();
    $("player").hidden = true;
    $("downloads").hidden = true;
    $("segments-wrap").hidden = true;
    $("audio-error").hidden = true;
    if (location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);
  }
  renderList();
  renderJob();
}

function statusText(job) {
  if (job.status === "queued") {
    return job.position ? `В очереди: перед ней задач - ${job.position}` : "В очереди: следующая";
  }
  if (job.status === "running") return `Распознается, прошло ${length(since(job.started))}`;
  if (job.status === "failed") return `Ошибка: ${job.error}`;
  const seconds = (Date.parse(job.finished) - Date.parse(job.started)) / 1000;
  const kept = job.expires ? `, хранится до ${new Date(job.expires).toLocaleString("ru-RU")}` : "";
  const found = job.segments ? `фрагментов ${job.segments}` : "речи в записи не найдено";
  return `Готово: ${found}, распознано за ${length(seconds)}${kept}`;
}

async function renderJob() {
  const id = state.current;
  const job = state.jobs.get(id);
  if (!job) return;
  $("placeholder").hidden = true;
  $("result").hidden = false;
  $("title").textContent = job.filename;
  const details = [megabytes(job.size)];
  if (job.duration !== null) details.push(length(job.duration));
  if (job.channels !== null) details.push(`каналов ${job.channels}`);
  details.push(`загружена ${new Date(job.created).toLocaleString("ru-RU")}`);
  $("details").textContent = details.join(" · ");
  const status = $("status");
  status.textContent = statusText(job);
  status.className = job.status === "failed" ? "status error" : "status";
  if (state.loaded === `${id}:${job.status}`) return;
  if (job.duration === null && job.status !== "done") return;
  state.loaded = `${id}:${job.status}`;
  await loadWaveform(id);
  if (job.status === "done") await loadResult(id);
}

async function loadWaveform(id) {
  if (state.waveform) return;
  const response = await api(`/v1/jobs/${encodeURIComponent(id)}/waveform`);
  if (!response.ok || state.current !== id) return;
  state.waveform = await response.json();
  const audio = $("audio");
  audio.src = `/v1/jobs/${encodeURIComponent(id)}/audio`;
  $("player").hidden = false;
  drawWave();
}

async function loadResult(id) {
  const response = await api(`/v1/jobs/${encodeURIComponent(id)}/result?format=json`);
  if (!response.ok || state.current !== id) return;
  const result = await response.json();
  state.segments = result.segments;
  renderDownloads(id);
  renderSegments();
  drawWave();
}

function renderDownloads(id) {
  const box = $("downloads");
  box.replaceChildren();
  for (const format of state.info.formats) {
    const link = document.createElement("a");
    link.className = "button";
    link.href = `/v1/jobs/${encodeURIComponent(id)}/result?format=${format}`;
    link.download = "";
    link.textContent = format.toUpperCase();
    box.append(link);
  }
  box.hidden = false;
}

function renderSegments() {
  const body = $("segments");
  body.replaceChildren();
  state.rows = state.segments.map((segment) => {
    const row = document.createElement("tr");
    const cells = [clock(segment.start), clock(segment.end), segment.speaker ?? "", segment.text];
    cells.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index < 2) cell.className = "time";
      if (index === 2) {
        cell.className = "speaker";
        cell.style.color = speakerColor(segment.channel);
      }
      row.append(cell);
    });
    row.addEventListener("click", () => seek(segment.start, true));
    body.append(row);
    return row;
  });
  $("segments-wrap").hidden = state.segments.length === 0;
}

function seek(time, play) {
  const audio = $("audio");
  audio.currentTime = time;
  if (play) audio.play().catch(() => {});
  highlight();
  drawWave();
}

function highlight() {
  const time = $("audio").currentTime;
  for (const row of state.playing) row.classList.remove("playing");
  state.playing = state.rows.filter((_, index) => {
    const segment = state.segments[index];
    return segment.start <= time && time < segment.end;
  });
  for (const row of state.playing) row.classList.add("playing");
}

function drawWave() {
  const wave = state.waveform;
  const canvas = $("wave");
  if (!wave || canvas.clientWidth === 0) return;
  const rows = Math.max(1, wave.peaks.length);
  const rowHeight = rows > 2 ? 44 : 64;
  const width = canvas.clientWidth;
  const height = rows * rowHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const x = (time) => (wave.duration ? (time / wave.duration) * width : 0);

  context.globalAlpha = 0.18;
  for (const segment of state.segments) {
    const targets = segment.channel === null ? [...Array(rows).keys()] : [segment.channel];
    context.fillStyle = speakerColor(segment.channel);
    for (const row of targets) {
      if (row < rows) context.fillRect(x(segment.start), row * rowHeight, Math.max(1, x(segment.end) - x(segment.start)), rowHeight);
    }
  }
  context.globalAlpha = 1;

  const loudest = Math.max(0.05, ...wave.peaks.flat());
  const css = getComputedStyle(document.documentElement);
  context.fillStyle = css.getPropertyValue("--wave").trim();
  wave.peaks.forEach((peaks, row) => {
    const middle = row * rowHeight + rowHeight / 2;
    const amplitude = rowHeight / 2 - 3;
    for (let pixel = 0; pixel < width; pixel += 1) {
      const first = Math.floor((pixel / width) * peaks.length);
      const last = Math.max(first + 1, Math.floor(((pixel + 1) / width) * peaks.length));
      let peak = 0;
      for (let index = first; index < last; index += 1) peak = Math.max(peak, peaks[index]);
      const half = Math.max(0.5, (peak / loudest) * amplitude);
      context.fillRect(pixel, middle - half, 1, half * 2);
    }
  });

  context.fillStyle = css.getPropertyValue("--line").trim();
  for (let row = 1; row < rows; row += 1) context.fillRect(0, row * rowHeight, width, 1);
  context.fillStyle = css.getPropertyValue("--accent").trim();
  context.fillRect(Math.min(width - 2, x($("audio").currentTime)), 0, 2, height);
}

function animate() {
  highlight();
  drawWave();
  if (!$("audio").paused) state.frame = requestAnimationFrame(animate);
}

function start() {
  $("token-form").hidden = true;
  $("upload").hidden = false;
  const info = state.info;
  const limits = [`до ${megabytes(info.max_size)}`];
  if (info.max_duration) limits.push(`до ${length(info.max_duration)}`);
  $("limits").textContent = limits.join(", ");
  state.jobs = new Map(storedIds().map((id) => [id, null]));
  renderList();
  const id = location.hash.slice(1);
  if (id) {
    if (!state.jobs.has(id)) {
      state.jobs = new Map([[id, null], ...state.jobs]);
      storeIds();
    }
    state.current = id;
  }
  if (state.current) openJob(state.current);
  if (!state.polling) {
    state.polling = true;
    poll();
  }
}

async function init() {
  const input = $("file");
  input.multiple = true;
  input.addEventListener("change", () => {
    upload([...input.files]);
    input.value = "";
  });
  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    upload([...event.dataTransfer.files]);
  });
  $("token-form").addEventListener("submit", login);
  $("delete").addEventListener("click", async () => {
    const id = state.current;
    if (!id || !confirm("Удалить запись и результат распознавания?")) return;
    const response = await api(`/v1/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (response.ok || response.status === 404) {
      state.jobs.delete(id);
      storeIds();
      closeJob();
    }
  });
  const audio = $("audio");
  audio.addEventListener("play", () => {
    cancelAnimationFrame(state.frame);
    animate();
  });
  audio.addEventListener("pause", animate);
  audio.addEventListener("seeked", animate);
  audio.addEventListener("error", () => {
    if (audio.getAttribute("src")) $("audio-error").hidden = false;
  });
  $("wave").addEventListener("click", (event) => {
    const wave = state.waveform;
    if (!wave) return;
    const box = event.currentTarget.getBoundingClientRect();
    seek(((event.clientX - box.left) / box.width) * wave.duration, false);
  });
  window.addEventListener("resize", drawWave);
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (id && id !== state.current) {
      if (!state.jobs.has(id)) state.jobs = new Map([[id, null], ...state.jobs]);
      state.current = id;
      refresh(id).then(() => (state.jobs.has(id) ? openJob(id) : closeJob()));
    }
  });

  try {
    state.info = await (await fetch("/v1/info", { credentials: "same-origin" })).json();
  } catch {
    $("service").textContent = "Сервис недоступен";
    return;
  }
  $("service").textContent = `Модель ${state.info.model} · stt-ru ${state.info.version}`;
  if (state.info.auth && !state.info.authorized) showToken();
  else start();
}

init();
