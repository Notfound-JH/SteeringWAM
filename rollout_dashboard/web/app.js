const state = {
  runId: null,
  data: null,
  frameIndex: -1,
  camera: "egocentric_mirrored_camera",
  timer: null,
};

const cameraLabels = {
  egocentric_mirrored_camera: "VIEWPORT",
  wrist_cam: "WRIST",
  head_camera: "HEAD",
  over_shoulder_left_camera: "LEFT SHOULDER",
  over_shoulder_right_camera: "RIGHT SHOULDER",
};

const $ = (selector) => document.querySelector(selector);
const setText = (selector, value) => { $(selector).textContent = value ?? "--"; };

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function statusClass(status) {
  if (status.includes("complete") || status === "frame_captured") return "complete";
  if (status.includes("fail") || status.includes("error") || status.includes("timeout")) return "failed";
  return "running";
}

function formatTime(iso) {
  if (!iso) return "--";
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(iso));
}

function renderRuns(runs) {
  const select = $("#run-select");
  const previous = state.runId;
  select.innerHTML = "";
  runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.task} · ${run.id.replace("rollout_", "")}`;
    select.append(option);
  });
  if (previous && runs.some((run) => run.id === previous)) state.runId = previous;
  else state.runId = runs[0]?.id ?? null;
  select.value = state.runId ?? "";
}

function availableCameras() {
  const names = new Set();
  (state.data?.frames ?? []).forEach((frame) => Object.keys(frame.images).forEach((name) => names.add(name)));
  return Object.keys(cameraLabels).filter((name) => names.has(name));
}

function renderCameraTabs() {
  const host = $("#camera-tabs");
  host.innerHTML = "";
  const cameras = availableCameras();
  if (!cameras.includes(state.camera)) state.camera = cameras[0] ?? "egocentric_mirrored_camera";
  cameras.forEach((camera) => {
    const button = document.createElement("button");
    button.className = `camera-tab ${camera === state.camera ? "active" : ""}`;
    button.textContent = cameraLabels[camera];
    button.onclick = () => { state.camera = camera; renderCameraTabs(); renderFrame(); };
    host.append(button);
  });
}

function renderTimeline() {
  const frames = state.data?.frames ?? [];
  const host = $("#timeline");
  host.innerHTML = "";
  if (state.frameIndex < 0 || state.frameIndex >= frames.length) state.frameIndex = frames.length - 1;
  frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.className = `timeline-node ${index === state.frameIndex ? "active" : ""}`;
    button.title = frame.label;
    const label = document.createElement("span");
    label.textContent = frame.label === "Final" ? "FIN" : frame.step;
    button.append(label);
    button.onclick = () => { state.frameIndex = index; renderTimeline(); renderFrame(); };
    host.append(button);
  });
  setText("#timeline-position", frames.length ? `${state.frameIndex + 1} / ${frames.length}` : "0 / 0");
}

function renderFrame() {
  const frame = state.data?.frames?.[state.frameIndex];
  const image = $("#frame");
  const empty = $("#empty-frame");
  const source = frame?.images?.[state.camera];
  if (source) {
    const versionedSource = `${source}?v=${encodeURIComponent(state.data.updated_at)}`;
    if (image.dataset.source !== versionedSource) {
      image.dataset.source = versionedSource;
      image.src = versionedSource;
    }
    image.style.display = "block";
    empty.style.display = "none";
  } else {
    delete image.dataset.source;
    image.removeAttribute("src");
    image.style.display = "none";
    empty.style.display = "grid";
  }
  setText("#frame-label", frame ? `${frame.label.toUpperCase()} · ${cameraLabels[state.camera]}` : "NO FRAME");
  setText("#frame-time", formatTime(state.data?.updated_at));
}

function renderChunks() {
  const host = $("#chunks");
  host.innerHTML = "";
  const chunks = state.data?.chunks ?? [];
  if (!chunks.length) {
    host.innerHTML = '<div class="empty-list">WAITING FOR FIRST POLICY REQUEST</div>';
    return;
  }
  chunks.forEach((chunk) => {
    const row = document.createElement("article");
    row.className = "chunk";
    const image = document.createElement("img");
    image.alt = `Chunk ${chunk.index} policy input`;
    if (chunk.input) image.src = `${chunk.input}?v=${encodeURIComponent(state.data.updated_at)}`;
    const body = document.createElement("div");
    body.className = "chunk-body";
    const inference = chunk.inference_seconds == null ? "INFERENCING" : `${chunk.inference_seconds.toFixed(2)}s`;
    body.innerHTML = `<div class="chunk-head"><span>CHUNK ${String(chunk.index).padStart(2, "0")}</span><span>${inference}</span></div>`;
    const prompt = document.createElement("p");
    prompt.textContent = chunk.prompt ?? "Policy request pending";
    prompt.title = prompt.textContent;
    const meta = document.createElement("div");
    meta.className = "chunk-meta";
    meta.textContent = `${chunk.executed_steps} STEPS · PREFIX ${chunk.prefix_matches === true ? "MATCH" : chunk.prefix_matches === false ? "MISMATCH" : "PENDING"}`;
    body.append(prompt, meta);
    row.append(image, body);
    host.append(row);
  });
}

function renderSummary() {
  const data = state.data;
  setText("#status", data.status.toUpperCase());
  setText("#task", data.task);
  setText("#instruction", data.instruction ?? "等待任务指令与第一批观测");
  setText("#steps", data.steps);
  setText("#chunk-count", `${data.chunks.length}${data.requested_chunks == null ? "" : ` / ${data.requested_chunks}`}`);
  const inference = data.chunks.reduce((sum, chunk) => sum + (chunk.inference_seconds ?? 0), 0);
  setText("#inference", data.chunks.some((chunk) => chunk.inference_seconds != null) ? `${inference.toFixed(2)}s` : "--");
  setText("#native-result", data.native_success === true ? "SUCCESS" : data.native_success === false ? "NOT YET" : "PENDING");
  setText("#gpu", data.renderer_gpu == null ? "GPU --" : `GPU ${data.renderer_gpu} · ${data.physics_device?.toUpperCase() ?? "--"} PHYSX`);
  setText("#updated", formatTime(data.updated_at));
  $("#status-dot").className = `status-dot ${statusClass(data.status)}`;
}

async function refreshLog() {
  if (!state.runId) return;
  const data = await getJSON(`/api/runs/${encodeURIComponent(state.runId)}/log`);
  const log = $("#log");
  log.textContent = data.lines.join("\n") || "Waiting for output...";
  log.scrollTop = log.scrollHeight;
}

async function refresh(options = {}) {
  try {
    const listing = await getJSON("/api/runs");
    renderRuns(listing.runs);
    if (!state.runId) throw new Error("No rollout directories found");
    const wasAtLatest = state.frameIndex < 0 || state.frameIndex === (state.data?.frames?.length ?? 0) - 1;
    state.data = await getJSON(`/api/runs/${encodeURIComponent(state.runId)}`);
    if (wasAtLatest || options.forceLatest) state.frameIndex = state.data.frames.length - 1;
    renderSummary();
    renderCameraTabs();
    renderTimeline();
    renderFrame();
    renderChunks();
    await refreshLog();
  } catch (error) {
    setText("#status", "DISCONNECTED");
    setText("#instruction", error.message);
    $("#status-dot").className = "status-dot failed";
  }
}

function configurePolling() {
  clearInterval(state.timer);
  if ($("#live-toggle").checked) state.timer = setInterval(refresh, 2000);
}

$("#run-select").addEventListener("change", (event) => {
  state.runId = event.target.value;
  state.frameIndex = -1;
  refresh({ forceLatest: true });
});
$("#live-toggle").addEventListener("change", configurePolling);
$("#refresh").addEventListener("click", () => refresh({ forceLatest: true }));

refresh({ forceLatest: true });
configurePolling();