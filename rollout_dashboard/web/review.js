const state = { runId: null, data: null };

const $ = (selector) => document.querySelector(selector);
const setText = (selector, value) => { $(selector).textContent = value; };

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function renderRuns(runs) {
  const select = $("#run-select");
  const previous = state.runId;
  select.innerHTML = "";
  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.task} · ${run.id.replace("rollout_", "")}`;
    select.append(option);
  }
  state.runId = previous && runs.some((run) => run.id === previous) ? previous : runs[0]?.id ?? null;
  select.value = state.runId ?? "";
}

function renderSummary(data) {
  const complete = data.status.includes("complete");
  setText("#status", data.status.toUpperCase());
  setText("#task", data.task);
  setText("#chunk-count", `${data.chunks.length} CHUNKS`);
  setText("#steps", `${data.steps} STEPS`);
  setText("#native-result", data.native_success ? "TASK SUCCESS" : "TASK INCOMPLETE");
  $("#status-dot").className = `status-dot ${complete ? "complete" : "running"}`;
}

function renderChunks(chunks) {
  const host = $("#chunks");
  host.innerHTML = "";
  if (!chunks.length) {
    host.innerHTML = '<div class="empty-state">NO COMPLETED ACTION CHUNKS</div>';
    return;
  }

  for (const chunk of chunks) {
    const card = document.createElement("article");
    card.className = "review-card";

    const videoShell = document.createElement("div");
    videoShell.className = "video-shell";
    if (chunk.viewport_video) {
      const video = document.createElement("video");
      video.controls = true;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.src = chunk.viewport_video;
      if (chunk.viewport_poster) video.poster = chunk.viewport_poster;
      videoShell.append(video);
    } else {
      const pending = document.createElement("div");
      pending.className = "pending-video";
      pending.textContent = "VIEWPORT VIDEO PENDING";
      videoShell.append(pending);
    }
    const videoLabel = document.createElement("span");
    videoLabel.className = "video-label";
    videoLabel.textContent = "EXECUTED ACTION · VIEWPORT";
    videoShell.append(videoLabel);

    const body = document.createElement("div");
    body.className = "card-body";
    const head = document.createElement("div");
    head.className = "card-head";
    const title = document.createElement("h2");
    title.textContent = `CHUNK ${String(chunk.index).padStart(2, "0")}`;
    const meta = document.createElement("span");
    meta.className = "card-meta";
    const inference = chunk.inference_seconds == null ? "--" : `${chunk.inference_seconds.toFixed(2)}s`;
    meta.textContent = `${chunk.executed_steps} STEPS · ${inference}`;
    head.append(title, meta);
    const label = document.createElement("div");
    label.className = "instruction-label";
    label.textContent = "INSTRUCTION";
    const instruction = document.createElement("p");
    instruction.className = "instruction";
    instruction.textContent = chunk.prompt ?? "Instruction unavailable";
    body.append(head, label, instruction);
    card.append(videoShell, body);
    host.append(card);
  }
}

async function refresh() {
  try {
    const listing = await getJSON("/api/runs");
    renderRuns(listing.runs);
    if (!state.runId) throw new Error("No rollout directories found");
    state.data = await getJSON(`/api/runs/${encodeURIComponent(state.runId)}`);
    renderSummary(state.data);
    renderChunks(state.data.chunks);
  } catch (error) {
    setText("#status", "DISCONNECTED");
    setText("#task", error.message);
    $("#status-dot").className = "status-dot";
  }
}

$("#run-select").addEventListener("change", (event) => {
  state.runId = event.target.value;
  refresh();
});
$("#refresh").addEventListener("click", refresh);

refresh();