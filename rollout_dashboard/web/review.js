const state = {
  running: false,
  remaining: 45 * 60,
  total: 45 * 60,
  timer: null,
  units: 1248,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function renderIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function formatTime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return [hours, minutes, remainder].map((value) => String(value).padStart(2, "0")).join(":");
}

function updateCountdown() {
  $("#countdown").textContent = formatTime(state.remaining);
  const circumference = 2 * Math.PI * 52;
  const progress = state.remaining / state.total;
  $("#ring-progress").style.strokeDashoffset = String(circumference * (1 - progress));
}

function setRunning(running) {
  state.running = running;
  const label = running ? "Running" : "Ready";
  const stateElements = [$("#workflow-state"), $("#current-state")];
  stateElements.forEach((element) => {
    element.textContent = label;
    element.classList.toggle("running", running);
  });
  const button = $("#start-operation");
  button.innerHTML = running
    ? '<i data-lucide="pause" aria-hidden="true"></i>Pause Operation'
    : '<i data-lucide="play" aria-hidden="true"></i>Start Operation';
  renderIcons();
}

function tick() {
  if (!state.running) return;
  if (state.remaining <= 0) {
    window.clearInterval(state.timer);
    state.timer = null;
    setRunning(false);
    showToast($("#auto-advance").checked ? "Step complete. Auto-advance queued." : "Operation step complete.");
    return;
  }
  state.remaining -= 1;
  if (state.remaining % 12 === 0) {
    state.units += 1;
    $("#units-completed").textContent = state.units.toLocaleString();
  }
  updateCountdown();
}

function toggleOperation() {
  setRunning(!state.running);
  if (state.running && !state.timer) state.timer = window.setInterval(tick, 1000);
  showToast(state.running ? "Morning Sort started." : "Operation paused.");
}

function resetOperation() {
  state.remaining = state.total;
  setRunning(false);
  updateCountdown();
  showToast("Operation timer reset.");
}

function filterDashboard(query) {
  const normalized = query.trim().toLowerCase();
  let visible = 0;
  $$(".searchable").forEach((card) => {
    const match = !normalized || card.dataset.search.includes(normalized);
    card.hidden = !match;
    if (match) visible += 1;
  });
  $("#search-empty").hidden = visible !== 0;
}

function selectOperation() {
  const select = $("#operation-select");
  const label = select.options[select.selectedIndex].text;
  $("#workflow-title").textContent = label;
  $("#current-operation").textContent = label;
  resetOperation();
  showToast(`${label} loaded.`);
}

$$(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    $$(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});

$("#start-operation").addEventListener("click", toggleOperation);
$("#reset-operation").addEventListener("click", resetOperation);
$("#operation-select").addEventListener("change", selectOperation);
$("#search-input").addEventListener("input", (event) => filterDashboard(event.target.value));
$(".notification-btn").addEventListener("click", () => showToast("3 exceptions require review."));
$(".avatar").addEventListener("click", () => showToast("Signed in as Jordan Davis."));
$$("[data-add]").forEach((button) => {
  button.addEventListener("click", () => {
    showToast(button.dataset.add === "step" ? "New workflow step ready to configure." : "Employee assignment panel opened.");
  });
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $("#search-input").focus();
  }
});

updateCountdown();
renderIcons();
