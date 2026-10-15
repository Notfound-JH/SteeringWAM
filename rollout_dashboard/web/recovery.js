"use strict";

const $ = (id) => document.getElementById(id);
const sourceNames = { initial: "初始状态", wam: "WAM 执行", agent_correction: "Agent 修正", agent_condition: "Condition 修正", recovery_evaluation: "统一 WAM 评估窗口", agent_control: "Agent 直接控制", wam_continuation: "WAM 续跑" };
const sourceClasses = { initial: "wam", wam: "wam", agent_correction: "correction", agent_condition: "correction", recovery_evaluation: "correction", agent_control: "correction", wam_continuation: "continuation" };
const eventNames = { failure: "发现失败", proposal: "Agent 介入", assessment: "局部评估", handoff: "交还 WAM", success: "任务成功", condition_change: "条件变更" };
const conditionKinds = { initial: "首次条件", unchanged: "条件不变", recovery_rewrite: "纠错改写", subgoal_progression: "子目标推进" };
const cameraNames = { wrist_cam: "手腕相机", over_shoulder_left_camera: "左肩视角", over_shoulder_right_camera: "右肩视角", head_camera: "头部视角" };
let episode = null;
let activeStep = -1;
let selectedSegment = null;
let loading = false;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}
function fixed(value, digits = 4) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
  $("message").hidden = !text;
}
async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(`${response.status}: ${body?.detail || response.statusText}`);
  }
  return response.json();
}
function setStep(step, seek = false) {
  if (!episode) return;
  step = Math.max(0, Math.min(episode.steps, Math.round(step)));
  if (seek) $("episode-video").currentTime = step * episode.dt;
  if (step === activeStep) return;
  activeStep = step;
  const observation = episode.observations[step];
  $("step-slider").value = step;
  $("current-step").textContent = `${step} / ${episode.steps}`;
  $("current-source").textContent = sourceNames[observation.source] || observation.source;
  updateCamera();
  $("ee-position").replaceChildren(...["X", "Y", "Z"].map((axis, index) => {
    const cell = node("div", undefined, "coordinate");
    cell.append(node("span", axis), document.createTextNode(fixed(observation.robot.ee_pos?.[index])));
    return cell;
  }));
  $("gripper").textContent = fixed(observation.robot.gripper_pos?.[0], 3);
  $("robot-state").textContent = JSON.stringify({
    measured_arm_joint_pos: observation.robot.measured_arm_joint_pos || observation.robot.arm_joint_pos,
    measured_arm_joint_vel: observation.robot.measured_arm_joint_vel,
    actual_environment_command: observation.action,
  }, null, 2);
}
function updateCamera() {
  if (!episode || activeStep < 0) return;
  const url = episode.observations[activeStep].frames[$("camera-select").value];
  if (!url) {
    $("observation-image").hidden = true;
    message("此控制步缺少所选相机的实际观测。", true);
    return;
  }
  $("observation-image").hidden = false;
  $("observation-image").src = url;
}
function drawChart(tracking) {
  const svg = $("error-chart");
  svg.replaceChildren();
  const element = (tag, attributes, text) => {
    const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => item.setAttribute(key, value));
    if (text !== undefined) item.textContent = text;
    svg.append(item);
  };
  if (!tracking.length) {
    element("text", { x: 20, y: 65, fill: "#6c7d85", "font-size": 12 }, "无 Cartesian 修正响应，无法计算位置跟踪误差。");
    return;
  }
  const max = Math.max(1, ...tracking.map((point) => point.error_mm));
  for (let index = 0; index < 3; index++) {
    const y = 15 + index * 45;
    element("line", { x1: 40, y1: y, x2: 580, y2: y, stroke: "#e0e8e6", "stroke-dasharray": "3 4" });
    element("text", { x: 4, y: y + 4, fill: "#6c7d85", "font-size": 10 }, fixed(max * (1 - index / 2), 1));
  }
  const points = tracking.map((point, index) => `${40 + index / Math.max(1, tracking.length - 1) * 540},${105 - point.error_mm / max * 90}`).join(" ");
  element("polyline", { points, fill: "none", stroke: "#197768", "stroke-width": 2.5, "stroke-linejoin": "round" });
  element("text", { x: 40, y: 127, fill: "#6c7d85", "font-size": 10 }, `Step ${tracking[0].step}`);
  element("text", { x: 580, y: 127, fill: "#6c7d85", "font-size": 10, "text-anchor": "end" }, `Step ${tracking.at(-1).step}`);
}
function renderAttempt() {
  const attempt = episode.attempts[Number($("attempt-select").value)];
  $("attempt-empty").hidden = Boolean(attempt);
  $("attempt-content").hidden = !attempt;
  if (!attempt) return;
  const proposal = attempt.proposal;
  $("correction-mode").textContent = proposal.correction_mode.toUpperCase();
  $("diagnosis").textContent = proposal.diagnosis;
  $("recovery-goal").textContent = proposal.recovery_goal;
  $("mode-reason").textContent = proposal.mode_reason || (
    episode.experiment_mode === "agent_control" ? "Agent-only 固定动作接口，不参与三模式选择。" : "旧记录未保存独立的模式选择理由。"
  );
  const evaluation = attempt.evaluation;
  $("evaluation-window").textContent = evaluation
    ? `WAM 评估：${evaluation.executed_steps}/${evaluation.requested_steps} 步 · ${evaluation.complete ? "窗口完整" : `窗口未完成 (${evaluation.interrupted_by})`} · ${proposal.correction_mode === "action" ? "沿用纠错前条件" : "使用纠错后条件"}`
    : episode.experiment_mode === "agent_control" ? "非 Recovery 评估窗口。"
      : episode.protocol_version === "wam-recovery-v2" ? "没有完成评估窗口记录，不可视作已通过评估。" : "旧协议：没有统一评估窗口，不与新协议直接混算。";
  $("control-code").textContent = [
    proposal.control_code || (proposal.program ? JSON.stringify(proposal.program, null, 2) : ""),
    proposal.condition ? `Condition: ${proposal.condition}\nWAM execute steps: ${proposal.wam_execute_steps}` : "",
  ].filter(Boolean).join("\n\n");
  const success = attempt.assessment?.local_recovery_success;
  $("local-outcome").textContent = episode.experiment_mode === "agent_control" ? "控制程序 · 非 Recovery" : success === true ? "局部恢复成功" : success === false ? "局部恢复失败" : "未评估";
  $("local-outcome").className = `badge ${success === true ? "" : "amber"}`;
  const camera = "over_shoulder_left_camera";
  const before = episode.observations[attempt.start].frames;
  const after = episode.observations[attempt.end].frames;
  $("before-image").src = before[camera] || before.wrist_cam;
  $("after-image").src = after[camera] || after.wrist_cam;
  $("before-caption").textContent = `BEFORE · Step ${attempt.start}`;
  $("after-caption").textContent = `AFTER · Step ${attempt.end} · ${attempt.executed_steps} 实际步`;
  const last = attempt.tracking.at(-1);
  $("pose-table").replaceChildren(...[["目标", last?.target], ["实测", last?.measured]].map(([label, values]) => {
    const row = node("tr");
    row.append(node("td", label), ...[0, 1, 2].map((index) => node("td", fixed(values?.[index]))));
    return row;
  }));
  drawChart(attempt.tracking);
  $("assessment").textContent = attempt.assessment?.evidence || "尚无局部恢复评估；不要将动作执行完成视作恢复成功。";
}
function renderSegment() {
  selectedSegment = episode.segments[Number($("segment-select").value)];
  $("prediction-empty").hidden = Boolean(selectedSegment);
  $("prediction-content").hidden = !selectedSegment;
  if (!selectedSegment) return;
  $("prediction-input").src = selectedSegment.input;
  $("prediction-video").src = selectedSegment.prediction;
  const change = selectedSegment.condition_change;
  $("prompt").textContent = [
    selectedSegment.prompt,
    change ? `条件用途：${conditionKinds[change.kind] || change.kind}${change.reason ? `\n变更理由：${change.reason}` : ""}` : "旧记录未标注条件变更用途。",
  ].join("\n\n");
  $("segment-stats").textContent = `${sourceNames[selectedSegment.source] || selectedSegment.source} · 起点 ${selectedSegment.start} · 预测 ${selectedSegment.predicted_steps} 步 / 实际 ${selectedSegment.executed_steps} 步 · 推理 ${fixed(selectedSegment.inference_seconds, 1)} s`;
}
function render() {
  const agentOnly = episode.experiment_mode === "agent_control";
  const strictPlacement = episode.success_criterion === "native_and_stable_blue_placement";
  $("task-title").textContent = episode.task;
  $("task-goal").textContent = episode.goal;
  $("episode-id").textContent = episode.id;
  $("outcome").textContent = episode.verified ? "闭环验证通过" : episode.mock_execution !== false ? "非真实执行验证" : episode.native_success ? "尚未通过闭环验证" : "任务未成功";
  if (agentOnly) $("outcome").textContent = episode.agent_control_verified ? "Agent 控制任务验证通过" : "Agent 控制未验证成功";
  if (strictPlacement) $("outcome").textContent = episode.stable_blue_placement_verified ? "单锤稳定放置验证通过" : "稳定放置尚未验证通过";
  $("outcome-detail").textContent = `${strictPlacement ? "原生条件 + 加严放置检查" : "原生任务"}：${episode.native_success === true ? "成功" : episode.native_success === false ? "失败" : "未知"} · ${episode.verification ? "已保存独立校验结果" : "缺少独立校验结果"}`;
  if (agentOnly && episode.task === "ToolsPickingHammerTask" && !strictPlacement) $("outcome-detail").textContent += " · 未验证稳定放置";
  const agentSteps = episode.observations.filter((item) => ["agent_correction", "agent_condition", "agent_control"].includes(item.source)).length;
  const continuation = episode.observations.filter((item) => item.source === "wam_continuation").length;
  $("step-count").textContent = episode.steps;
  $("phase-counts").textContent = `Agent ${agentSteps} 步 · WAM 续跑 ${continuation} 步`;
  $("attempt-count").textContent = agentOnly ? `${episode.attempts.length} 个控制程序` : `${episode.attempts.length} / ${episode.attempt_limit}`;
  const tracking = episode.attempts.flatMap((attempt) => attempt.tracking);
  $("tracking-error").textContent = tracking.length ? `${fixed(tracking.at(-1).error_mm, 2)} mm` : "—";
  $("execution-kind").textContent = episode.mock_execution === false ? "真实仿真 · 记录回放" : "MOCK / 未确认执行类型";
  if (agentOnly) $("execution-kind").textContent += " · Agent-only 基线，非 WAM Recovery";
  $("execution-kind").className = `badge ${episode.mock_execution === false ? "" : "amber"}`;
  const assisted = episode.agent_input_mode === "debug_ground_truth_assisted";
  const inputMode = assisted ? "Agent 使用仿真物体真值辅助，不是纯视觉评测" : `Agent 输入来源：${episode.agent_input_mode || "未记录，不能判定为纯视觉"}`;
  $("agent-provenance").textContent = `${inputMode} · Agent 模型版本：${episode.agent_model_version || "未记录精确版本"}`;
  $("agent-provenance").className = assisted ? "failure-note" : "muted";
  $("video-duration").textContent = `${fixed(1 / episode.dt, 0)} FPS · ${fixed(episode.duration, 2)} s`;
  $("episode-video").src = episode.video;
  $("step-slider").max = episode.steps;
  $("camera-select").replaceChildren(...Object.keys(episode.observations[0].frames).map((camera) => {
    const option = node("option", cameraNames[camera] || camera); option.value = camera; return option;
  }));
  if (episode.observations[0].frames.over_shoulder_left_camera) $("camera-select").value = "over_shoulder_left_camera";
  $("phase-track").replaceChildren(...episode.phases.map((phase) => {
    const button = node("button", "", sourceClasses[phase.source] || "wam");
    button.type = "button";
    button.style.flex = phase.end - phase.start + 1;
    button.title = `${sourceNames[phase.source] || phase.source}: ${phase.start}–${phase.end}`;
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => setStep(phase.start, true));
    return button;
  }));
  $("timeline").replaceChildren(...episode.timeline.map((event) => {
    const button = node("button", eventNames[event.type] || event.type, event.type);
    button.type = "button";
    button.append(node("small", `STEP ${event.step} · ${fixed(event.step * episode.dt, 2)} s`));
    button.addEventListener("click", () => setStep(event.step, true));
    return button;
  }));
  $("attempt-select").replaceChildren(...episode.attempts.map((attempt, index) => {
    const option = node("option", `#${attempt.number} · ${attempt.proposal.correction_mode.toUpperCase()}`); option.value = index; return option;
  }));
  $("failure-evidence").textContent = episode.failure ? `${episode.failure.failure_type} — ${episode.failure.evidence}` : "此实验未记录失败触发。";
  $("errors-title").textContent = `执行错误记录 · ${episode.errors.length} 条`;
  $("errors").textContent = episode.errors.length ? JSON.stringify(episode.errors, null, 2) : "没有记录到执行错误。所有失败尝试仍展示在尝试列表中。";
  $("segment-select").replaceChildren(...episode.segments.map((segment, index) => {
    const option = node("option", `${String(index + 1).padStart(2, "0")} · Step ${segment.start} · ${sourceNames[segment.source] || segment.source}`); option.value = index; return option;
  }));
  const labels = { manifest: "Manifest", events: "事件 JSONL", trajectory: "轨迹 HDF5", verification: "独立校验 JSON", video: "完整视频" };
  $("download-links").replaceChildren(...Object.entries(episode.downloads).map(([kind, url]) => {
    const link = node("a", `↓ ${labels[kind] || kind}`); link.href = url; link.download = ""; return link;
  }));
  activeStep = -1;
  setStep(0);
  renderAttempt();
  renderSegment();
  $("content").hidden = false;
}
async function load(id) {
  $("content").hidden = true;
  message("正在读取实际执行记录…");
  episode = await getJSON(`/api/recovery/${encodeURIComponent(id)}`);
  render();
  history.replaceState(null, "", `/recovery?episode=${encodeURIComponent(id)}`);
  message("");
}
async function refresh() {
  if (loading) return;
  loading = true;
  $("refresh").disabled = true;
  $("episode-select").disabled = true;
  try {
    const previous = episode ? $("episode-select").value : new URLSearchParams(location.search).get("episode");
    const { episodes } = await getJSON("/api/recovery");
    $("episode-select").replaceChildren(...episodes.map((item) => {
      const option = node("option", `${item.task || "读取异常"} · ${item.id}`); option.value = item.id; return option;
    }));
    if (!episodes.length) {
      $("content").hidden = true;
      message("还没有归档的 Recovery Episode。完成实验并 finalize 后，在这里刷新查看。");
      return;
    }
    if (episodes.some((item) => item.id === previous)) $("episode-select").value = previous;
    await load($("episode-select").value);
  } catch (error) {
    $("content").hidden = true;
    message(`无法读取实验：${error.message}`, true);
  } finally {
    loading = false;
    $("refresh").disabled = false;
    $("episode-select").disabled = !$("episode-select").options.length;
  }
}
$("refresh").addEventListener("click", refresh);
$("episode-select").addEventListener("change", refresh);
$("camera-select").addEventListener("change", updateCamera);
$("attempt-select").addEventListener("change", renderAttempt);
$("segment-select").addEventListener("change", renderSegment);
$("step-slider").addEventListener("input", (event) => setStep(Number(event.target.value), true));
$("episode-video").addEventListener("timeupdate", () => {
  if (episode) setStep(Math.floor(($("episode-video").currentTime + 0.00001) / episode.dt));
});
$("segment-jump").addEventListener("click", () => {
  if (selectedSegment) { setStep(selectedSegment.start, true); $("playback").scrollIntoView(); }
});
for (const video of document.querySelectorAll("video")) {
  video.addEventListener("error", () => message("视频无法加载。请检查媒体文件是否存在、浏览器是否支持该视频编码。", true));
}
for (const image of document.querySelectorAll("img")) {
  image.addEventListener("error", () => message("实际观测图像加载失败，请检查实验文件是否完整。", true));
}
refresh();
