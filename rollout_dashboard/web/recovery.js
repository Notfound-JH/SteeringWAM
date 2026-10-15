"use strict";

const $ = (id) => document.getElementById(id);
const sourceNames = { initial: "初始状态", wam: "WAM 执行", agent_correction: "Agent 修正", agent_condition: "Agent 改写条件后的 WAM", recovery_evaluation: "WAM 纠错评估窗口", agent_control: "Agent 直接控制", wam_continuation: "恢复成功后 WAM 续跑", wam_after_budget: "预算耗尽后 WAM（恢复未成功）" };
const eventNames = { failure: "发现失败", command_error: "执行异常", proposal: "Agent 介入", assessment: "局部评估", handoff: "交还 WAM", budget_handoff: "预算耗尽 → WAM", success: "任务成功", condition_change: "条件变更" };
const conditionKinds = { initial: "首次条件", unchanged: "条件不变", recovery_rewrite: "纠错改写", subgoal_progression: "子目标推进" };
const cameraNames = { wrist_cam: "手腕相机", over_shoulder_left_camera: "左肩视角", over_shoulder_right_camera: "右肩视角", head_camera: "头部视角" };
let episode = null;
let activeStep = -1;
let selectedSegment = null;
let loading = false;
let failureLinks = [];

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
function previewURL(url) {
  const preview = new URL(url, window.location.href);
  preview.searchParams.set("preview", "true");
  return preview.pathname + preview.search;
}
function setVideoQuality(preservePosition = false) {
  if (!episode) return;
  const video = $("episode-video");
  const time = preservePosition ? video.currentTime : 0;
  const resume = preservePosition && !video.paused;
  const preview = $("video-quality").value === "preview";
  video.pause();
  video.onloadedmetadata = () => {
    video.currentTime = Math.min(time, video.duration);
    if (resume) video.play().catch((error) => message(`播放未恢复：${error.message}`, true));
  };
  video.preload = preview ? "auto" : "metadata";
  video.src = preview ? previewURL(episode.video) : episode.video;
  $("video-quality-note").textContent = preview
    ? "显示压缩浏览副本，最高 640px 宽；原始录像和训练数据未修改。"
    : "显示原始录像，需要更多传输带宽。";
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
  if ($("episode-video").paused || seek) updateCamera();
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
  updateAxisStep(step);
}
function focusAttempt(index) {
  $("attempt-select").value = index;
  renderAttempt();
  $("correction").scrollIntoView({ block: "start" });
}
function updateAxisStep(step) {
  $("axis-step").textContent = `STEP ${step} / ${episode.steps}`;
  $("axis-playhead").style.left = `${step / Math.max(1, episode.steps) * 100}%`;
  for (const band of document.querySelectorAll(".axis-band")) {
    band.setAttribute("aria-current", String(step >= Number(band.dataset.start) && step <= Number(band.dataset.end)));
  }
  const index = episode.segments.findIndex((segment) => step > segment.start && step <= segment.end);
  const attemptIndex = episode.attempts.findIndex((attempt) => attempt.action_ranges.some((range) => step >= range.start && step <= range.end));
  $("axis-detail").hidden = index < 0 && attemptIndex < 0;
  if (index >= 0) {
    const segment = episode.segments[index];
    $("axis-selection-title").textContent = `WAM · Chunk ${index + 1} · 实际动作 ${segment.start + 1}–${segment.end} · ${sourceNames[segment.source] || segment.source}`;
    $("axis-selection-text").textContent = `输入指令：${segment.condition ?? segment.prompt}`;
    $("axis-detail").textContent = "查看完整 Prompt 与预测";
    $("axis-detail").onclick = () => {
      $("segment-select").value = index;
      renderSegment();
      $("predictions").scrollIntoView({ block: "start" });
    };
  } else if (attemptIndex >= 0) {
    const attempt = episode.attempts[attemptIndex];
    $("axis-selection-title").textContent = `Agent · #${attempt.number} ${attempt.proposal.correction_mode.toUpperCase()} · 受限控制器实际动作`;
    $("axis-selection-text").textContent = `纠错理由：${attempt.proposal.mode_reason || attempt.proposal.diagnosis || "历史记录未保存理由。"}\n这些步不是 WAM 推理输出；对应的 WAM 评估另列在蓝色轨道。`;
    $("axis-detail").textContent = "查看 Agent 提案与实际响应";
    $("axis-detail").onclick = () => focusAttempt(attemptIndex);
  } else {
    $("axis-selection-title").textContent = step === 0 ? "初始观测 · 尚未执行动作" : `${sourceNames[episode.observations[step].source] || episode.observations[step].source} · STEP ${step}`;
    $("axis-selection-text").textContent = step === 0
      ? "动作区间 1–N 表示执行第 1 到第 N 个命令；Step N 的错误节点表示在执行完第 N 步之后发现问题。"
      : "此步缺少对应的已归档推理块或控制提案，不能还原其输入指令 / 理由。";
  }
}
function axisBand(parent, start, end, label, title, className, onClick) {
  const button = node("button", label, `axis-band ${className}`);
  button.type = "button";
  button.dataset.start = start;
  button.dataset.end = end;
  button.style.left = `${(start - 1) / Math.max(1, episode.steps) * 100}%`;
  button.style.width = `${(end - start + 1) / Math.max(1, episode.steps) * 100}%`;
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", () => { setStep(start, true); if (onClick) onClick(); });
  parent.append(button);
  return button;
}
function drawFailureConnections() {
  const svg = $("failure-connectors");
  const bounds = $("axis-canvas").getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
  const element = (tag, attributes) => {
    const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => item.setAttribute(key, value));
    return item;
  };
  const defs = element("defs", {});
  const arrow = element("marker", { id: "failure-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" });
  arrow.append(element("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#b95947" }));
  defs.append(arrow);
  svg.append(defs);
  for (const { marker, card } of failureLinks) {
    const origin = marker.querySelector("i").getBoundingClientRect();
    const target = card.getBoundingClientRect();
    svg.append(element("line", {
      x1: origin.left + origin.width / 2 - bounds.left, y1: origin.bottom - bounds.top,
      x2: target.left + target.width / 2 - bounds.left, y2: target.top - bounds.top - 2,
      stroke: "#b95947", "stroke-width": 1.5, "marker-end": "url(#failure-arrow)",
    }));
  }
}
function renderAxis() {
  const colors = ["#426981", "#477d70", "#766398", "#9a6539", "#8f536a"];
  const conditions = new Map();
  const total = Math.max(1, episode.steps);
  $("axis-ruler").replaceChildren();
  const interval = Math.max(1, Math.ceil(total / 60) * 10);
  const ticks = [];
  for (let step = 0; step < episode.steps; step += interval) ticks.push(step);
  ticks.push(episode.steps);
  for (const step of ticks) {
    const tick = node("span", step);
    tick.style.left = `${step / total * 100}%`;
    $("axis-ruler").append(tick);
  }
  for (const id of ["wam-lane", "agent-lane", "condition-lane", "condition-cards", "failure-rail", "failure-cards"]) $(id).replaceChildren();
  episode.segments.forEach((segment, index) => {
    if (!segment.executed_steps) return;
    if (!conditions.has(segment.prompt)) conditions.set(segment.prompt, {
      label: `C${conditions.size + 1}`, color: colors[conditions.size % colors.length],
      segment, ranges: [], changes: [],
    });
    const condition = conditions.get(segment.prompt);
    const previous = condition.ranges.at(-1);
    if (previous && previous.end === segment.start) previous.end = segment.end;
    else condition.ranges.push({ start: segment.start + 1, end: segment.end });
    if (segment.condition_change && segment.condition_change.kind !== "unchanged") condition.changes.push(segment.condition_change);
    axisBand($("wam-lane"), segment.start + 1, segment.end, String(index + 1).padStart(2, "0"),
      `WAM Chunk ${index + 1} · ${segment.start + 1}–${segment.end} · ${condition.label}\n${sourceNames[segment.source] || segment.source}\n${segment.prompt}`, "wam-band");
  });
  episode.attempts.forEach((attempt) => {
    for (const range of attempt.action_ranges) axisBand($("agent-lane"), range.start, range.end, `A${attempt.number}`,
      `Agent #${attempt.number} · ${attempt.proposal.correction_mode.toUpperCase()} · 实际动作 ${range.start}–${range.end}`, "agent-band");
  });
  for (const condition of conditions.values()) {
    const { segment } = condition;
    const card = node("article", undefined, "condition-card");
    card.style.setProperty("--condition-color", condition.color);
    card.append(node("h4", `${condition.label} · ${segment.condition == null ? "完整模型 Prompt" : "输入 Condition"}`));
    card.append(node("p", `WAM STEPS ${condition.ranges.map((range) => `${range.start}–${range.end}`).join(" · ")}`, "condition-ranges"));
    card.append(node("p", segment.condition ?? segment.prompt));
    for (const change of condition.changes) card.append(node("p",
      `${conditionKinds[change.kind] || change.kind}${change.reason ? `：${change.reason}` : ""}`, "muted"));
    const details = node("details");
    details.append(node("summary", "完整模型 Prompt（记录原文）"), node("pre", segment.prompt));
    card.append(details);
    $("condition-cards").append(card);
    for (const range of condition.ranges) {
      const band = axisBand($("condition-lane"), range.start, range.end, condition.label,
        `${condition.label} · WAM ${range.start}–${range.end}\n${segment.prompt}`, "condition-band", () => {
          document.querySelectorAll(".condition-card").forEach((item) => item.classList.remove("selected"));
          card.classList.add("selected");
        });
      band.style.setProperty("--condition-color", condition.color);
    }
  }
  const errors = episode.timeline.filter((event) => ["failure", "command_error"].includes(event.type));
  failureLinks = [];
  $("axis-canvas").style.minWidth = `${Math.max(860, errors.length * 360)}px`;
  $("failure-cards").style.gridTemplateColumns = `repeat(${Math.max(1, errors.length)}, minmax(0, 1fr))`;
  $("failure-cards").hidden = !errors.length;
  errors.forEach((event, index) => {
    const card = node("article", undefined, "failure-card");
    card.dataset.eventId = event.id;
    const marker = node("button", undefined, "failure-marker");
    marker.type = "button";
    marker.style.left = `${event.step / total * 100}%`;
    marker.setAttribute("aria-label", `${eventNames[event.type]} · Step ${event.step}`);
    marker.append(node("i"), node("span", `E${index + 1} · ${event.step}`));
    const select = () => {
      setStep(event.step, true);
      document.querySelectorAll(".failure-card").forEach((item) => item.classList.remove("selected"));
      card.classList.add("selected");
    };
    marker.addEventListener("click", select);
    $("failure-rail").append(marker);
    const title = node("button", `E${index + 1} · STEP ${event.step} · ${eventNames[event.type]}`, "failure-title");
    title.type = "button";
    title.addEventListener("click", select);
    card.append(title, node("h3", event.failure?.failure_type || event.operation || "执行异常"));
    const evidence = event.failure?.evidence ?? event.error;
    card.append(node("p", typeof evidence === "string" ? evidence : JSON.stringify(evidence)));
    if (event.failure?.subgoal) card.append(node("p", `当时子目标：${event.failure.subgoal}`, "muted"));
    const linked = episode.attempts.map((attempt, attemptIndex) => ({ attempt, attemptIndex })).filter(({ attempt }) => attempt.failure_id === event.id);
    if (!linked.length) card.append(node("p", "没有关联的 Agent 纠错提案。", "muted"));
    for (const { attempt, attemptIndex } of linked) {
      const item = node("div", undefined, "linked-attempt");
      const outcome = attempt.assessment?.local_recovery_success;
      const result = outcome === true ? "局部成功（Agent 评估）" : outcome === false ? "局部失败" : "未评估";
      const button = node("button", `#${attempt.number} ${attempt.proposal.correction_mode.toUpperCase()} · ${result}`);
      button.type = "button";
      button.addEventListener("click", () => { setStep(attempt.start, true); focusAttempt(attemptIndex); });
      item.append(button, node("p", `STEP ${attempt.start} 介入 · 执行 / 评估至 ${attempt.end}`, "muted"));
      item.append(node("p", `Agent 模式选择理由：${attempt.proposal.mode_reason || "历史记录未保存独立的模式选择理由。"}`, "attempt-reason"));
      const details = node("details");
      details.append(node("summary", "诊断、恢复目标与实际结果"));
      for (const [label, text] of [
        ["诊断", attempt.proposal.diagnosis], ["恢复目标", attempt.proposal.recovery_goal],
        ["评估证据", attempt.assessment?.evidence],
      ]) details.append(node("p", `${label}：${text || "未记录"}`));
      item.append(details);
      card.append(item);
    }
    $("failure-cards").append(card);
    failureLinks.push({ marker, card });
  });
  const notes = [];
  if (!errors.length) notes.push("没有记录到失败触发或执行异常；不会把局部评估结果伪造成新的错误节点。");
  if (!conditions.size) {
    notes.push("没有已完成的 WAM 推理块，无法展示其输入指令。");
    $("condition-cards").append(node("p", "未记录 WAM 指令。Agent-only 实验的动作只显示在 Agent 轨道。", "muted"));
  }
  $("axis-empty-note").hidden = !notes.length;
  $("axis-empty-note").textContent = notes.join(" ");
}
function updateCamera() {
  if (!episode || activeStep < 0) return;
  const url = episode.observations[activeStep].frames[$("camera-select").value];
  if (!url) {
    $("observation-image").hidden = true;
    $("original-observation").hidden = true;
    $("observation-step").textContent = `图像 Step ${activeStep} · 缺失`;
    message("此控制步缺少所选相机的实际观测。", true);
    return;
  }
  $("observation-image").hidden = false;
  $("original-observation").hidden = false;
  $("observation-image").src = previewURL(url);
  $("original-observation").href = url;
  $("observation-step").textContent = `图像 Step ${activeStep} · 浏览预览`;
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
  const failure = episode.timeline.find((event) => event.id === attempt?.failure_id)?.failure;
  $("failure-evidence").textContent = failure
    ? `${failure.failure_type} — ${failure.evidence}` : "没有记录与当前尝试关联的失败触发。";
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
  $("before-image").src = previewURL(before[camera] || before.wrist_cam);
  $("after-image").src = previewURL(after[camera] || after.wrist_cam);
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
  $("prediction-input").src = previewURL(selectedSegment.input);
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
  const agentSteps = episode.observations.filter((item) => ["agent_correction", "agent_control"].includes(item.source)).length;
  const continuation = episode.observations.filter((item) => item.source === "wam_continuation").length;
  const afterBudget = episode.observations.filter((item) => item.source === "wam_after_budget").length;
  $("step-count").textContent = episode.steps;
  $("phase-counts").textContent = `Agent ${agentSteps} 步 · 恢复后 WAM ${continuation} 步 · 预算后 WAM ${afterBudget} 步`;
  $("attempt-count").textContent = agentOnly ? `${episode.attempts.length} 个控制程序` : `${episode.attempts.length} / ${episode.attempt_limit}`;
  $("budget-policy").textContent = agentOnly ? "Agent-only 控制程序，不计 Recovery 尝试。" :
    `预算耗尽：${episode.budget_exhaustion_policy === "continue_wam" ? "返回 WAM，不标作恢复成功" : "停止纠错"} · 总上限 ${episode.max_episode_steps ?? "沿用原任务"} 步`;
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
  setVideoQuality();
  $("step-slider").max = episode.steps;
  $("camera-select").replaceChildren(...Object.keys(episode.observations[0].frames).map((camera) => {
    const option = node("option", cameraNames[camera] || camera); option.value = camera; return option;
  }));
  if (episode.observations[0].frames.over_shoulder_left_camera) $("camera-select").value = "over_shoulder_left_camera";
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
  $("errors-title").textContent = `执行错误记录 · ${episode.errors.length} 条`;
  $("errors").textContent = episode.errors.length ? JSON.stringify(episode.errors, null, 2) : "没有记录到执行错误。所有失败尝试仍展示在尝试列表中。";
  $("segment-select").replaceChildren(...episode.segments.map((segment, index) => {
    const option = node("option", `${String(index + 1).padStart(2, "0")} · Step ${segment.start} · ${sourceNames[segment.source] || segment.source}`); option.value = index; return option;
  }));
  const labels = { manifest: "Manifest", events: "事件 JSONL", trajectory: "轨迹 HDF5", verification: "独立校验 JSON", video: "完整视频" };
  $("download-links").replaceChildren(...Object.entries(episode.downloads).map(([kind, url]) => {
    const link = node("a", `↓ ${labels[kind] || kind}`); link.href = url; link.download = ""; return link;
  }));
  renderAxis();
  activeStep = -1;
  setStep(0);
  renderAttempt();
  renderSegment();
  $("content").hidden = false;
  requestAnimationFrame(drawFailureConnections);
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
$("video-quality").addEventListener("change", () => setVideoQuality(true));
$("attempt-select").addEventListener("change", renderAttempt);
$("segment-select").addEventListener("change", renderSegment);
new ResizeObserver(drawFailureConnections).observe($("axis-canvas"));
$("step-slider").addEventListener("input", (event) => setStep(Number(event.target.value), true));
$("episode-video").addEventListener("timeupdate", () => {
  if (episode) setStep(Math.floor(($("episode-video").currentTime + 0.00001) / episode.dt));
});
for (const event of ["pause", "seeked"]) {
  $("episode-video").addEventListener(event, () => {
    if (!episode) return;
    setStep(Math.floor(($("episode-video").currentTime + 0.00001) / episode.dt));
    updateCamera();
  });
}
$("segment-jump").addEventListener("click", () => {
  if (selectedSegment) { setStep(selectedSegment.start, true); $("playback").scrollIntoView(); }
});
for (const video of document.querySelectorAll("video")) {
  video.addEventListener("error", () => message("视频无法加载；若流畅预览生成失败，可切换原始录像。请检查媒体文件与浏览器编码支持。", true));
}
for (const image of document.querySelectorAll("img")) {
  image.addEventListener("error", () => message("实际观测图像加载失败，请检查实验文件是否完整。", true));
}
refresh();
