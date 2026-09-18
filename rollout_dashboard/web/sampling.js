const $ = (id) => document.getElementById(id);
const state = { run: null, data: null, visible: [], selected: null, loading: false };
const verdicts = { accept: "接受", reject: "拒绝", uncertain: "不确定" };
const aliases = { fruits: "水果", tools: "工具", blocks: "积木", black_items: "黑色物品", organic: "有机物", toys: "玩具" };
const text = (id, value) => { $(id).textContent = value ?? "未记录"; };

async function get(url) {
    if (window.SAMPLING_SNAPSHOT) return url === "/api/sampling"
        ? { runs: [{ id: window.SAMPLING_SNAPSHOT.id }] } : window.SAMPLING_SNAPSHOT;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`读取失败 (${response.status})`);
    return response.json();
}
function option(value, label) { const item = document.createElement("option"); item.value = value; item.textContent = label; return item; }
function selectCandidate(id) { state.selected = id; renderRows(); renderDetail(); }
function filtered() {
    const task = $("task").value, verdict = $("verdict").value;
    state.visible = (state.data?.candidates || []).filter(item => (!task || item.scope === task) &&
        (!verdict || (verdict === "selected" ? item.selected : verdict === "pending" ? !item.review : item.review?.verdict === verdict)));
    if (!state.visible.some(item => item.id === state.selected)) state.selected = state.visible[0]?.id || null;
    renderRows(); renderDetail();
}
function renderRows() {
    $("rows").replaceChildren(); text("visible-count", `${state.visible.length} 条`);
    for (const item of state.visible) {
        const row = document.createElement("tr"); row.tabIndex = 0; row.classList.toggle("active", item.id === state.selected);
        const values = [`${aliases[item.task_alias] || item.task_alias} · 轮 ${item.chunk + 1}`, `#${item.attempt + 1} / 批${item.batch}`,
        item.selected ? "已选中" : (verdicts[item.review?.verdict] || (item.status === "failed" ? "执行失败" : "待审阅")),
        item.video ? "可播放" : item.status === "complete" ? "生成中" : "无完整记录"];
        values.forEach((value, index) => {
            const cell = document.createElement("td"); cell.textContent = value;
            if (index === 0 && item.scope.includes("retries/")) { const note = document.createElement("small"); note.textContent = "补采组"; cell.append(note); }
            if (index === 2) cell.className = item.review?.verdict || ""; row.append(cell);
        });
        row.addEventListener("click", () => selectCandidate(item.id));
        row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectCandidate(item.id); } });
        $("rows").append(row);
    }
}
function renderDetail() {
    const item = state.visible.find(candidate => candidate.id === state.selected);
    $("detail").hidden = !item;
    if (!item) return;
    text("title", `${aliases[item.task_alias] || item.task_alias} / 轮 ${item.chunk + 1} / 候选 ${item.attempt + 1}`);
    text("badge", item.selected ? "已提交" : "候选分支");
    const video = $("video");
    if (video.dataset.url !== (item.video || "")) {
        video.pause(); video.removeAttribute("src"); video.dataset.url = item.video || "";
        video.poster = item.poster || item.sheet || "";
        if (item.video) video.src = item.video; video.load();
    }
    video.playbackRate = Number($("speed").value);
    $("video-state").hidden = Boolean(item.video);
    text("video-state", item.failure || (item.status === "complete" ? "视频生成中" : "尚无完整执行记录"));
    $("download").hidden = !item.video; $("download").href = item.video || "#";
    $("sheet").hidden = !item.sheet; $("sheet").href = item.sheet || "#";
    text("instruction", item.instruction.instruction);
    text("subgoal", item.instruction.subgoal);
    text("condition", `${item.instruction.completion_condition || "未记录"}\n避免：${item.instruction.avoid || "未记录"}`);
    text("evidence", item.review ? `${verdicts[item.review.verdict] || item.review.verdict}\n${item.review.evidence}` : "待 Copilot 审阅，尚无判断输出");
    const completed = item.review ? (item.review.subgoal_complete ? "已证明" : "未证明") : "待审";
    text("completion", `子目标：${completed}；原生 checker：${item.native_success === true ? "成功" : item.native_success === false ? "未成功" : "未记录"}`);
    text("reason", item.selection_reason || "未选中 / 尚无选优记录"); text("prompt", item.prompt);
    text("candidate-id", `${item.id} · 实际执行 ${item.executed_steps}/32 步 · 未执行尾部未验证`);
    const index = state.visible.indexOf(item); $("previous").disabled = index === 0; $("next").disabled = index === state.visible.length - 1;
}
function showRun() {
    const prior = $("task").value; $("task").replaceChildren(option("", "全部任务"));
    for (const task of state.data.tasks) $("task").append(option(task.scope, `${aliases[task.alias] || task.alias}${task.extra_pool ? " · 补采" : ""} · ${task.status} · ${task.chunk}/${task.stop_chunk}`));
    if ([...$("task").options].some(item => item.value === prior)) $("task").value = prior;
    const counts = state.data.counts; $("summary").replaceChildren();
    for (const [label, value] of [["完整候选", counts.complete], ["视频", counts.videos], ["已审阅", counts.reviewed], ["已选中", counts.selected]]) {
        const item = document.createElement("span"); item.append(`${label} `); const strong = document.createElement("strong"); strong.textContent = value; item.append(strong); $("summary").append(item);
    }
    const updated = document.createElement("span"); updated.textContent = `读取于 ${new Date().toLocaleTimeString()}`; $("summary").append(updated);
    $("progress").replaceChildren(); for (const record of state.data.progress) { const pre = document.createElement("pre"); pre.textContent = record.text; $("progress").append(pre); }
    text("console", state.data.console_tail || "尚无控制者输出"); text("lifecycle", state.data.lifecycle || "尚无进程记录");
    filtered();
}
async function refresh() {
    if (state.loading || !state.run) return; state.loading = true; $("refresh").disabled = true;
    try { state.data = await get(`/api/sampling/${encodeURIComponent(state.run)}`); $("error").hidden = true; showRun(); }
    catch (error) { text("error", error.message); $("error").hidden = false; }
    finally { state.loading = false; $("refresh").disabled = false; }
}
$("run").addEventListener("change", () => { state.run = $("run").value; state.selected = null; $("task").value = ""; history.replaceState(null, "", `?session=${encodeURIComponent(state.run)}`); refresh(); });
$("task").addEventListener("change", filtered); $("verdict").addEventListener("change", filtered);
$("refresh").addEventListener("click", refresh); $("speed").addEventListener("change", () => { $("video").playbackRate = Number($("speed").value); });
for (const [id, delta] of [["previous", -1], ["next", 1]]) $(id).addEventListener("click", () => { const index = state.visible.findIndex(item => item.id === state.selected); if (state.visible[index + delta]) selectCandidate(state.visible[index + delta].id); });
if (window.SAMPLING_SNAPSHOT) { $("auto").disabled = true; $("refresh").hidden = true; }
window.setInterval(() => { if ($("auto").checked && !document.hidden) refresh(); }, 15000);
(async () => {
    try {
        const data = await get("/api/sampling"); for (const run of data.runs) $("run").append(option(run.id, run.id));
        const requested = new URLSearchParams(location.search).get("session"); state.run = data.runs.some(run => run.id === requested) ? requested : data.runs[0]?.id;
        if (!state.run) { text("error", "尚无指令采样实验"); $("error").hidden = false; return; } $("run").value = state.run; await refresh();
    } catch (error) { text("error", error.message); $("error").hidden = false; }
})();