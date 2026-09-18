# RoboLab + Cosmos3-Edge: 外部视觉 Agent 探索与采样

本文面向接管实验的 agent。实现与命令按 2026-09-16 的本地代码核对。
这是现有 RoboLab 安装上的推理流程，不是训练流程，也不是开箱通用的任意任务搜索器。
历史分支搜索任务固定为 `ToolsPickingAllHammersTask`；下面的新 session harness 另行支持六任务队列，不能与历史 session 混用。

## 四路指令采样 Harness（新实现，2026-09-17）

### 初始姿态修复与检查模型

已复现旧入口在 reset 后驱动目标仍为零，8个原始物理预热步把机械臂拉向零位的问题。
当前入口显式启用 `reset_joint_targets=True`，每个预热步写入场景控制目标；检查点恢复后
同步机器人位置目标并清空速度/力矩目标。模型生成动作本身不修改。
水果单环境初始化、工具单环境初始化/重复reset/检查点恢复已生成实测证据；旧实验保持原样，
不能更改 hash 后直接续跑。诊断不加载模型，可用 `--initialization-diagnostic --task ...`。
详细证据及真实4x3指令/seed核对位于当前实验的 `media/initial_pose_fix/REPORT.md`。
历史各组并不统一：水果/工具初轮12条不同文本，但多个任务/后续轮次是同文不同seed。
`audit_sampling_prompts.py --session ... --output ...` 从实际请求与服务日志生成只读核对报告。
后续 Copilot 检查明确指定 `--model gpt-6-astra --reasoning-effort low`，不得自动替换其他模型。

入口：[instruction_sampling.py](instruction_sampling.py)；持久 worker：
[instruction_sampling_worker.py](instruction_sampling_worker.py)。仅由 Cosmos 生成动作，
不包含 Agent 直接动作或动作修正。不是缺失旧实现的恢复版本，不读取或迁移旧 session。

| 顺序 | CLI 任务名 | 原生任务 | 最多提交轮数 | 物理 GPU |
| --- | --- | --- | --- | --- |
| 首批并行 | `fruits` | `FruitsOnPlate3Task` | 8 | 0,1,2,3 |
| 首批并行 | `tools` | `ToolsPickingAllHammersTask` | 8 | 4,5,6,7 |
| 随后依次 | `blocks` | `BlocksInBinTask` | 10 | 0,1,2,3 |
| 随后依次 | `black_items` | `BlackItemsInBinTask` | 10 | 0,1,2,3 |
| 随后依次 | `organic` | `ClearOrganicObjectsTask` | 10 | 0,1,2,3 |
| 随后依次 | `toys` | `CleanUpToysTask` | 10 | 0,1,2,3 |

每张卡独立加载一个模型并绑定一个环境的渲染器；PhysX 沿用 CPU。
每个任务从同一已提交起点并行执行四个候选，人工审阅后才能再开下一批。
三批共 12 个候选中选一个已接受候选提交，再进入下一轮；不是把四段动作串起来执行。
原生全任务成功可提前选择/停止。每组固定 12 条；用户现授权第一组全无 accept 时，
从同一已提交检查点再试一组 12 条，最多补一次。第二组仍全无可接受候选则停止该起点。
预测 32 步、执行前 16 步；原生终止可缩短有效长度，未执行尾部未经验证。
任务总指令和 checker 不变，Agent 可在提交轮边界自由选择下一子目标。
同轮四个候选可使用四种措辞，批间可根据实际失败证据改写，但仍针对同一子目标和起点。

启动前需要可用的八卡、两个互不冲突的八端口区间，以及不存在的独立目录。
`start` 会检查既有 compute 进程，拒绝共享或终止其他作业；不会复用不明模型服务。
用户明确授权共享时，追加 `--allow-gpu-sharing`；此授权会记录到 experiment manifest 并用于后续任务切换。
该开关不会终止、重配或接管其他进程，也不保证共享时的峰值显存或性能。
模型基准种子为 `3407 + GPU编号`，服务内部 RNG 随请求推进，场景种子固定 3407。
推理参数沿用当前启动脚本；采样模式仅关闭预测视频解码，保留实际相机帧。

```bash
cd /home/aiscuser/SteeringWAM
PY="$PWD/RoboLab/.venv/bin/python"
SESSION="$PWD/diagnostics/instruction_search_new"
"$PY" instruction_sampling.py start --session "$SESSION" --base-port 18840 --master-port 29840
"$PY" instruction_sampling.py status --session "$SESSION" --task fruits
```

服务 PID、启动命令、卡号、端口、源码身份及日志保存在新目录中。
八个模型和首批两个任务就绪后 `start` 返回；后续必须由外部 Agent 逐批看图、写 decision，
持久服务存活不代表候选在自动采样。水果与工具的 `run-batch` 可由两个独立终端同时调用，
同一任务禁止第二个控制者；状态转换持有任务文件锁并消费最新 `request_id`。

每轮首先看 `status` 的检查点真实图像，再用文件编辑工具写独立 instruction decision：

```json
{
  "request_id": "CURRENT_TOKEN",
  "instruction": {
    "subgoal": "VISUALLY_CHOSEN_SUBGOAL",
    "instruction": "SHORT ENGLISH INSTRUCTION BASED ON CURRENT IMAGES",
    "completion_condition": "OBSERVABLE COMPLETION CONDITION",
    "avoid": "ONE IMPORTANT CONSTRAINT"
  }
}
```

```bash
"$PY" instruction_sampling.py instruct --session "$SESSION" --task fruits --decision /absolute/path/instruction.json
"$PY" instruction_sampling.py status --session "$SESSION" --task fruits
"$PY" instruction_sampling.py run-batch --session "$SESSION" --task fruits --decision /absolute/path/batch.json
"$PY" instruction_sampling.py status --session "$SESSION" --task fruits
"$PY" instruction_sampling.py review --session "$SESSION" --task fruits --decision /absolute/path/review.json
```

`batch.json` 至少包含最新 `request_id`；可增加 `instructions` 数组，恰好四项、
每项与上例 `instruction` 同结构。缺省时四个候选共用当前指令，模型随机种子独立。
`review.json` 包含最新 `request_id` 和恰好四项 `reviews`，每项包含
`candidate_id`、`verdict`、`subgoal_complete`、`evidence`、`reviewed_frame_sha256`。
这些内容由 Agent 实际看图后填写，harness 不自动生成 accept、证据或已审阅 hash。

**简短审阅方式：** 用图像工具打开该批 `review_sheet`，每个候选保留五时刻的手腕/viewport
两视角，再逐候选写一句“实际变化；有用/有害；是否完成”。不确定时打开原分辨率图片，
不要靠缩略图猜。viewport 明确是调试视角，不宣称是真机观测。
下一批只解释最相关的指令修改；第三批后比较已接受候选，记录一句选优理由。

```bash
"$PY" instruction_sampling.py select --session "$SESSION" --task fruits --decision /absolute/path/select.json
"$PY" instruction_sampling.py export --session "$SESSION" --task fruits
"$PY" instruction_sampling.py export --session "$SESSION" --task tools
"$PY" instruction_sampling.py next-task --session "$SESSION"
```

选择文件字段为 `request_id`、`candidate_id`、`reason`。每次提交后重新读取状态和末态图片。
达到轮数/成功/协议停止后才导出；首批两个任务均停止并导出后 `next-task` 激活 `blocks`，
此后每个新增任务停止、导出，再调用一次 `next-task`。不会自动把未完成子目标标成成功。
`stop-services --session ...` 只停止此目录记录且 PID 启动身份仍匹配的服务。

新版 `instruction-sampling-v3-visual-review` 按用户澄清，保留四卡三批、12 条中选一条末态进入下一轮。
恢复是重放与场景校正，不是完整 PhysX 快照。执行候选前实际写回所选检查点；
重放中的状态数值差、事件差和校正前后误差只记录诊断，不再用 `1e-5` 阻塞。
起点相机图像先落盘，图像 MAE 同样只记录；是否一致、是否有用由 Agent 看真实图像判断。
保留字段/形状/有限值、文件 hash、动作前缀和 token 校验。重放提前终止时不能在冻结环境上继续执行，
仍作为运行错误报告。这些内部状态不得用作 Agent 选动作或评分的依据。
历史 v1/v2 session 不迁移、不替换 hash，新版使用独立新 session，不声称数值轨迹精确复现。
失败留在 `execution_unknown`，禁止再开一批。`recover --session ... --task ...` 仅导入四个
有完整成功凭据的结果，不重跑；缺失或失败记录仍为技术阻塞。
导出保留全部已选动作、逐段来源/长度/评价、实际双视角视频，并解码检查视频。
所有候选的独立 HDF5、原始预测、执行动作、恢复轨迹和审阅记录留在原候选目录，不覆盖。

验证：根目录独立的四组检查不加载 Isaac，可运行：

```bash
"$PY" -m pytest test_instruction_sampling.py -q
```

当前四组检查通过。首次 v1 共享八卡运行在像素校验处停止，未生成或执行模型动作；
这些历史失败证据和 SAS 备份保留。用户随后澄清采用实际图像审阅，新版不以 MAE 阻塞。
八卡共享仍需显式 `--allow-gpu-sharing`，不终止既有训练。
既有 RoboLab 客户端十项测试通过，但其测试配置会初始化 Isaac 渲染，不能称作纯 CPU 测试；
测试进程已退出，没有产生本次任务候选。实际采样、模型/环境同卡峰值显存、跨卡图像一致性仍待验证。

### 一次额外十二候选组

单进程十二环境入口见下节；本节描述的仍是旧四路三批流程。

新增 [retry_instruction_sampling.py](retry_instruction_sampling.py) 独立入口，不修改已锁定的
父实验代码、状态或历史评价。只允许父任务处于 `stopped/no_acceptable_candidate`、12 条全部
执行完成并审阅且无 accept 时创建额外组；未知执行、缺失产物或已接受候选不能走补采入口。

创建独立 decision，字段为 `request_id`（最新父 token）、`user_request`（真实用户补采授权）、
`reason`（依据上组实际图像准备怎样调整指令），然后：

```bash
"$PY" retry_instruction_sampling.py --session "$SESSION" --task fruits --decision /absolute/path/retry.json
```

返回 `retry_session`，其目录在父实验根的 `retries/` 下。通过原 CLI 的 `--session RETRY_SESSION`
继续 `instruct/run-batch/review/select/export`，仍为四路三批。补采继承 genesis、已选动作、
当前 checkpoint、step/chunk 及总轮数上限，不从初始场景重新计数。完成后只对这个子目录
运行 `stop-services`，不在子目录执行 `next-task`；原任务队列由原实验控制者维护。
共享模型记录为借用，子目录不会停止父实验模型；所有新环境/自有模型 PID 均留痕。

同一检查点最多一个额外组，重复决策或目录已存在会拒绝，不允许删除目录后补第三组。
合法选择推进到新 checkpoint 后，该新起点可再享有一次额外组。父状态和导出保持原样，
子记录包含第一组历史、授权、父状态 hash 和额外入口代码身份。SAS 递归同步这些子目录。
当前控制者先完成原队列已有工作，再逐个补采此前全拒绝的任务，避免抢占正在使用的四卡。
判定仍以“对当前指令有可见贡献”为准，而不是必须一段内抓取或抬升成功。

### 单任务、单进程十二环境

新增 [sampling_multi_env.py](sampling_multi_env.py)，不修改旧会话锁定源码。
一个 Isaac 进程创建同一任务的 12 个环境，共同起点、各自指令；共享一个现有 Cosmos
WebSocket 服务，顺序发送 12 次推理请求，不做模型 batch/microbatch。收齐动作后，
每步以 `(12, 8)` 同步执行，默认执行模型 32 步预测的前 16 步。

指令 JSON 必须是 12 个对象组成的数组，每项包含 `subgoal`、`instruction`、
`completion_condition`、`avoid`，且共享同一 subgoal。可用的格式示例：
[multi_env_instructions.example.json](multi_env_instructions.example.json)。示例是 fruits 的
approach 指令，不代表已根据当前图像审阅或已执行的候选。

从 SteeringWAM 根目录运行；使用新的输出目录，端口指向已就绪的 Cosmos 服务：

```bash
export LD_LIBRARY_PATH="$PWD/packages/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RoboLab/.venv/bin/python sampling_multi_env.py \
  --task FruitsOnPlate3Task --sim-gpu 4 --policy-port 18745 \
  --instructions multi_env_instructions.example.json \
  --output-dir diagnostics/multi_env_round_001 --execute-steps 16
```

若 GPU 已有模型或其他计算进程，入口默认拒绝启动；检查余量后显式添加
`--allow-gpu-sharing`。入口不启动、停止或重启模型服务。要模型和仿真同卡，需确保该
端口的模型已部署在 `--sim-gpu` 指定的物理卡上。不要向正用于其他实验的模型服务发送
测试请求，以免改变其随机数序列。`--initialization-only` 只检查十二环境和相机，不请求
模型；该检查仍会占用仿真显存。`--no-videos` 跳过 MP4 编码，保留所有 PNG。

每轮输出 `candidate_00` 至 `candidate_11`，包含模型输入、原始/处理后动作、实际执行
前缀、逐步场景、相机 PNG、双视角 MP4、审阅拼图、结果和单环境 checkpoint。
共享 `rollout.hdf5` 内是独立 `demo_0` 至 `demo_11`，每路执行动作均与模型前缀校验。
提前终止的环境不再增加有效轨迹。只有根目录 `result.json` 的 `status=complete`
才表示整轮产物已完成校验；失败后保留目录，不自动重试。

人工/视觉审阅后，新一轮命令添加以下参数，并使用新的输出目录及新的十二条指令：

```bash
--checkpoint diagnostics/multi_env_round_001/candidate_03/checkpoint.json
```

续跑按 hash 验证父链，从 genesis 在十二环境回放已选动作，再校正各段场景状态。
拒绝续接已终止的候选；状态使用环境相对坐标，避免克隆环境原点偏移。这里只支持
此入口生成的 checkpoint，不迁移旧 session。当前是独立轮次执行入口，不自动做视觉
接受/选优，也尚未接入旧 dashboard、SAS 同步或旧会话的轮数/重试预算状态机。

验证：`RoboLab/.venv/bin/python -m pytest test_instruction_sampling.py -q`。
新增测试覆盖十二指令、状态复制、串行请求、提前终止、HDF5 一致性和父链续跑；
完整闭环使用替身环境。真实十二环境 GPU 初始化、相机画面和峰值显存仍待空闲资源实测。

### Copilot CLI 与 tmux 接管（旧会话）

本机已安装独立 GitHub Copilot CLI 1.0.86：`/home/aiscuser/.local/bin/copilot`。
2026-09-18 已通过官方设备码登录，并验证图片附件和 `view` 读取实际候选 PNG；
这些只读测试不是候选接受记录。凭据保存在用户本地，禁止写入实验产物或 SAS。
本机无系统密钥链，CLI 使用本地配置；目录/配置权限已检查为 700/600。

当前接管的是初态修复后的 `instruction_posefixed_20260918T055850Z`，不是历史 shared/visual/relaxed session。
唯一 CLI 会话 UUID：`80f0fbfd-23e8-4c78-aacb-86feb4d36f44`，显式模型 `gpt-6-astra`、effort `low`。
tmux 会话 `robolab-posefixed-055850` 内，`controller` 窗格运行 CLI，`sas` 窗格运行增量上传器。
八环境初始化均已核对关节初态；详见新 session 的 `INITIAL_POSE_VALIDATION.json`。
八份模型和八个环境属于独立新实验；历史已选、待审和失败候选全部保留，不移植或改写。
旧 visual/relaxed 控制者及其 GPU 服务已停止，不要恢复为第二个采样控制者。
旧 tmux 中的只读 videos/review 服务仍继续展示各实验结果，不是采样控制者。

```bash
tmux attach -t robolab-posefixed-055850
tmux capture-pane -p -t robolab-posefixed-055850:controller -S -60
```

接入后按 `Ctrl-b d` 仅离开 tmux，不终止任务。CLI 的运行轨迹保存在 session 的
`copilot_control/console.log`、`lifecycle.log` 和 `CLI_PROGRESS.md`；权威进度仍是经校验的
任务 `session.json`。`ACK.md` 是控制者确认，不是任务完成证据。
启动脚本使用固定 UUID、独立控制锁、受限工具/目录和有限 autopilot 次数，不使用
`--allow-all`、`--allow-all-paths` 或 `--yolo`，不自动循环重启失败实验。

如果控制者已退出，先查看退出码、真实进程和最后状态；特别是 `pending_review` 必须先审阅，
`execution_unknown` 不得重跑。仅在确认不存在旧控制者且没有未解决阻塞后，才可重新运行
该 session 内 `copilot_control/run_controller.sh`，它用相同 UUID 续接。
不要在控制者活着时另开 `copilot --resume` 或从编辑器同时提交决策。
服务进程存活、tmux 存活、ACK 存在均不等于采样成功，需核对真实 review/select 记录。

### 全部 Chunk 视频与 Copilot 输出

旧 `8765` 首页只连接历史 rollout；同端口的新增静态入口会自动更新最新实验快照，
可直接沿用已有 Dev Tunnel 地址打开 `/assets/sampling-latest.html`，不用重启旧服务。
所有实验的实时筛选页面另在独立端口：

```text
http://127.0.0.1:8766/sampling?session=instruction_relaxed_20260918T041558Z
```

远程 VS Code 需转发端口 `8766`，打开该端口地址后的 `/sampling`。页面按实验、任务、
判定筛选，逐候选显示实际执行双视角视频、Copilot 当前指令、完整模型 prompt、review
原文、子目标完成标记、是否选中和选优理由；页面底部还有 CLI 进度与最近控制者输出。
所有文本来自实际记录，不生成新的判定，不把接受或视频存在当作任务成功。

[rollout_dashboard/sampling_media.py](rollout_dashboard/sampling_media.py) 在 tmux 的 `videos`
窗格独立运行，串行补齐所有 `instruction_*` 实验及 `retries/` 中的已完成候选，包括被拒绝
候选。检查源图 hash，按 `control_dt` 编码从第0帧到实际有效末帧；不使用未执行预测尾部，
不插帧、不连接不同弃选分支。每个 MP4 验证帧数及完整解码后才发布，画面包含实际指令。
执行失败而没有完整帧的候选不会编造视频，页面会展示失败原因。

每个实验内的新增派生产物位于：

```text
media/chunks/<candidate_id>/actual.mp4
media/chunks/<candidate_id>/poster.jpg
media/chunks/<candidate_id>/manifest.json
media/copilot_outputs.json
media/review.html
media/video_status.json
```

`review.html` 为自包含离线页面，保持实验目录中的相对路径即可播放视频；
`copilot_outputs.json` 保存全部可见指令、审阅和选优输出。SAS 已有同步器会上传整个
`media/`，不会修改原候选 `result.json`、动作、hash 或 session 状态。
新增候选自动编码，已有视频命中缓存；页面可手动刷新或启用15秒自动刷新。

```bash
"$PY" rollout_dashboard/sampling_media.py --watch
"$PY" rollout_dashboard/server.py --host 0.0.0.0 --port 8766
"$PY" -m pytest test_sampling_media.py -q
```

这两个服务只是生成媒体和展示结果，不启动采样、不提交评价，也不接管 Copilot 控制者。

### 结果增量备份到 SAS

[sync_instruction_results.py](sync_instruction_results.py) 是独立进程，不修改 session、token 或锁定执行源码。
使用仓库外 SAS 文件和 Azure Blob SDK；凭据不进入命令参数、输出或备份状态。
默认每 5 秒检查一次，完整候选先核对并上传 `artifacts`，最后上传 `result.json`。
后续 review、selection、导出和日志继续增量上传。上传串行执行；网络错误保留待传状态并重试，
不重跑候选、不删除本地或远端文件。核对远端长度和 SHA-256 元数据，并启用传输内容校验。

```bash
"$PY" sync_instruction_results.py --session "$SESSION"
```

默认目标为 `jinghe/steeringwam/instruction_sampling/<session目录名>/`，可用 `--remote-prefix`
指定。上传状态保存在仓库外 `~/.blob-sync/instruction-results/<session目录名>/status.json`。
`completed_candidates` 表示已上传并校验的候选，`errors` 非空则仍需重试，不能声称备份完成。
失败分支也备份已有产物，并单独列入 `failed_candidates_backed_up`，不能计为完成候选。
`--once` 只补传一轮；重复启动同一上传器会被文件锁拒绝。每轮重读 SAS 文件，续期可替换私密文件。
不要把 SAS 粘贴到日志或决策文件。根目录验证命令为 `"$PY" -m pytest test_sync_instruction_results.py -q`。

同步延迟是检查间隔加上传耗时，不是零延迟；未完成分支的动作文件不会冒充已完成候选。
初始化及服务记录也会备份，但它们不代表成功采样。同步进程不会代替 Agent 看图、审阅或启动下一批。

## 当前机器恢复说明（2026-09-17）

当前 checkout 为 `/home/aiscuser/SteeringWAM`，下文路径已按此布局调整。
已按用户确认的范围打通基础视觉闭环：真实观测 -> Cosmos Policy-DROID -> RoboLab 执行动作 -> 更新后的真实观测。
本次验证 2 次规划、32 个控制步，不包含十候选分支搜索，也不表示完成了四锤任务。

- 官方 `NVlabs/RoboLab` 不包含本文使用的 `scripts/run_cosmos3_toolpickup_agent_loop.py` 和 `robolab/eval/toolpickup_agent_loop.py`。必须恢复原实验的定制 checkout 或对应补丁，不能用普通 policy client 替代分支搜索协议。
- 当前机器为 NVIDIA A100-SXM4-80GB，没有 RT Core，不满足官方 RTX 硬件要求，但本机已实测 CPU PhysX + GPU 4 渲染 + GPU 5 模型推理的基础闭环。A100 的 DLSS-RR 不受支持，输出有噪点；本机结果不改变官方支持范围。
- RoboDojo 微调 checkpoint 与 `Cosmos3-Edge-Policy-DROID` 是不同策略；本流程使用后者的 Panda 关节动作，不能直接替换为 RoboDojo 双臂 EE checkpoint。
- 基础闭环按下一节运行。第 6 节起的历史 `init/instruct/run-next/review/select` 控制器命令仍需恢复定制代码，不能与基础闭环混用。

### 已验证的基础闭环

最终运行目录为 [rollout_visual_N0CiC7](diagnostics/rollout_visual_N0CiC7/result.json)，
仿真进程退出码 0，`status=rollout_complete`、`closed_loop_complete=true`。
模型服务日志在 [server.log](diagnostics/cosmos_server_HBkg1u/server.log)。

- 模型为本地官方 `Cosmos3-Edge-Policy-DROID`，单卡 GPU 5；仿真为 CPU PhysX，渲染单卡 GPU 4，未占用 GPU 0-3。
- 2 次模型请求均返回 `(32,8)`，每次执行前 16 步。第二次输入使用第 16 步的真实图像和关节状态，而非重复初始观测。
- 32 个实际动作与处理后的模型前缀逐项一致，`rollout.hdf5` 的 `data/demo_0/actions` 也逐项一致。
- 手腕视图的结构随物理运动明显变化，已实际查看；不是只靠噪点或文件 hash 判断更新。
- 原生 `native_success=false`，没有触发任务终止；停止原因是本次 32 步预算结束，不报告抓取成功。

预览：[手腕视频](diagnostics/rollout_visual_N0CiC7/wrist_preview.mp4)、
[记录视角视频](diagnostics/rollout_visual_N0CiC7/viewport_preview.mp4)。
视频由第 4、8、...、32 步的真实帧组成，以 3.75 FPS 播放，对应 15 Hz 控制频率；不是完整逐步视频。
原始五路帧、每段输入图、状态和整段动作均保存在运行目录。

先在独立终端启动模型服务（默认 GPU 5，端口 18745；脚本拒绝 GPU 0-3）：

```bash
cd /home/aiscuser/SteeringWAM
MODEL_RUN="$(mktemp -d "$PWD/diagnostics/cosmos_server_XXXXXX")"
bash launch_cosmos3_robolab_server.sh "$MODEL_RUN" >"$MODEL_RUN/server.log" 2>&1
```

服务监听后，在另一个终端运行基础闭环。入口先请求一次 `/healthz`，服务未就绪就报错退出，不启动仿真：

```bash
cd /home/aiscuser/SteeringWAM
export CUDA_VISIBLE_DEVICES=GPU-9bda66e3-4891-0eb0-79dd-c3530fa33f8a
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$PWD/packages/ffmpeg/lib"
export PATH="$PWD/packages/ffmpeg/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
RUN="$(mktemp -d "$PWD/diagnostics/rollout_basic_XXXXXX")"
timeout --signal=TERM --kill-after=5s 595s \
  ./RoboLab/.venv/bin/python probe_isaac_camera.py \
  --task ToolsPickingAllHammersTask --rollout-chunks 2 --execute-steps 16 \
  --policy-port 18745 --output-dir "$RUN" >"$RUN/launch.log" 2>&1
printf 'Rollout directory: %s\n' "$RUN"
```

### FruitsOnPlate3Task 与在线看板

`FruitsOnPlate3Task` 已在本机使用同一基础闭环完成两次独立的 32 步检查：
[rollout_fruits_u8Az63](diagnostics/rollout_fruits_u8Az63/result.json) 和
[rollout_fruits_9fd59y](diagnostics/rollout_fruits_9fd59y/result.json)。两次均为
`status=rollout_complete`、`closed_loop_complete=true`，各包含 2 次模型请求和 32 个实际动作；
生成动作前缀、实际动作与 HDF5 记录逐项一致。两次 `native_success=false`，因此只验证了
水果任务的视觉闭环和数据记录，不表示三个水果已经放到盘中，也不证明策略能完成该任务。

运行水果任务只替换基础入口的任务名和输出目录；仿真仍固定使用物理 GPU 4，模型服务通常使用 GPU 5：

```bash
RUN="$(mktemp -d "$PWD/diagnostics/rollout_fruits_XXXXXX")"
timeout --signal=TERM --kill-after=5s 595s \
  ./RoboLab/.venv/bin/python probe_isaac_camera.py \
  --task FruitsOnPlate3Task --rollout-chunks 2 --execute-steps 16 \
  --policy-port 18745 --output-dir "$RUN" >"$RUN/launch.log" 2>&1
printf 'Rollout directory: %s\n' "$RUN"
```

入口默认使用 640 x 480 app renderer 和 Balanced DLSS（`--rendering-mode balanced`）。
此前的 320 x 240 + Performance 组合在 A100 上实际只有 160 x 120 的 DLSS 输入；提高后日志记录为
371 x 278，内部像素约为此前的 5.4 倍。真实 `FruitsOnPlate3Task` 相机检查保存在
[camera_fruits_hires_SksPeW](diagnostics/camera_fruits_hires_SksPeW/result.json)，状态为
`frame_captured`，手腕图像的颗粒明显减少。可显式覆盖参数：

```bash
./RoboLab/.venv/bin/python probe_isaac_camera.py \
  --task FruitsOnPlate3Task --render-width 1280 --render-height 720 \
  --rendering-mode quality --output-dir "$CAPTURE_DIR"
```

建议先使用默认的 640 x 480 + Balanced；1280 x 720 + Quality 会显著增加 GPU 时间、显存和
RTX 管线编译压力。A100 仍不支持当前构建的 DLSS-RR，因此提高分辨率只能减轻噪点，不能补回
RTX GPU 上可用的完整实时光追降噪能力。

每次命令创建独立仿真和独立初始状态；新目录不是从上一次 `final/` 物理状态续跑。
当前入口也没有 `--sim-gpu`，不能仅修改 `CUDA_VISIBLE_DEVICES` 就把并行仿真分配到 GPU 5-7；
需要先参数化并验证 renderer 的物理 GPU 绑定。模型服务可用不同的 `MODEL_GPU`、`PORT` 和
`MASTER_PORT` 分别启动，但这不等于仿真侧已经支持安全并行。

仓库根目录的 [rollout_dashboard](rollout_dashboard/server.py) 会扫描 `diagnostics/rollout_*`，
按 chunk 展示该次模型指令、模型生成的未来视频和对应动作实际执行后的 viewport 视频；页面只在打开、切换 run 或点击刷新时
读取结果，不持续轮询，也不控制仿真或重发动作。新 rollout 在 15 Hz 的每个控制步保存全部五路相机，
每段执行视频使用 chunk 起点和每个动作后的 `egocentric_mirrored_camera` 实际帧，不做插帧。
模型服务同时 VAE 解码 33 帧生成结果；runner 去掉第一个条件帧，将后续 32 帧以 15 FPS 编码为
`chunks/NNN/prediction.mp4`。页面左侧 `PREDICTED FUTURE · MODEL` 是模型生成结果，右侧
`EXECUTED · VIEWPORT` 是仿真实际执行记录，不能将二者混称。VAE 解码和约 33 MB 的本机
WebSocket 原始帧传输会增加每次规划的延迟与峰值内存。
旧 run 没有保存生成 latent，无法追溯恢复当时的预测视频，页面会明确显示未捕获；旧的稀疏执行运行仍按原始 3.75 Hz 采样间隔播放。
视频第一次访问时由本机 FFmpeg 编码为 H.264 并缓存到对应 `chunks/NNN/viewport.mp4`。
使用 RoboLab 环境启动：

```bash
cd /home/aiscuser/SteeringWAM
./RoboLab/.venv/bin/python rollout_dashboard/server.py \
  --host 0.0.0.0 --port 8765
```

本机可访问 `http://127.0.0.1:8765`。远程 VS Code 中打开“端口”面板，转发端口 `8765`，
再用面板给出的 Forwarded Address 访问；不要直接使用节点私网 IP `100.64.*`。
Private Dev Tunnel 会先要求 GitHub 登录，也可在端口面板按需要将可见性改为 Public。
浏览器出现“未能加载页面”或 502/504 时，先在节点检查：

```bash
ss -ltnp | grep ':8765'
curl -fsS http://127.0.0.1:8765/api/runs >/dev/null
```

两项通过表示服务正常，故障在端口转发或 Dev Tunnel；重新转发/刷新即可，不要因此重启 rollout。
页面不做实时轮询；新 chunk 落盘后手动点击刷新即可，避免慢速 Dev Tunnel 持续请求导致 504。

`--rollout-chunks 0`（默认）仅导出相机；大于 0 才调用模型并执行动作。
每个模型请求限时 240 秒且不自动重发，输出目录必须没有旧的 `result.json`，失败或超时后不自动续跑未知动作。
服务器不产生额外 Reasoner 文本，直接调用动作生成 API。服务增加了通用配置已有的
`guardrails` / `use_torch_compile` 透传开关；本机脚本关闭两者，不影响它们原本的默认值。
本地 Wan VAE 从 SAS 恢复位置链接到配置预期路径，启动时不再重复下载。

CPU 物理必须和渲染状态同步：本入口同时启用 `use_fabric=True`、
`/physics/fabricUpdateTransformations=true`、`/physics/fabricUpdateJointStates=true`，
并设置 `/physics/fabricUseGPUInterop=false`，PhysX 仍强制 CPU。
仅开启 Fabric 接口不足以启用变换输出。此前 `rollout_basic_VnW2U0`、`rollout_synced_GpRl2a`
虽完成动作传输，但视觉场景未正确更新，不能作为闭环验收结果；最终采用上述 `rollout_visual_N0CiC7`。

2026-09-17 08:27 UTC 经用户授权，在已运行任务的物理 GPU 4 上短时共享执行一次
独立相机检查：[启动日志](diagnostics/camera_gpu4_20260917T082727Z/launch.log)、
[结果摘要](diagnostics/camera_gpu4_20260917T082727Z/result.json)。关闭多卡、反射、
间接漫反射、透明效果和抗锯齿，但保留 RGB 必需的渲染器。Vulkan 唯一活动卡为物理 GPU 4；
启动时报 `libXt.so.6`、`libGLU.so.1` 缺失，约 19 秒后 SIGSEGV（退出码 139），
尚未进入 CPU PhysX 场景，没有生成 RGB。未加载 Cosmos 或运行 rollout，检查进程已退出，
没有停止原任务。随后已在 `packages/ffmpeg/` 补齐 `xorg-libxt`、`libglu` 和 `libopengl`，修复该崩溃。

### 本机相机出图

[probe_isaac_camera.py](probe_isaac_camera.py) 使用 IsaacLab 的官方无窗口相机 experience，
显式固定物理 GPU 4、关闭多卡，PhysX 强制 CPU。渲染仍使用 GPU，不是纯 CPU 软件渲染。
最小场景已保存 [真实 RGB](diagnostics/camera_gpu4_direct_20260917T090749Z/camera.png)，
[结果](diagnostics/camera_gpu4_direct_20260917T090749Z/result.json) 为 `frame_captured`，进程退出码 0。
原生 `ToolsPickingAllHammersTask` 也已完成五路真实相机导出，进程退出码 0，
实际使用 `PARTIAL_RENDERING`、CPU PhysX、物理 GPU 4，重置后执行 8 次物理更新。
[任务结果](diagnostics/robolab_hammers_offscreen_20260917T093238Z/result.json) 为 `frame_captured`。
五张图已实际查看，机器人、工具箱、四把锤子、桌面及背景纹理可见；保留原生相机姿态，
未为展示旋转图像，画面仍有未降噪的噪点。

| 视角 | 真实图像 | 分辨率 |
| --- | --- | --- |
| 头部 | [head_camera.png](diagnostics/robolab_hammers_offscreen_20260917T093238Z/head_camera.png) | 1280 x 720 |
| 左肩 | [over_shoulder_left_camera.png](diagnostics/robolab_hammers_offscreen_20260917T093238Z/over_shoulder_left_camera.png) | 1280 x 720 |
| 右肩 | [over_shoulder_right_camera.png](diagnostics/robolab_hammers_offscreen_20260917T093238Z/over_shoulder_right_camera.png) | 1280 x 720 |
| 手腕 | [wrist_cam.png](diagnostics/robolab_hammers_offscreen_20260917T093238Z/wrist_cam.png) | 1280 x 720 |
| 记录相机 | [egocentric_mirrored_camera.png](diagnostics/robolab_hammers_offscreen_20260917T093238Z/egocentric_mirrored_camera.png) | 864 x 480 |

修复包括：补齐图形动态库；使用 `isaaclab.python.headless.rendering.kit`；
显式设置 `/isaaclab/render/offscreen=true`；直接读取物理渲染后的相机输出。
直接调用 `SimulationApp` 不会自动设置 `AppLauncher` 的离屏标志，漏设会让
`env.sim.render()` 在无窗口模式成为空操作，从而卡在 `reset()` 的纹理等待循环。
已恢复实际渲染，没有关闭 `wait_for_textures`、篡改加载完成标志或跳过任务初始化。
该 experience 禁用了 Replicator
orchestrator，不能额外调用 `rep.orchestrator.step()`，否则会等待不存在的调度帧。
首次 RTX 管线编译耗时数分钟，不能将两分钟超时误判成硬件拒绝。

在本机导出真实四锤任务图像（只占用物理 GPU 4，不启动 Cosmos）：

```bash
cd /home/aiscuser/SteeringWAM
export CUDA_VISIBLE_DEVICES=GPU-9bda66e3-4891-0eb0-79dd-c3530fa33f8a
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH="$PWD/packages/ffmpeg/lib"
export PATH="$PWD/packages/ffmpeg/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p diagnostics
CAPTURE_DIR="$(mktemp -d "$PWD/diagnostics/robolab_hammers_XXXXXX")"
timeout --signal=TERM --kill-after=5s 595s \
  ./RoboLab/.venv/bin/python probe_isaac_camera.py \
  --task ToolsPickingAllHammersTask --output-dir "$CAPTURE_DIR" \
  >"$CAPTURE_DIR/launch.log" 2>&1
printf 'Capture directory: %s\n' "$CAPTURE_DIR"
```

成功时输出各相机 PNG 和 `result.json`。外部超时会留下最后一个已完成阶段，不能把
`app_initialized` 或 `robolab_initialized` 当作出图成功。只有 `frame_captured` 且图像
实际可见才表示本次相机导出完成；不修改原任务 checker，也不声称抓取任务成功。

### 文件恢复与续装

| 项目 | 本机状态 / 来源 |
| --- | --- |
| RoboLab 代码与资产 | `RoboLab/`，官方提交 `ad45d4f`；1643 个 LFS 文件已恢复，无剩余指针 |
| RoboDojo 代码 | `RoboDojo/`，官方提交 `08b7ee4`，递归子模块与本仓库 Cosmos XPolicyLab 适配器已安装 |
| Cosmos Framework | `packages/cosmos3/`，官方提交 `efba207`，不是原实验的定制 checkout |
| RoboDojo 仿真资产 | SAS 恢复至 `RoboDojo/.cache/robodojo_assets_repo/Assets`，`RoboDojo/Assets` 已链接；15367 个文件，零传输失败 |
| RoboDojo 最新 checkpoint | `cookbooks/cosmos3/generator/action/finetune/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge/checkpoints/iter_000002000/`，20 个文件，零传输失败 |
| 基础模型与 VAE | `cookbooks/cosmos3/generator/action/finetune/checkpoints/`，来自 SAS，9 个文件，零传输失败 |
| Edge Policy-DROID | `models/Cosmos3-Edge-Policy-DROID/`，来自官方 `nvidia/Cosmos3-Edge-Policy-DROID`；模型索引对应的四个权重文件及 VAE 已落盘 |
| RoboDojo 数据 | LeRobot v3 EE：602 个文件、119928768368 字节；HDF5：14001 个文件、522982222916 字节。两批均零传输失败，`RoboDojo/data/` 下标准软链接已建立 |
| RoboLab 环境 | `RoboLab/.venv/` 已安装：Python 3.11.16、IsaacLab 2.2.0、Isaac Sim 5.0.0.0、Torch 2.7.0+cu128；CPU PhysX + GPU 4 的原生四锤任务五路相机出图通过 |
| Cosmos 环境 | `packages/cosmos3/.venv/` 已安装：Python 3.13.13、Cosmos Framework 1.2.2、Torch 2.10.0+cu130、官方训练依赖及 policy-server 依赖；另已补装 `msgpack-numpy==0.4.8`，基础导入通过 |
| FFmpeg 与构建工具 | `packages/ffmpeg/` 已安装 FFmpeg 6.1.2、CMake、pkg-config、Git 2.55.0，以及 GLib、libXt、libGLU、libOpenGL 图形运行库 |
| RoboDojo 环境 | Conda 环境 `/home/aiscuser/.conda/envs/RoboDojo` 已安装 Python 3.11.16、Isaac Sim 5.1.0.0、Torch 2.7.0+cu128、固定子模块 CuRobo 和基础依赖；IsaacLab 因版本冲突尚未安装 |

RoboDojo 官方资产路径更新脚本已执行，Franka 和 X5 的两份 CuRobo 配置已指向本机目录。
下载部分已完成；原实验定制代码的来源稍后提供。按后续用户选择，当前已完成基础闭环，不进入十候选分支搜索。

SAS 的新旧两处 checkpoint 备份均只列出 500、1000、1500、2000 步，完整备份的
`latest_checkpoint.txt` 为 `iter_000002000`。因此本机恢复的是 2000 步，不是其它文档中
提及的 60000 步；未下载不需要的历史 checkpoint。训练配置和已有评估结果也已恢复。
RoboDojo 数据和资产保留 2026-09-15 的 SAS 快照，不混入官方之后更新的版本。

下载脚本 [restore_robodojo_assets_from_sas.sh](restore_robodojo_assets_from_sas.sh)
支持 `weights`、`assets`、`data`、`all`。它从仓库外的私密 SAS 文件取凭据，
串行运行 AzCopy，并按已有文件时间戳续传。不要使用 `bash -x` 或分享带凭据的进程参数；
传输日志位于 `/home/aiscuser/.blob-sync/restore-logs/`，不要直接粘贴原始日志。

以下为 RoboLab/Cosmos 的安装配方，当前已完成，不需要再次执行；这些命令不启动服务或仿真：

```bash
cd /home/aiscuser/SteeringWAM
export PATH="$HOME/.local/bin:$PATH"
conda create -y -p "$PWD/packages/ffmpeg" --override-channels -c conda-forge 'ffmpeg>=6,<7'
uv sync --project "$PWD/RoboLab" --python 3.11 --extra isaac50
LD_LIBRARY_PATH= uv sync --project "$PWD/packages/cosmos3" --all-extras --group=cu130-train --group=policy-server
uv pip install --python "$PWD/packages/cosmos3/.venv/bin/python" 'msgpack-numpy==0.4.8'
```

本机缺少系统 GLib，运行时必须使用第 4 节的 `LD_LIBRARY_PATH`，指向
`/home/aiscuser/SteeringWAM/packages/ffmpeg/lib`；留空会导致 OpenCV 导入报
`libgthread-2.0.so.0` 缺失。修正该环境变量后，Cosmos 的 Torch、OpenCV、
Cosmos Framework、msgpack-numpy，以及 RoboLab 的 Torch、OpenCV、h5py 均已通过基础导入。
Cosmos 安装曾被 GitHub DNS 阻塞，最终使用用户目录内的新版 Git 和仅限该安装进程的
`http.curloptResolve` 完成；没有修改系统 DNS 或固定系统 hosts。

RoboDojo 的独立仿真环境按 [适配器前置要求](cookbooks/cosmos3/generator/action/finetune/robodojo_xpolicylab/Cosmos3_RoboDojo/README.md#prerequisites)
准备；本次只调用 Python 依赖安装步骤，跳过系统提权和子模块更新，资产也不重新初始化。
安装脚本固定 `starlette==0.45.3`，但固定 IsaacLab 子模块的
[setup.py](RoboDojo/third_party/IsaacLab/source/isaaclab/setup.py) 要求 `starlette==0.49.1`，
导致依赖解析失败。未通过强制忽略依赖、修改声明或切换子模块版本绕过；需先选择兼容的
IsaacLab/Isaac Sim 依赖组合。CuRobo 已单独安装，未执行运行时编译或仿真。
RoboDojo 环境另按官方 Isaac 约束固定运行时版本，并使用 `huggingface_hub==0.36.2`
避免新版客户端与 `click==8.1.7` 冲突。
这些依赖安装不能补回缺失的原实验 RoboLab/Cosmos 定制代码，也不改变官方对 A100 的支持范围。
已按后续授权加载模型并完成两段基础视觉闭环；未运行训练或测试套件，未验证十候选分支协议。

## 1. 核心思路与探索结论

外部 agent 负责看真实图像、拆解目标、写自指令、审阅与选优；Cosmos3-Edge 只根据图像和自指令生成动作；RoboLab 执行真实仿真分支并记录证据。

1. 在同一个已提交起点生成一个候选，执行前 16 步。
2. agent 读取该候选的实际图像，判断是否有用，并可修改下一候选的自指令。
3. 下一候选仍从本轮原起点恢复，不能接着上一个弃选候选执行。
4. 常规每轮审阅十个候选，从已执行且接受的候选中选一个提交。
5. 下一轮从选中候选末态继续；不是重新生成一次“最佳指令”。

这是一种带在线指令修正的分支搜索，不是固定 prompt 的 best-of-10 对照实验。
同轮指令可能不同，分析时必须保留每个候选自己的 prompt 和审阅理由。

历次探索带来的关键约束：

- 有用的接近、对齐准备可以 `accept` 且 `subgoal_complete=false`；未在 16 步内完成不等于无进展。
- `uncertain` 是外部视觉审阅结论，不是模型自动置信度或数值门槛。
- 夹爪闭合不证明抓住工具；应观察目标随夹爪移动，再验证离箱、桌面放置、松手。
- 若目标被上方工具遮挡，先从图像确认可接近的遮挡物并清理；二维重叠本身不能证明物理压住。
- 画面左/右、机器人坐标轴、世界坐标轴不同。不能把固定像素方向直接解释为世界轴。
- 旧的直接关节控制与第 204 步近似重放接管属于独立实验，不是本分支搜索入口。
- 现有数据证明流程可以执行与留证，不证明已经完成目标锤放置，更不证明四锤全任务成功。

## 2. 代码地图

| 代码 | 职责 |
| --- | --- |
| `RoboLab/scripts/run_cosmos3_toolpickup_agent_loop.py`（待恢复） | CLI、初始化、独立 worker、恢复导入、轨迹导出 |
| `RoboLab/robolab/eval/toolpickup_agent_loop.py`（待恢复） | AgentLoop 状态机、决策校验、分支捕获、原子记录 |
| [episode.py](RoboLab/robolab/eval/episode.py) | 仿真执行、前缀重放、恢复边界与终止；需原实验适配 |
| [scene_state.py](RoboLab/robolab/core/replay/scene_state.py) | 场景状态恢复与误差计算 |
| [client.py](RoboLab/policies/cosmos3/client.py) | Cosmos 通信、动作与生成 trace；需原实验适配 |
| [tools_picking_all_hammers.py](RoboLab/robolab/tasks/benchmark/tools_picking_all_hammers.py) | 原生四锤任务与成功条件 |

配套 Cosmos checkout 的服务入口为 `cosmos_framework.scripts.action_policy_server_robolab`。
需要本地 RoboLab 适配版本、checkpoint 和既有依赖；仅安装任意上游 Cosmos 包不足以保证兼容。

## 3. Agent 必须遵守的协议

- 决策只基于实际图像和指令；禁止读取物体真值坐标、HDF5 几何量替代视觉规划。
- 每候选查看 `actual` 和 `viewport` 的 0、4、8、12、16 帧；短分支按实际长度调整。
- 必须使用能真正看图的工具。读 PNG 字节、hash、生成视频或日志不等于看图。
- 每轮最多十候选。十个全无 `accept` 则停止，禁止追加第十一个或偷偷重置额度。
- 除原生全任务成功外，不能在十个候选审阅完前提前选优。
- 主轨迹上限 1500 步，通常每次提交 16 步，最后剩 12 步时执行 12 步。
- `chunk` 从 0 开始；“跑到 chunk=8”是完成轮次 0..7，不能再 reserve 第 8 轮。
- 同一 session 只能有一个控制者。每次转换前读取最新 `request_id`，禁止复用旧 token。
- 使用 CLI/API 写状态，禁止直接改 `session.json`、hash、历史评价或已执行动作。
- 不修改原任务 checker：单锤子目标完成与四锤全任务成功必须分别报告。

动作协议：生成数组 `(32,8)`；前七维是 Panda 七关节绝对目标角，单位弧度，不是末端位移或关节增量。
第八维为夹爪，客户端以 `>0.5` 判为闭合，0 张开、1 闭合。执行数组通常 `(16,8)`。
当前捕获/校验路径要求保存的生成动作前缀与执行动作逐项一致；不要自行再做坐标转换或缩放。
控制频率由环境配置确定，当前实验为 15 Hz。

## 4. 启动前检查

以下命令适用于本机布局；换机器时修改三个绝对路径。不要凭活动 shell 的 Python 判断解释器。

```bash
export ROBOLAB=/home/aiscuser/SteeringWAM/RoboLab
export COSMOS=/home/aiscuser/SteeringWAM/packages/cosmos3
export CHECKPOINT=/home/aiscuser/SteeringWAM/models/Cosmos3-Edge-Policy-DROID
export RUN="$ROBOLAB/output/my_hammer_search"
export PORT=18745
export MASTER_PORT=29745
export MODEL_GPU=5
export SIM_GPU=4
export PATH="/home/aiscuser/SteeringWAM/packages/ffmpeg/bin:$PATH"
export LD_LIBRARY_PATH="/home/aiscuser/SteeringWAM/packages/ffmpeg/lib"
"$ROBOLAB/.venv/bin/python" --version
"$COSMOS/.venv/bin/python" --version
nvidia-smi
ss -ltnp
```

确认目录、权重可用、GPU 显存足够、端口空闲且没有其他控制者。以上 GPU/端口只是示例，不是资源预留。
RoboLab 当前使用其 Python 3.11 环境；Cosmos 使用自己的 venv，其版本可以不同。不要混用依赖环境。
不要读取或打印 token 文件；不要停止其他任务。先检查对应 CLI `--help`，不在生产采样时猜参数。

## 5. 启动 Cosmos3-Edge 服务

本机基础闭环使用 [launch_cosmos3_robolab_server.sh](launch_cosmos3_robolab_server.sh)。
它使用 `--checkpoint-path`、JSON prompt、4 步 UniPC 和 guidance interval `(960,1001)`，
关闭辅助 Guardrail、torch.compile 和预测视频解码，固定单 rank。
旧定制服务的 `--no-reasoner-recurrence`、`--no-disable-generator-language-condition`、
`--prediction-output-dir` 等参数在当前官方服务中不存在，不能继续照抄旧命令。

```bash
mkdir -p "$RUN/model"
MODEL_GPU="$MODEL_GPU" PORT="$PORT" MASTER_PORT="$MASTER_PORT" \
  bash /home/aiscuser/SteeringWAM/launch_cosmos3_robolab_server.sh "$RUN/model"
```

确认服务完成加载并监听后再采样。Reasoner 必须关闭，Generator 必须保留语言条件。
保存完整启动命令、checkpoint 身份、实际加载源码 hash、版本、GPU、端口、PID 和日志。
可将这些信息写成 server manifest，在 `init` 时加 `--server-manifest /absolute/path/to/manifest.json`。
注意：当前实现记录 manifest 的引用，但 `server_provenance_verified` 仍为 false；提供文件不等于已自动认证模型来源。

## 6. 初始化与单轮命令

在另一个终端设置同样的路径变量，然后：

```bash
cd "$ROBOLAB"
export SESSION="$RUN/session"
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py init \
  --session "$SESSION" --sim-gpu "$SIM_GPU" --host 127.0.0.1 \
  --port "$PORT" --seed 3407 --scene-seed 0 --worker-timeout 1200
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py status --session "$SESSION"
```

`SESSION` 必须不存在。`init` 会启动仿真获取零动作根检查点，不会自动采样。
失败后不要重复对同一目录运行 init；先检查日志和产物。
`status` 输出并更新 `request.json`；`session.json` 为带校验的权威状态。
根图像在 `root/capture`，也可从 `checkpoint.image.path` 定位。

先实际看图，再由 agent 创建 `instruction.json`。示例字段如下，token 必须替换为最新值：

```json
{
  "request_id": "REPLACE_WITH_CURRENT_REQUEST_ID",
  "instruction": {
    "target": "wood_hammer",
    "subgoal": "align_wood_handle",
    "instruction": "Move the open gripper above the exposed wooden handle and align the fingers around it without closing.",
    "completion_condition": "Both open fingers visibly straddle the wooden handle, clear of adjacent tools.",
    "avoid": "Do not close on another handle or push the bin."
  }
}
```

允许 target：`husky_hammer`、`blue_hammer`、`red_hammer`、`wood_hammer`。
对象视觉身份必须看图确认。当前 API 只接受锤子 target；清理非锤工具不能虚构一个新 target 值，需先明确受支持的任务扩展。

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py instruct \
  --session "$SESSION" --decision "$RUN/instruction.json"
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py status --session "$SESSION"
```

取当前 `request_id`，仅执行一次：

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py run-next \
  --session "$SESSION" --request-id CURRENT_REQUEST_ID
```

等待 worker 结束并确认 `pending_review`。查看 `candidates/<id>/capture/actual_*.png` 与 `viewport_*.png`。
不要用模型预测图片代替这些实际仿真帧。

由 agent 创建独立的 review JSON：

```json
{
  "request_id": "REPLACE_WITH_CURRENT_REQUEST_ID",
  "candidate_id": "chunk_000_candidate_00",
  "verdict": "accept",
  "subgoal_complete": false,
  "evidence": "REPLACE_WITH_OBSERVED_MOTION_AND_TARGET_EVIDENCE",
  "reviewed_frame_sha256": ["ACTUAL_FRAME_0_HASH", "ACTUAL_FRAME_4_HASH", "ACTUAL_FRAME_8_HASH", "ACTUAL_FRAME_12_HASH", "ACTUAL_FRAME_16_HASH"]
}
```

verdict 为 `accept/reject/uncertain`。hash 从该候选 `result.frames` 取得，必须对应实际查看过的帧。
代码最低校验首/中/末 hash；本流程要求五时刻双视角，不能把最低 schema 校验当作完整视觉审阅。
`evidence` 写具体运动、目标关系、是否接触/随动和不确定性，不要复制示例或只写“通过”。
可额外提供 `revised_instruction`，结构与 instruction 相同，target/subgoal 必须保持本轮一致。
修正必须适用于原起点，而不是候选末态。例如弃选分支抓到了物体，也不能让下个原起点候选直接松手。

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py review \
  --session "$SESSION" --decision "$RUN/review.json"
```

若返回 ready，读取新 token 后采下一个。若 need_selection，看完本轮所有证据后创建：

```json
{
  "request_id": "REPLACE_WITH_CURRENT_REQUEST_ID",
  "candidate_id": "REPLACE_WITH_ACCEPTED_CANDIDATE_ID",
  "reason": "REPLACE_WITH_VISUAL_COMPARISON_AND_SELECTION_REASON"
}
```

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py select \
  --session "$SESSION" --decision "$RUN/select.json"
```

选中后 `step` 增加执行步数，`chunk` 增加 1。若 need_instruction，继续下轮。
若已选候选 `subgoal_complete=false`，下一轮 instruct 不能改变 target/subgoal，但可改同子目标指令措辞。

## 7. 状态分派与停止

| 当前 status | 唯一合适的后续动作 |
| --- | --- |
| need_instruction | 看当前图像，提交 instruct；先检查 chunk 停止界限 |
| ready | 检查界限与新 token，run-next 一次 |
| execution_unknown | 检查进程、退出记录、完整 result；禁止重复 run-next |
| pending_review | 只审阅已执行候选，然后 review |
| need_selection | 在本轮已 accept 候选中 select |
| no_acceptable_candidate | 协议停止，不补采 |
| success / budget_exhausted / cancelled | 停止采样，导出和报告 |

`chunk>=8` 是外部控制者的停止条件，CLI 没有 `--max-chunks`。到达后保留状态、记录停止原因并导出，不手改 status。
`cost.attempts` 包含预留/未知尝试；`candidate_steps`、`replay_steps` 与已提交主轨迹步数分开统计。
种子路由为 `seed + chunk*10 + attempt`，`replan_index=chunk`；native trace 校验派生种子和真实自指令。

## 8. 恢复、换目标与用户指定 fork

恢复条件：检查 `process.json`、`exit.json`、`worker.log`、`capture/result.json`。
只有 worker 确认 exit code 0、status complete 且 result 完整时，才能：

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py recover --session "$SESSION"
```

recover 仅导入结果，不重新执行。无法证明执行结果时记录阻塞；不能自动重试未知动作。
pending_review 已经有结果，直接看图 review，不需要 recover。

恢复使用动作前缀重放与场景状态校正，不是完整 PhysX 内存快照。
每次验证状态误差 <=1e-5、像素 MAE <=1/255 标度中的 1，并校验动作、事件和祖先引用。
这些门槛不证明隐藏接触求解器状态完全相同。不能放宽门槛或覆盖源码归档以继续。

初始化归档六个运行源码，并在 worker 前验证 live/archive hash。源码改变会阻塞旧 session；先做显式版本迁移方案，不绕过锁。

retarget 仅在 `need_instruction/ready` 且本轮没有候选时可用，必须换不同 target，并保存用户授权。
decision 为 `request_id`、`instruction`、非空 `user_request`、`reason`，通过 CLI `retarget --decision ...` 提交。

用户明确指定从旧候选继续时，可建立独立 fork，不能作为旧轮第十一次尝试：

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py fork \
  --session "$RUN/new_fork" --source-session /absolute/path/to/source_session \
  --candidate-id chunk_002_candidate_06 --user-request "Exact explicit user authorization" \
  --sim-gpu "$SIM_GPU" --host 127.0.0.1 --port "$PORT"
```

当前 fork 只从 source 当前 `candidates` 中查找已 reviewed 候选，不搜索 history。
保留原 review；fork 是用户定向续跑，不代表原 uncertain 被改为 accept。
继承步数计入 1500 总预算，新成本另计。fork 导出仅含新增已选段，不假装包含继承的全部轨迹。

## 9. 持久运行与多实验隔离

- 每个实验使用独立 session、输出、模型端口、master port、日志和控制对话。GPU 需事先确认容量。
- tmux 存活不等于 agent 正在采样；核对进程、manifest 转换和新图像。
- Copilot 同名会话可能冲突，恢复用唯一 session UUID。非交互调用必须授权实际 venv 解释器目录，包括符号链接目标；不需要开启全路径权限。
- 旧 CLI 出现过 `600s waiting for background tasks` 退出；检查并按实际任务设置 `COPILOT_TASK_WAIT_TIMEOUT_SECONDS`，但仍需确认持续执行。
- 不要把控制权递归委派而失去进程归属。恢复时先确认无旧控制者，尤其注意候选已执行但未审阅的情况。
- 结束 tmux 主进程可能连带结束同进程组的模型/agent；不要认为只 kill 调度器就一定保留孩子进程。
- [历史调度器](../../output/hammer_chunk8_20260916/coordinate.py) 硬编码会话 UUID、路径和目标，只作历史参考，不能原样启动新实验。
- 当前本文任务只是文档整理，不授予启动新仿真、删除旧输出或终止已有实验的额外权限。

## 10. 数据、网页与量化

每个 session 保存 config/source、根检查点、session/request、候选、历史与提交链。
每候选包括 32x8 生成动作、执行前缀、HDF5、actual/viewport 图、NPZ、generation trace、恢复误差及 hash。
审阅文本保存在状态记录中；文件命名由外部 agent 决定，不能仅依赖某个固定 review 文件名。

```bash
.venv/bin/python scripts/run_cosmos3_toolpickup_agent_loop.py export --session "$SESSION"
```

输出 `trajectory_step_NNNN`；目录已存在会失败，先检查已有导出，不覆盖。视频是状态校正后的分支拼接，不能称无中断现场轨迹。

现有离线网页在 [site/index.html](../../output/wood_restart_20260914/site/index.html)。
快照与打包入口：

```bash
cd "$ROBOLAB"
.venv/bin/python output/wood_restart_20260914/extend_metrics.py
.venv/bin/python output/wood_restart_20260914/package_site.py
```

这两个是历史四组实验专用工具，不自动发现新 session；新实验须显式扩充 `extend_metrics.py` 的 SESSIONS 映射。
网页是快照，运行中新增数据不会自动出现。原始 `render_review.py` 有旧三轮假设，不把它当通用导出器。
打包保留页面外下载依赖及目录结构，解压后打开入口 HTML。

量化是事后分析，不反馈在线视觉 agent：

- HDF5 `ee_pose/position` 是机器人根坐标；用 `robot_root_pose` 的位置和 wxyz 四元数转换到 env-local。
- `bbox/centroid/<target>` 是 env-local 包围盒中心，float16 米。
- 当前 recorder 连杆名为 `base_link`，不是已验证的指尖中心；距离不是指尖到手柄表面的间隙。
- 候选指标切片 `[start_step:start_step+length]` 对应动作后第 1..16 步；不含第 0 步末端采样，净位移覆盖 15 个间隔。
- 可报告起末/最小中心距离、接近量、XYZ 净位移、路径长度、相对目标位移、目标中心 ΔZ。
- “向左”需明确相机或轴；当前网页只允许显式选择 env-local 正负轴，未自动验证画面左移。
- 数值变近、目标微小漂移、夹爪闭合均不能单独证明子目标成功。

## 11. 交接验收清单

其他 agent 开始前：读本文、检查当前源码/CLI、读取权威 session、确认唯一控制权和可用图像工具。
每次提交前：新 token、真实图像证据、合法状态、同起点与候选额度、未越主轨迹或 chunk 界限。
结束时：记录各分支 step/chunk/status、已执行/已审阅/已提交数量、停止原因、单锤与原生成功证据。
验证导出视频可解码、动作与 HDF5 一致、网页链接完整；只停止自己拥有的服务。
交接记录必须写明待审候选 ID、是否存在未知执行、进程与端口、下一条合法动作；不把旧 PROGRESS 文本当实时状态。

CPU 状态机测试应选 `tests` 下实际相关文件，使用 `.venv/bin/python -m pytest --noconftest ...`，避免默认 conftest 启动 Isaac。
在没有实际运行测试或成功证据时，明确写“未验证”，不要用命令启动成功替代任务完成。