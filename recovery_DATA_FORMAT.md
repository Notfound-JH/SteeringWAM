# Recovery 轨迹数据对接说明

本文描述**已经实际落盘的数据**，不是拟议格式。数据来自真实 Isaac Sim
步进，不是物理机器人，也不是 Mock。原始格式标识为
`wam-agent-recovery-v1`。已采集历史轨迹的工具协议为 `wam-recovery-v1`；
V2 实验使用 `wam-recovery-v2`，其模式理由、条件变更和评估窗口见第 10 节。
显式开启任务进程策略的新实验使用 `wam-recovery-v3`，见下方说明。
容器格式不变，不代表各协议的评估语义相同。

### 四锤完整任务的逐物体进程

新采集的 `ToolsPickingAllHammersTask` 在每个实际执行响应的 `events`、
chunk 结果及最终 manifest 中增加 `hammer_task_progress`：
`per_object`、`completed_count`、`required_count=4`、`remaining_objects`。
每把锤子复用原生 success 函数及其原有参数，只把对象列表缩为该单个对象；
四者全部通过必须与原生任务成功一致，否则明确报错。
该记录不修改成功条件，不会在一把完成后结束完整任务。
这是当前时刻的判据快照，不是累计锁存：已放置的锤子再次不满足判据时会退回未完成。
历史 episode 没有这些字段时不能补造逐物体成功结果。

### V3：任务进程与独立阶段判断

- `task_progress_review` 在普通 WAM chunk 的实际执行端点记录任务对象、
  预期/观察阶段、目标与阶段是否对齐、进程变化及证据。未判断不能执行下一
  chunk；原生终止后不再要求下一轮进程检查。
- `recovery_proposal` 增加 `agent_role=correction_agent`、`agent_id`、
  `task_progress` 和 `condition_reason`。目标/阶段错误或不确定时，先做
  Condition；Action 与当前 Action-first Joint 仅用于目标/阶段已对齐的情况。
- `recovery_evaluation_started/completed` 使用 `stage=action_effect` 或
  `condition_progress`。Action 阶段**不含 WAM 动作**，立即交给独立判断；
  Condition 阶段包含下一轮 16 步 WAM，原生终止允许更短的实际前缀。
- Joint 明确使用 Action → 判断 → Condition/WAM → 判断；Action 判断未通过，
  本次尝试结束，不执行其 Condition 阶段。一次 Joint 仍只消耗一次预算。
- 每个阶段的 `result.json`、`evaluation.json`、`assessment.json` 保存在
  `recovery_attempts/NN/<stage>/`；Cartesian 编译结果仍在尝试根目录，
  Action 阶段目录也保存一份相同的编译结果供原控制器引用；实际 IK 命令位于
  `action_effect/executed_joint_actions.npy`。
- 判断包含 `agent_role=recovery_judge`、与提案不同的实际 `agent_id`、
  `correction_effect`、`task_progress`、`recovery_success`、
  `continuation_reason`、`evidence` 和准确的评估端点。
  角色/ID 校验本身不证明实际调用了独立 Agent，实验需另存调用输入与输出。
- V3 的 `local_recovery_success` 仅是 `correction_effect == "achieved"` 的
  兼容字段；**是否允许 WAM 继续看 `recovery_success`**。Manifest 的
  `task_progress_recovered` 表示当前恢复进程状态，后续 WAM 再失败会清除。
  原生任务成功仍单独记录，不能由 Judge 文本推断。
- `progress_contract_verified` 校验上述边界；`comparison_contract_verified=null`
  表示不适用 V2 同长窗口比较，而非校验失败。V3 不与 V2 的局部成功率混算。
- V3 可在 Condition 后的 WAM chunk 已达到原生目标、且判断通过时结束整链，
  不强制原生终止后再执行独立 continuation chunk。纯控制器完成或失败预算
  耗尽后的 WAM 自行成功不算 `acceptance_success`。

## 1. 推荐的接入入口

每个实验目录就是一个 episode，目录名作为 `episode_id`：

```text
diagnostics/<episode_id>/
├── experiment.json                 # 任务、初始化、Agent 输入来源、实验模式
├── recovery/
│   ├── session.json                # 当前/最终状态；运行中会更新
│   ├── manifest.json               # finalize 后产生的总索引与结局
│   ├── verification.json           # 独立校验结果
│   ├── events.jsonl                # 完整有序事件；主要接入入口
│   ├── trajectory.hdf5             # 实际动作、实际状态、EE 位姿
│   ├── episode_wrist.mp4           # 全过程腕部视角，非多机位合成视频
│   ├── observations/<step>_<label>/
│   │   ├── wrist_cam.png
│   │   ├── head_camera.png
│   │   ├── over_shoulder_left_camera.png
│   │   ├── over_shoulder_right_camera.png
│   │   ├── egocentric_mirrored_camera.png  # 调试视角，勿混为策略输入
│   │   ├── robot.json
│   │   └── scene.npz               # 仿真状态，包含 privileged object poses
│   ├── segments/000_wam/           # 仅实际调用 WAM 时存在
│   │   ├── request.json
│   │   ├── request_state.npz
│   │   ├── input.png
│   │   ├── raw_actions.npy
│   │   ├── predicted_actions.npy
│   │   ├── executed_actions.npy
│   │   ├── predicted_future.npy
│   │   ├── predicted_future.mp4
│   │   ├── predicted_future_frames/
│   │   └── result.json
│   ├── recovery_attempts/01/       # WAM Recovery 的纠错尝试
│   │   ├── proposal.json
│   │   ├── control_code.txt
│   │   ├── compiled_cartesian_targets.npy
│   │   └── executed_joint_actions.npy
│   └── control_programs/001/       # 独立 Agent-only 实验；不是 Recovery attempt
│       ├── proposal.json
│       ├── control_code.txt
│       ├── compiled_cartesian_targets.npy
│       ├── executed_joint_actions.npy
│       └── result.json
└── ...                            # 客户端响应、运行日志等辅助文件
```

Condition-only 纠错没有 Cartesian target / IK 文件。Joint 纠错的局部动作在
`recovery_attempts/`，修改条件后调用的 WAM 预测在 `segments/`，通过事件连接。
Recovery 每次尝试的完成结果在 `recovery_attempt_completed` 事件内，
不要假设它与 Agent-only 一样有 `result.json`。

**建议：JSONL 是时序与语义主表，HDF5 是数值轨迹，PNG 是实际视觉，
NPY 是预测/控制意图，manifest + verification 是结局依据。**

## 2. 四种“成功”必须分开

| 字段/检查 | 表示什么 | 不表示什么 |
|---|---|---|
| `recording_verified` | 记录、连续步号、动作和相关文件一致 | 任务一定成功 |
| `local_recovery_success` | Agent 对本次局部恢复目标的评估 | 已抓稳或完整任务成功 |
| `native_success` | 当前实验配置的终止成功条件成立 | 原生条件一定足够严格 |
| `acceptance_success` | 已验证 WAM failure → correction → handoff → WAM continuation → success | Agent-only 成功 |
| `agent_control_success` | 独立 Agent-only 实验到达其成功条件 | WAM Recovery 成功 |
| `stable_blue_placement_verified` | 严格单蓝锤放置检查经复核通过 | 可用于任意任务的通用成功指标 |

读取 `experiment_mode` 和 `success_criterion` 后，再解释布尔结局：

- `experiment_mode="wam_recovery"`：WAM 主导的闭环恢复。
- `experiment_mode="agent_control"`：Agent 全程直接调用受限控制接口，无 WAM。
- 早期记录缺少 `experiment_mode`；当前读取器按 legacy WAM Recovery 处理，
  接入时应保留这是历史兼容推断。
- `success_criterion="native"`：原任务条件。
- `success_criterion="native_and_stable_blue_placement"`：**另行标记的严格实验**。
  原始单锤条件仍必须成立，但还要求整个锤子包围盒与箱体水平分离至少
  5 mm、发出张开命令且实测闭合度 < 0.1、夹爪基座距锤子包围盒至少
  20 cm、桌面向上支撑力，以及连续 15 步线速度 ≤ 0.01 m/s、
  角速度 ≤ 0.05 rad/s。原始条件值另存为 `original_native_success`。

缺失标签应当记作“未知/未验证”，不能自动变成负例。尤其：
Agent-only 的 `acceptance_success=false` 是“不是此类闭环”，不能据此给
Agent-only 动作打失败标签。

已确认的对接样本：

| Episode | 实际步数 | 语义 |
|---|---:|---|
| `recovery_cube_20261011` | 212 | 完整 WAM Recovery 验收成功 |
| `agent_tools_coding_20261012T1600` | 1576 | 四锤 Agent-only 失败，43 个控制程序保留 |
| `agent_blue_hammer_20261012` | 712 | 单蓝锤原生条件成功；**未证明稳定放置** |
| `agent_blue_hammer_strict_20261012` | 900 | 到达原始时限；已松爪但严格放置未通过，不标为成功 |

六任务批次以及两次额外 tools Recovery 的失败数据也保留。
完整目录清单由本次 handoff 的 `episodes.json` 提供，运行中行只是快照。
更新后的清单包含 12 条已结束轨迹，共 9,868 个真实仿真执行步。
严格实验的测量与判据差异另见 handoff 中的 `strict_placement_report.json`：
833–835 步低速且有桌面支撑力；836 步起速度及报告的支撑力均为零，
原始成功条件变为 false。物理休眠可能影响接触力报告，但尚未独立确认；
包围盒分离也未通过，不能仅凭最终图像覆盖这些结局标签。

## 3. 对齐规则：最容易接错的部分

对于第 `k` 次真实控制步，`k` 从 1 开始：

```text
state[k-1] -- action[k-1] --> state[k]
```

| 存储位置 | 第 k 次执行对应的索引 |
|---|---|
| `action_authorized.episode_step` | `k-1`，动作执行前 |
| `actual_execution_response.episode_step` | `k`，动作执行后 |
| HDF5 `data/demo_0/actions[k-1]` | 第 k 次提交到环境的动作 |
| HDF5 `states/... [k-1]` | 第 k 次执行后的状态 |
| HDF5 `ee_pose/... [k-1]` | 第 k 次执行后的 EE 状态 |
| 实际观测 `observation.step` | `k` |
| 腕部视频帧索引 | `k`；帧 0 是初始观测 |

已在真实 cube 轨迹上逐项比较：HDF5 的 post-step EE position 和 arm joint
position 与 JSONL 对应状态的最大误差都是 **0**，不是仅凭名称推测。

- 初始状态从 `session_started` 的 `state` / `observation` 获取；
  不要假设 HDF5 一定还有 `initial_state` 通道。
- 本批 `control_dt=1/15` 秒。模拟时间是 `k * control_dt`。
  `recorded_at` 是 Unix wall-clock 秒，包含推理、工具往返、人工观察和等待，
  不能把两个事件的 wall-clock 差当作物理积分步长。
- `event_index` 才是所有事件的严格顺序。多个事件可以有同一个 step。
  `control_handed_back_to_wam` 还可能先于同一步的 `recovery_assessment` 写入。
- 同一物理 step 可能有逐动作观测和程序边界观测两份目录，不要按目录数统计步数。
- `subtask/*` 是另一种记录频率。cube 的这些通道只有 6 条，而动作有 212 条，
  **不能**直接与逐步动作 zip。
- 终止发生在程序中途时，实际执行是计划的一个前缀。使用 `executed_steps`
  和实际事件数，不要用计划长度冒充实际执行长度。

## 4. 动作、位姿和单位

| 数据 | 形状 | 含义 |
|---|---|---|
| WAM `raw_actions.npy` | `(32,8)` float32（当前样本） | 模型原始输出 |
| WAM `predicted_actions.npy` | `(32,8)` float32 | 后处理预测；夹爪按 `>0.5` 二值化 |
| WAM `executed_actions.npy` | `(L,8)` float32 | 真正执行的预测前缀，当前每段最多 16 步 |
| `compiled_cartesian_targets.npy` | `(M,8)` float32 | `[x,y,z,qw,qx,qy,qz,g]`，不是关节动作 |
| `executed_joint_actions.npy` | `(L,8)` float32 | IK 后真正提交的 7 个关节目标和夹爪命令 |
| JSONL `actual_action` / HDF5 `actions` | `(8,)` / `(T,8)` | 同上；**是环境接收的命令，不是实测关节角** |
| `measured_arm_joint_pos` | `(7,)` | 实测 Panda 关节角，rad |
| `measured_arm_joint_vel` | `(7,)` | 实测关节速度，rad/s |
| `ee_pos`, `ee_quat` | `(3,)`, `(4,)` | Robot-root frame；m；四元数 **wxyz** |
| `gripper_pos` | `(1,)` | 实测归一化闭合度，裁剪至 `[0,1]`，0 开、1 闭 |

关节动作顺序为 `panda_joint1 ... panda_joint7, gripper_command`。
这里的 EE 是 Robotiq **`base_link` 基座**，不是指尖中心或物体接触点。
夹爪命令为 1，不代表实测已闭合，也不代表抓到了物体。
严格实验的 `placement_verification.gripper_closure` 则保留
`finger_joint / (π/4)` 的未裁剪值，可能略小于 0 或大于 1。
校验时应先与 `measured_joint_pos` 中对应关节核对，再裁剪后对齐
`gripper_pos`，不能直接要求两个字段完全相等。

完整机器人状态 HDF5 有 **13 个关节**（7 机械臂 + 夹爪联动关节），
不要误当作 13 维动作。Agent-only 的 `robot.json` 额外保存
`joint_names`、`body_names`、`body_pos_w`、`body_quat_w` 等调试信息；
旧 Recovery 记录不一定有这些字段。

其他坐标细节：

- `scene.npz` / HDF5 `states/.../root_pose` 是 `[x,y,z,qw,qx,qy,qz]`，
  位姿相对 env origin；`root_velocity` 是 `[vx,vy,vz,wx,wy,wz]`。
- HDF5 `ee_pose` 的 position/orientation 在 robot-root frame，
  linear/angular velocity 仍用 world axes。
- `bbox/bbox_mm/<object>` 是 `(T,8,3)` **int16 毫米**，需除以 1000；
  这是 OBB 的 8 个角，不是 `(min,max)` 两点。
- `bbox/centroid/<object>` 是 `(T,3)` float16 **米**。
- 不同任务的物体键不同；使用实际键名，不硬编码只有 cube 或四个锤子。

## 5. 视觉与 WAM Predicted Future

当前 cube 样本实测：

| 文件 | 实测尺寸/形状 |
|---|---|
| 四个实际相机 PNG | 宽 1280 × 高 720，RGB |
| `egocentric_mirrored_camera.png` | 宽 864 × 高 480，调试视角 |
| WAM `input.png` | 宽 640 × 高 540 |
| `predicted_future.npy` | `(33,528,640,3)` uint8 |
| 腕部实际视频 | 213 帧 = 初始帧 + 212 个真实动作后帧，15 fps |

尺寸不是协议常量，读取文件 shape 后再处理。

WAM 输入拼图布局：上方是 360×640 wrist，下方是左右各 180×320
的 left/right shoulder；head camera 并未进入该 WAM 请求。
模型预测图像高度与输入拼图高度也不完全相同，不要直接做像素级差分。

`predicted_future.npy` 保留模型完整视频数组；当前保存逻辑把第 0 帧视作
条件帧，`predicted_future.mp4` / `predicted_future_frames/` 只保存其后帧。
预测未来是 **model prediction**，绝不能混为真实后续观测。
未执行的 chunk 后半段也没有相应的真实执行标签。

实际观测优先使用无损 PNG；MP4 主要用于回看。当前这套 handoff
不提供统一的逐帧相机内外参标定包，也没有真实硬件时间戳。

## 6. 事件表与修正尝试

每行 JSONL 都有：
`event_type`, `event_index`, `episode_step`, `recorded_at`。

主要事件：

| 事件 | 用途 |
|---|---|
| `session_started` | 初始视觉/机器人状态 |
| `wam_prediction` | prompt、subgoal、预测动作/视频引用、预测长度、推理时长 |
| `action_authorized` | 执行前命令；Cartesian 时还有未限幅 IK 输出 |
| `actual_execution_response` | 每个真实步的动作、实测状态、观测与终止标志 |
| `failure_detected` | failure_type、evidence、失败状态、当前 subgoal |
| `recovery_proposal` | diagnosis、recovery_goal、mode、proposal 文件引用 |
| `condition_changed` | V2 条件的 before/after、用途、理由、source、attempt |
| `recovery_evaluation_started/completed` | V2 固定 WAM 评估窗口的计划、真实长度和中断原因 |
| `correction_program_compiled` | 限制过的逐步 Cartesian 意图 |
| `recovery_attempt_completed` | 本次尝试全部实际结果与观测 |
| `recovery_assessment` | 局部恢复布尔评估与文字证据 |
| `control_handed_back_to_wam` | 真正交还 WAM 的边界 |
| `control_returned_to_wam_after_budget` | 预算耗尽且最后一次评估失败后返回 WAM；不是恢复成功 |
| `agent_control_proposal/completed` | Agent-only 控制程序，使用 `program_index` |
| `command_failed` | 接口或执行错误，不丢弃 |
| `session_finalized` | episode 封存结局 |

`action_source` 取值及含义：

- `wam`：正常 WAM。
- `agent_correction`：Recovery 的 Cartesian / IK 实际动作。
- `agent_condition`：纠错期间，**WAM** 在修改条件下生成并执行的动作。
- `recovery_evaluation`：V2 三种模式统一的 WAM 评估窗口；是否改条件需读取
  `condition_change`，不能仅凭这个 source 判断。它不是 Agent 的 Cartesian 动作。
- `wam_continuation`：局部恢复评估成功后交还控制权的 WAM。
- `wam_after_budget`：纠错预算耗尽、评估仍失败后的 WAM；不能作为成功恢复的监督标签。
- `agent_control`：独立 Agent-only 基线，不能冒充 Recovery。

可选运行策略在新记录中保存为 `budget_exhaustion_policy`（`stop` 或
`continue_wam`）、`max_episode_steps`、`budget_fallback` 和
`wam_after_budget_steps`。尝试预算仍由 `safety_limits.max_attempts` 给出，
按整个 episode 计数，不会在 WAM 重新接管时重置。预算后续跑即使最终完成
原生任务，也不会把此前失败的局部评估改成成功；`acceptance_success`
仍要求成功恢复后的 `wam_continuation` 链条。旧记录缺少这些运行策略字段时，
校验器返回 `runtime_policy_verified=null`，不追溯修改旧标签。

WAM 真正收到的 prompt 是：

```text
Overall task: <native instruction>
Current subgoal: <condition>
```

`request.json.subgoal` 是另外的元数据字段，不一定等于实际送入模型的
`condition`。不要只抽 `subgoal` 而漏掉 `prompt` / `condition`。

Action / Joint proposal 保存两种表示：原始 `control_code` 和解析后的
`program`。代码只是受限 DSL 文本，不是可任意 `exec` 的 Python。
对接训练时按数据读取；如果重新执行，必须使用新 request token，
并在提交时去掉 `control_code` / `program` 中的一种，不能直接照搬保存的
规范化 proposal。

## 7. 真实的 Intent → Execution → Response 样例

来自 `recovery_cube_20261011`：

```text
失败：step 96；夹爪在方块上方空闭合，方块未离桌。
目标：恢复张开的预抓取姿态，把抓取和放置留给 WAM。
程序：open 4 steps → move 12 steps → hold 4 steps。
执行：step 97..116；source=agent_correction。
最后目标 EE：[0.475, -0.065, 0.195] m。
实测 EE：   [0.473232746, -0.064890556, 0.197156370] m。
位置误差：  2.79 mm。
后续：step 117..212 为 WAM continuation，完整闭环成功。
```

这条样本的 `actual_action == ik_joint_command` 只说明环境接收了指定关节
目标；2.79 mm 的差值才是 Cartesian 意图与真实运动响应的一部分。
碰撞、滑脱、抓取保持与否，还需要结合图像、物体状态和后续连续帧。

本次 handoff 附带：

- `correction_example.json`：这次纠错的完整 proposal、failure、assessment
  以及 step 116 的真实响应。
- `transitions.sample.jsonl`：cube 的 96/97/116/117 步和单蓝锤的 712 步，
  展示 pre-state → intent/action → post-state；路径已经转换为 episode-relative。
- `episodes.json`：当前 episode 清单、模式、标签、记录是否 finalized。

这些是**派生对接样例**，不是另一份原始训练集，也没有复制大批图像。

## 8. 下载、路径重映射与完整性

Azure blob key：

```text
jinghe/steeringwam/recovery/<episode_id>/<episode-relative-path>
```

例如：

```text
jinghe/steeringwam/recovery/recovery_cube_20261011/recovery/events.jsonl
```

访问需要现有私有存储授权；不要把 SAS/token 写进数据集、代码或文档。
原始 JSON 里的 artifact 引用为：

```json
{"path": "/original/machine/diagnostics/EPISODE/recovery/FILE", "sha256": "..."}
```

**不要改写原始 JSON 来替换绝对路径**，否则事件/manifest 的原始哈希会失效。
下载后在 reader 中重映射：

```python
import hashlib
from pathlib import Path, PurePosixPath

def local_artifact(reference, manifest, downloaded_episode):
    original_root = PurePosixPath(manifest["events"]["path"]).parent.parent
    relative = PurePosixPath(reference["path"]).relative_to(original_root)
    root = Path(downloaded_episode).resolve()
    target = (root / relative.as_posix()).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Artifact escapes the downloaded episode")
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != reference["sha256"]:
        raise ValueError(f"Checksum mismatch: {relative}")
    return target
```

当前 [verify_recovery_episode.py](verify_recovery_episode.py) 的路径读取仍以
原始路径为准；跨机器读取请使用上述重映射，而不是期待原样运行就能找到文件。
原始存储机上的检查命令：

```bash
./RoboLab/.venv/bin/python verify_recovery_episode.py diagnostics/EPISODE/recovery
```

运行中的 session 会继续追加日志、更新状态和上传。不要训练或验收正在写的
HDF5，不要把某一轮 `synced` 当作最终数据已齐。推荐只接收
`manifest.json` + `verification.json` 已产生、相关引用哈希匹配的 finalized
episode。失败 episode 也同样接收。

## 9. 对接 Reasoner 的推荐样本边界

建议第一版先用 **attempt-level**，而不是把所有动作拍平成无语义序列：

```text
输入：
  failure 前观测/状态 + 当前 condition + 最近实际轨迹
  + WAM predicted future（明确标注为预测）
  + Agent diagnosis / recovery_goal / correction_mode / proposal
目标：
  每步实测响应 + after observation + 局部评估
  + 是否交还 WAM + WAM continuation + episode outcome
```

以上将 proposal 放入输入，适用于学习“给定纠错意图后的执行响应”。
如果训练的是**模式选择器**，`correction_mode`、`mode_reason`、
`diagnosis`、`recovery_goal` 和新 condition 应作为输出监督，不能预先
放进决策输入。只使用决策时已经可见的观测、已有条件及历史反馈。

另外保留 step-level transition 便于动作/状态对齐与 tracking-error 学习。
保留失败、未执行预测后缀的掩码、提前终止长度、未评估标签和所有纠错尝试。

防止数据泄漏：

1. Post observation / 最终成功标签属于监督目标，不能混进当时 Agent 的输入。
2. `scene.npz` 中的 object poses、BBox、body state 是仿真特权信息。
   本轮 tools 调试明确使用过这些信息，不是纯视觉 Agent；按
   `agent_input_mode` / `observation_access` 分开建集。
3. Agent-only 没有 WAM prediction / continuation 是正常的缺省，
   不要造零数组或伪造 WAM 输出补齐。
4. 同一历史 genesis 派生的多个实验及其 replay 高度相关；
   按 genesis/任务家族分组切分，避免随机逐步划分导致 train/test 泄漏。
5. Hosted Agent 的精确模型版本未暴露时保留 unknown，不反推或猜测版本。

## 10. V2 的可审计纠错比较约定

本节是新代码的执行约定，不是对前述 12 条历史轨迹的重新标注。

### 模式理由

新 `recover` proposal 必须包含非空 `mode_reason`，解释为什么需要改
action、condition 或两者。已有 `diagnosis` 解释故障，`recovery_goal`
解释目标；三者不能混为一个标签。理由仍是 Agent 判断，不是最优模式的证明。
Agent-only 的固定动作接口不参与三模式比较。

### 条件变更用途

每个 WAM 请求和 `wam_prediction` 都含：

```json
{
  "condition_change": {
    "before": "Grasp the block",
    "after": "Keep holding the block and carry it above the bin",
    "changed": true,
    "kind": "subgoal_progression",
    "reason": "The grasp stage is complete; begin transport."
  },
  "attempt": null
}
```

- `initial` / `unchanged`：自动记录，reason 为 null。
- `subgoal_progression`：正常子目标推进，必须给出理由。
- `recovery_rewrite`：纠错改写，必须给出理由。正式 Condition/Joint
  使用 proposal 的 `mode_reason` 作为理由，并关联 attempt。
- `condition_changed` 事件只在文本变化时产生，首次条件也包括在内。
  统计纠错次数时必须排除 `initial`，并区分用途；普通 `wam` 改写的
  attempt 为 null，不会自动变成正式 Recovery attempt。

### 统一窗口与 Action 条件锁

三种模式都执行固定 **16 步 WAM** 后才允许局部评估：

| 模式 | 先执行局部控制 | 评估窗口条件 |
|---|---|---|
| Action | 是 | 严格等于本次纠错前的条件 |
| Condition | 否 | 改写后的条件 |
| Joint | 是 | 动作修复之后使用改写后的条件 |

新 Condition/Joint 必须真的改变条件；传入的 `wam_execute_steps` 如存在
必须为 16。窗口由 worker 内部执行，不能通过外部 `wam` 请求插入另一条
指令。Action 组只在比较窗口内锁定条件；评估后可以有明确标注的子目标变化。

`recovery_evaluation_completed.evaluation` 示例：

```json
{
  "condition_before": "Grasp the block",
  "condition": "Grasp the block",
  "start_step": 100,
  "end_step": 116,
  "requested_steps": 16,
  "executed_steps": 16,
  "complete": true,
  "interrupted_by": null
}
```

这仅是 schema 示例，不是一条新采集的真实轨迹。窗口实际动作对应
101–116 步。环境提前终止时 `executed_steps` 可以为 0–15，
`complete=false`，并保留 `native_success`、`native_horizon` 或
`native_termination` 中断原因；不会补步、reset 或伪造局部评估。
若恰好执行满 16 步时终止，窗口完整，但不会再接受 Agent 的评估请求。

`recovery_attempt_completed` 和 session/manifest 也保留窗口对象。
新 `recovery_assessment.assessment` 额外包含 `evaluation_steps`、
`evaluation_end_step`、`label_source="agent_assessment"`，绑定统一的
观察窗口。窗口完成、记录校验通过都不代表任务成功。

校验器对 V2 输出 `comparison_contract_verified`、
`complete_evaluation_windows`、`incomplete_evaluation_windows`。
V1 的 `comparison_contract_verified=null`：缺少新约定不是负例，
也不能把旧 Action/Joint 的不同观察窗口当作统一比较。训练和统计时按
protocol_version 分组，并保留模式理由、条件用途与窗口完整性。
