# Edge Policy 循环规划：实现进度与接口

更新日期：2026-09-17。

基于[研究方案](cosmos3_recurrent_latent_planning_idea.md)及设计审查。目标模型为
`nvidia/Cosmos3-Edge-Policy-DROID`，数据为 DROID，执行验证为 RoboLab/Panda。
本文记录工程实现进度。规划器注册、paired-FM 训练、验证、四步反馈推理已接通，
已通过小型 VFM 单卡/四卡测试，以及公开 Edge 权重和真实 DROID 子集上的两阶段训练 smoke。
反馈 checkpoint 已重新加载并生成有限的32步动作；尚未完成正式研究实验或环境闭环验证。
用户约束：GPU 仅使用物理 0–3 卡，不能改用其他卡。

## 当前范围

| 项目 | 状态 |
|---|---|
| UniPC 完整步后事件、输入/噪声/速度配对 | 已实现，CPU 测试覆盖 |
| 二阶 UniPC 历史重启 | 已实现，时间网格与当前位置不重置 |
| CFG 分支记录、生成入口透传 | 已实现，单 rank、顺序 CFG |
| 合法 paired-FM 状态构造 | 已接入 training_step，动作/视频共用同一时间网格 |
| 原生 attention/VFM 规划状态接口 | 已接通普通和 gen-only forward，GPU 小模型验证通过 |
| 固定 UND 锚点输出、packer 动作 mask、conditional-only 状态 | 已接入训练和请求级推理会话，CP=1 |
| Reasoner 初始化的核心与动作残差 | 支持 meta 预注册、materialize 后初始化和已有底座 LoRA 复制 |
| Edge DROID 普通 SFT 配方及参数白名单 | 四卡 dry-run、解冻集合测试通过 |
| 四卡 FSDP | 小型 VFM 规划训练、activation checkpoint、EMA、DCP保存恢复通过 |
| 规划器配置化注册、paired-FM training_step、确定性 validation | 已实现；static/scheduled/feedback/control 四组 |
| 公开HF到DCP原生权重一致性 | 549个tensor、6,739,314,048 bytes逐元素一致；不等于动作输出等价 |
| 真实数据SFT和反馈训练 | GPU0–3，各完成2次优化器更新、验证、checkpoint保存 |
| 真实规划checkpoint重载推理 | 验证集1个窗口通过；32×8动作有限，4步/5分支评估 |
| 200步checkpoint配对诊断 | 20个固定窗口、raw/EMA、四种干预，共160次评估完成 |
| B0-CONT/B4/B5归因比较 | B0-CONT/B4独立200步完成；9episode、18窗口、324次评估，配对hash通过 |
| RoboLab rollout、正式效果对照 | 未执行 |

## 200 步稳定性验证

已执行 [run_edge_stability.sh](run_edge_stability.sh)：仅GPU0–3，从共同两步SFT
checkpoint新建反馈规划实验，128个episode子集、seed42、全局batch4、200次更新。
学习率`1e-4`，warmup10步，周期200步，末端比例0.1；每25步验证、每50步保存。
报告见 [edge_stability_200.json](validation/edge_stability_200.json)。

| 指标 | 结果 |
|---|---|
| 优化器更新 | 完成200步，保存50/100/150/200步checkpoint |
| 初始 / 最终validation objective | 1.167829 / 1.180564（高约1.1%，没有改善） |
| 进程峰值显存（每卡最大值） | allocated约22.50 GiB，reserved约23.65 GiB |
| 后段迭代日志中位耗时 | 6.14秒，包含保存节点，不含初始预热格式 |
| 恢复加载 | 成功读取第200步model/optimizer/scheduler/trainer，未追加更新 |
| 重载推理 | 单个val窗口32×8未来动作有限，4步/5分支+1prefill，约2.36秒 |

本次工程稳定性通过，但不能判定方法有效；所有验证节点均未优于初始化。
验证仅每rank五个batch，不是完整验证集；OFU的dmon采样有告警，不能据此报告可靠利用率。
日志没有全rank非有限loss事件，不等于已逐项审计所有梯度的原始有限性。
暂不自动进入更大数据/更长训练，应先检查同条件的control/B4对照及监督有效性。

本次运行目录为
`outputs/edge_droid_feedback_stability_200_seed42/cosmos3_action/edge_droid_planning/action_policy_droid_edge_feedback_stability_200_seed42`，
最终checkpoint是其下的`checkpoints/iter_000000200`。
根输出目录还包含`train.log`、`resume_check.log`和`inference.log`。
重跑原脚本会恢复到已完成的200步；本轮已结束，无遗留训练进程。
离线推理工具现支持`--checkpoint`显式指定该目录，不再只能读取两步smoke权重。

## 200 步结果诊断

[diagnose_edge_planning.py](../packages/cosmos3/tools/diagnose_edge_planning.py)
已实际评估第200步checkpoint，报告为
[edge_stability_200_diagnostics.json](validation/edge_stability_200_diagnostics.json)。
仅GPU0，CUDA可见范围0–3；无参数更新。训练集随机选5个episode，验证集覆盖全部
5个episode（27、36、51、55、117），每个episode取首尾两个有效窗口，共20个窗口。
两类权重、四种干预共160次评估，评估循环耗时365.29秒（不含模型/数据准备）。

工具从DCP显式加载`net.*`（raw）或`net_ema.*`（EMA），各1244个tensor，BF16计算。
训练器验证使用EMA；此前保存YAML的离线推理路径直接加载`net.*`，使用的是raw，
不能仅凭`use_ema_weights`默认值判断这条加载路径的权重。此次两种权重单独报告。
每次验证固定噪声seed1729，保持原始监督目标，记录anchor/bridge的动作/视频分项。
干预之间严格断言bridge输入、sigma与未干预候选来源相同；关闭规划组允许候选随源模型变化。

下表为验证集的episode等权均值；动作/视频分别对anchor和bridge取均值，越低越好。
现有权重均为10，总目标等于`10 * (action + video)`，已逐组核验。

| 权重 | 候选/规划条件 | 动作FM | 视频FM | 总目标 |
|---|---|---|---|---|
| raw | 正确反馈 | 0.00286541 | 0.11845296 | 1.21318380 |
| raw | 反转候选时间顺序 | 0.00286800 | 0.11844782 | 1.21315813 |
| raw | 零候选 | 0.00286092 | 0.11844236 | 1.21303279 |
| raw | 关闭全部规划 | 0.00307582 | 0.11841664 | 1.21492461 |
| EMA | 正确反馈 | 0.00286370 | 0.11842436 | 1.21288064 |
| EMA | 反转候选时间顺序 | 0.00285795 | 0.11842290 | 1.21280853 |
| EMA | 零候选 | 0.00285533 | 0.11841922 | 1.21274552 |
| EMA | 关闭全部规划 | 0.00308642 | 0.11843079 | 1.21517208 |

解释与限制：

- 相比关闭规划，正确反馈的动作FM约低6.8%（raw）/7.2%（EMA），5个验证episode
  的首尾窗口均值方向一致；总目标仅低约0.14%/0.19%。视频项约占总目标97.6%，
  因此总loss不适合作为唯一动作指标。这不是梯度贡献测量，不据此直接调整loss权重。
- 正确反馈未优于零候选，时间反转也未稳定恶化指标。候选干预会改变bridge动作速度
  （RMS约0.0035–0.0038），说明通路可影响输出，但尚无反馈内容有效性的证据。
- raw与EMA总目标接近，EMA选择不是本次现象的主要解释。新窗口与原训练器验证窗口
  不同，不能把这里的1.21288直接接到原1.18056的验证曲线上。
- 关闭规划只是同checkpoint消融，不是另行训练的control，也不是严格恢复初始化EMA。
  零候选仍经过编码器、state/sigma和后两轮规划，不等于无反馈B4。
- 只有5个验证episode、每episode两个相关窗口、固定噪声种子和一个checkpoint；
  train侧是训练划分样本，不保证这些窗口被200步训练实际抽到。没有显著性结论。
- 这里是合法paired-FM状态上的诊断，没有运行真实采样轨迹干预、独立测试集或环境rollout。

从工作区根目录复现（`RUN_DIR`为上节完整运行目录）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
  packages/cosmos3/.venv/bin/python packages/cosmos3/tools/diagnose_edge_planning.py \
  --run-dir "$RUN_DIR" --checkpoint "$RUN_DIR/checkpoints/iter_000000200" \
  --data-root "$PWD/datasets/droid_plus_lerobot_640x360_20260412" \
  --report "$PWD/loop_reasoner/validation/edge_stability_200_diagnostics.json"
```

### 已锁定对照

2026-09-17启动前锁定：本轮将独立训练B0-CONT与B4各200步，B5使用既有200步，
不追加训练。内部experiment键`control`对外标为B0-CONT：更新原LoRA/动作头，
沿用与规划组相同的anchor+bridge paired-FM监督，不是换成单分支普通SFT目标。
B4预计算2+2轮并定时释放，不读取候选；共享新增模块采用同构初始化流程和seed42。
共同W_SFT、训练前128episode、增强配置、loss权重10/10、监督sigma分布、globalbatch4、
200次更新和调度周期均保持不变。固定第200步EMA为主、raw为敏感性，不按验证loss选点。
历史B5未记录逐batch窗口/噪声hash，因此只能确认相同数据配方和曝光预算，不能声称
历史训练随机性逐样本严格配对；raw/EMA也不是独立复现。

离线比较固定前512episode中的全部官方验证episode，每episode首尾两窗口；
资源已准备并核对：episode 27、36、51、55、117、135、382、437、464，共9个、18个窗口。
仅新增验证视频，不把512前缀标为可训练完整数据；下载清单为`validation_prefix_512.json`。
样本准备随机种子`42+dataset_index`，FM噪声seed1729，保存窗口/指令清单并逐方法核对。
额外核验实际anchor/bridge的动作、视频及文本输入SHA256一致；不匹配则拒绝汇总。
报告动作、关节、夹爪、加权动作/视频和总目标，按episode聚合配对差值并bootstrap 95% CI。
CI只反映本次episode抽样，不覆盖训练seed或站点差异。训练集不扩展、窗口长度不改变。
主要比较仍是合成paired-FM，不把FM上的差值等同执行收益。

真实采样的后续固定口径：4步UniPC、shift5、首步CFG3；B4/B5均在第二步结束重启历史，
B0-CONT应额外报告同位置restart-only对照。不以原生不重启B0和重启B5的差异归因反馈。
本轮FM比较不调用solver，不混入不匹配的采样结果。

启动前B5 EMA诊断已完成60次评估和2个batch的梯度分析，见
[edge_b5_mechanism_preflight.json](validation/edge_b5_mechanism_preflight.json)。
同任务供体为同episode另一时刻；错配供体按不同episode、指令词集合Jaccard<0.35筛选，
再选当前状态最接近者，来源与距离均写入报告。这是启发式可审计错配，不代表精确任务标签。
零候选是编码器输入前原始零向量，编码器含bias及state/time/sigma，不等于无候选状态。
三个环节均观察到非零变化，不能认定通路未使用候选；有益修订尚未证实。
所测slots/encoder/output-up梯度中，两个batch的加权动作梯度范数都大于视频，
部分方向相反；只覆盖episode27两个窗口和指定参数组，不据此重加权或宣称全网梯度主导。

[run_edge_stability.sh](run_edge_stability.sh)现支持`PLANNING_MODE=control|scheduled|static|feedback`，
默认feedback不变，各组自动使用独立job和输出目录。control使用原LoRA学习率`2e-5`、
动作头`1e-5`；规划组使用`1e-4`。control与B4已实际各完成200步，结果见下一节。
共同起点仍为两步SFT，seed42、全局batch4、200步、相同划分和调度周期；
这是相同曝光预算，不是相同计算量，也没有逐batch核验训练样本顺序。

```bash
PLANNING_MODE=control bash loop_reasoner/run_edge_stability.sh
PLANNING_MODE=scheduled bash loop_reasoner/run_edge_stability.sh
```

两组按顺序执行，不并发挤占GPU0–3。先比较control/B4/B5的固定窗口动作分项，
再补真实四步采样的匹配restart干预与环境验证。B5若只优于关闭规划、却不优于
独立B4或错配反馈，应优先检查反馈监督和候选分布，而非直接延长训练。

## 归因实验结果

[run_edge_attribution.sh](run_edge_attribution.sh)已顺序完成B0-CONT与B4各200步，
B5只读取既有200步权重。每组实际保存model/optim/scheduler/trainer；
三组训练数据配置、增强、噪声/损失配置、seed、更新预算、scheduler与共同起点逐项核对一致。
每组200个更新日志齐全，无记录到的全rank非有限loss事件。后段迭代日志中位耗时分别为
B0-CONT 5.09秒、B4 9.28秒、B5 6.14秒；运行时段不同且包含系统抖动，不能据此断言B4计算更多。

期间发现并修复B0-CONT的EMA初始化缺陷：warm-start跳过`net_ema.*`加载后，原生钩子
没有同步已加载的raw底座，导致初始EMA验证loss 56.354315。该次运行已停止，日志和
checkpoint保留在旧control目录，但不进入比较。修复仅在warm-start复制raw到EMA，
resume不覆盖已恢复的EMA，新增回归测试通过。有效B0-CONT从共同SFT重新开始200步，
使用`outputs/edge_droid_control_stability_200_seed42_ema_fix`；脚本默认control已指向此目录。
修复后初始验证1.167836，与B4 1.167752、既有B5 1.167829接近。
这也是不能只检查训练末端loss而省略起点审计的原因。

最终报告：[edge_attribution_200.json](validation/edge_attribution_200.json)。
原始记录：[B0-CONT](validation/edge_control_200_comparison.json)、
[B4](validation/edge_scheduled_200_comparison.json)、[B5](validation/edge_feedback_200_comparison.json)。
三个方法各18窗口、raw/EMA；B5另有重复正确、零候选、时间反转、同任务、错配、关闭规划干预，
共324次评估。窗口、完整任务清单、实际anchor/bridge输入哈希和sigma跨方法一致。
主结果固定EMA，不把raw当独立重复；所有统计以episode为单位，bootstrap2000次、seed42。

| EMA主结果（越低越好） | 动作FM | 7维关节FM | 夹爪FM | 总目标 |
|---|---|---|---|---|
| B0-CONT | 0.00160290 | 0.00146415 | 0.00257418 | 1.20928586 |
| B4 | 0.00263168 | 0.00253362 | 0.00331809 | 1.21982949 |
| B5 | 0.00264942 | 0.00254913 | 0.00335139 | 1.21993385 |

| 配对动作差值（后者减前者） | 均值 | episode bootstrap 95% CI | 后者改善/退化episode |
|---|---|---|---|
| B4 - B0-CONT | +0.00102877 | [+0.00097838, +0.00110440] | 0 / 9 |
| B5 - B0-CONT | +0.00104651 | [+0.00098773, +0.00113957] | 0 / 9 |
| B5 - B4 | +0.00001774 | [+0.00000554, +0.00003695] | 1 / 8 |

B0-CONT的动作FM比B5低约39.5%；关节与夹爪也在9个episode上均优于B5。
B5比B4高约0.67%，raw方向相同。当前数据/预算下，更多原策略适配明显强于新增规划路径，
并未观察到B5超出B4的价值；不能由此证明所有预算下规划或反馈都无效。
这轮没有控制新增参数容量、优化器参数组或学习率最优性，也没有多训练seed复现。

反馈三级诊断（EMA，18窗口平均RMS，相比正确候选）：

| 候选干预 | 反馈编码变化 | slots变化 | 动作速度变化 |
|---|---|---|---|
| 原候选重复评估 | 0 | 0 | 0 |
| 原始零向量 | 0.311130 | 0.007131 | 0.003654 |
| 同episode另一时刻候选 | 0.090084 | 0.001976 | 0.003611 |
| 不同任务近状态候选 | 0.054404 | 0.000996 | 0.003459 |

报告同时保留L2、RMS、参考范数与相对范数。重复正确候选误差为零，因此替换候选确实
改变了编码、slots和速度；结论是“已使用反馈，但未观察到有益修订”，不是“未使用反馈”。
零候选、同任务供体、错配供体相对正确候选的动作差值95%区间分别为
`[-8.80e-6, +2.49e-7]`、`[-2.17e-5, +4.57e-6]`、`[-8.39e-6, +3.10e-6]`，均跨零。
没有把显著改变输出当作改善预测，也没有把原始零候选当作无候选B4。

启动前梯度报告的两个batch未锁定数据集提示措辞，只用于局部梯度排查，不参与最终方法排名。
最终比较已固定样本准备seed，重复候选校验也通过。当前9episode中4个为新扩展覆盖，
CI仍不代表训练seed/站点不确定性，也未对多项比较做多重检验校正。
所有结果仍为合成paired-FM离线结果；本轮未执行真实采样前缀的三组比较或仿真rollout。

建议暂停追加B5训练，以B0-CONT为下一轮实际采样基线，先做相同四步采样与匹配restart
条件下的动作误差/轨迹稳定性，再决定是否值得针对反馈监督或训练-推理分布偏移另开实验。
不改本轮loss权重、不扩长窗口，不把这轮小样本负结果推广成机制不可能有效。

```bash
bash loop_reasoner/run_edge_attribution.sh
```

重跑会恢复已完成的两组200步并重新离线评估，不是新训练seed或从头复现；
需要独立复现时应另行指定新实验目录和seed，不覆盖本轮记录。

## 本机有效资源

资源与运行摘要见[edge_asset_smoke.json](validation/edge_asset_smoke.json)，
逐tensor转换报告见[edge_dcp_weights.json](validation/edge_dcp_weights.json)，
真实重载动作见[edge_feedback_inference.json](validation/edge_feedback_inference.json)。

以下路径相对于工作区根目录：

| 资源 | 位置 |
|---|---|
| 公开HF snapshot | `models/Cosmos3-Edge-Policy-DROID` |
| 转换后DCP | `models/Cosmos3-Edge-Policy-DROID-DCP` |
| DROID子集父目录 | `datasets/droid_plus_lerobot_640x360_20260412` |
| VAE | `cookbooks/cosmos3/generator/action/finetune/checkpoints/wan22_vae/Wan2.2_VAE.pth` |
| SFT checkpoint | `outputs/edge_droid_smoke/cosmos3_action/edge_droid_warmstart/action_policy_droid_edge/checkpoints/iter_000000002` |
| 反馈checkpoint | `outputs/edge_droid_feedback_smoke/cosmos3_action/edge_droid_planning/action_policy_droid_edge_feedback/checkpoints/iter_000000002` |

数据固定为官方 `nvidia/Cosmos3-DROID` 的
`5c11a20accb11497270a5247a7f1e66ad04c956c` 版本。全量约758 GB，本次仅下载
前128个成功episode引用的13个数据/视频文件及元数据，清单为数据父目录中的
`smoke_prefix_128.json`。保留官方episode编号和元数据，不冒充全量数据。
原始划分后有123个train episode、5个val episode；去掉不足窗口长度的轨迹后，
得到49,731个训练窗口与2,460个验证窗口。loader必须显式设置`episode_limit=128`。

本机TorchCodec缺共享FFmpeg库，smoke使用已安装的CPU PyAV，配置`video_backend=pyav`。
train/val实际解码得到 `(3,33,540,640)` 三视角视频和 `(33,8)` 动作（含当前状态），
动作数值有限。已有VAE有效，未重复下载。

本机一键smoke入口为[run_edge_smoke.sh](run_edge_smoke.sh)，已配置真实资源路径、
本地tokenizer和GPU0–3。脚本两次更新、每rank一条样本、累积1次，区别于正式训练预算。

```bash
bash loop_reasoner/run_edge_smoke.sh

BASE_CHECKPOINT_PATH="$PWD/outputs/edge_droid_smoke/cosmos3_action/edge_droid_warmstart/action_policy_droid_edge/checkpoints/iter_000000002" \
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/edge_droid_feedback_smoke" \
RECIPE=examples/toml/sft_config/action_policy_droid_edge_recurrent.toml \
bash loop_reasoner/run_edge_smoke.sh
```

原输出目录已有checkpoint，重跑相同命令会自动恢复，不是从头重新训练。
如需独立复现实验，设置新的输出目录。两阶段末次validation objective分别为
SFT `0.940172`、feedback `1.087555`；目标定义不同且仅两次更新，不能解释为效果排名。

真实运行暴露并修复了FusedAdam空本地分片问题：保留空分片优化器状态，但不把
零元素张量送入Transformer Engine multi-tensor kernel。四卡回归现已使用生产FusedAdam，
而非只测试PyTorch AdamW。

真实反馈checkpoint重载后，验证窗口生成 `(33,8)`（去掉state行为32步），
4次采样、5次CFG分支评估、额外1次锚点prefill。单次未暖机生成约2.2–2.5秒，
不代表p95延迟，也不满足0.533秒执行窗口预算；未下发动作到机器人或仿真环境。

## 采样合同

[UniPCSampler](../packages/cosmos3/cosmos_framework/model/generator/diffusion/samplers/unipc.py)
新增可选关键字参数：

- `post_step_callback`：所有样本完成当前完整数值步后调用一次。
- `restart_after_steps`：例如 `(1,)` 表示第二步结束后重启多步历史。

默认不启用，保持旧输出和旧采样前 `step_callback` 的语义。
`UniPCStepEvent` 提供请求内 sample index、step index、下一索引、整数 timestep、
浮点 sigma、下一 sigma、输入、实际速度、步后状态及 solver 的 clean prediction。
dataclass 字段不可重赋值；张量是独立快照，用户修改快照不会修改采样状态。
它不是全局请求标识，持久日志需由调用方补充 request/session ID。

候选必须来自：

$$
\widehat X_0=X_{\mathrm{input}}-\sigma_{\mathrm{input}}v_{\mathrm{effective}}.
$$

`post_step_sample` 不能代替 `model_input_sample`；整数 timestep 除以 1000
不能代替 solver 的浮点 sigma。事件保留的 clean prediction 来自当前
`scheduler.step(return_dict=False)` 的第二个返回值，已与上述公式做测试对照。
规划会话按 DROID 的 `[vision | action]` 布局提取候选，验证张量长度与来源 sigma，
去掉条件 state 行，只保留前 8 个有效动作维。其他模态/布局不在首版支持范围内。

[restart_history](../packages/cosmos3/cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py)
清空 `model_outputs`、`timestep_list`、`last_sample`、阶数预热状态，
保留 `sigmas`、`timesteps`、步索引和 RNG。不允许初始/最终节点重启；
当前仅支持 order=2、flow prediction、predict_x0、无 thresholding/solver_p。
重启不保证结果等于不重启版本，因此必须保留 restart-only 对照。

## CFG 与生产入口

[generate_samples_from_batch](../packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py)
新增 `evaluation_callback`、`restart_after_steps` 与 `planning_state_provider`。
provider 只在 conditional 分支调用；收到整数 timestep，精确 sigma 必须从调用方
持有的真实网格取得，不得简单除以 1000。unconditional 分支不接收规划状态。
[StepFeedbackRecorder](../packages/cosmos3/cosmos_framework/model/generator/diffusion/samplers/step_feedback.py)
在同一次评估内记录 conditional/unconditional prediction，再与 solver 事件配对。
它不通过 `velocity_postprocess_builder` 工作，不为记录日志增加模型评估。

保持现有严格区间 `low < timestep < high`。构造 shift=1、有效采样 shift=5、
四步、1000 训练时刻时，CPU 测得模型 timestep 为 `[999, 937, 833, 624]`。
区间 `(960,1001)` 只有第一步启用 CFG，顺序路径总计五次分支评估。
区间外返回 conditional；额外计算了 unconditional 不代表语义上启用了 CFG。

新推理入口显式拒绝多 rank、batched CFG、velocity postprocess 和 diffusion residual cache。
[RecurrentActionPolicyModel](../packages/cosmos3/cosmos_framework/model/generator/recurrent_action_policy_model.py)
自动建立请求内状态：B3 提前完成四轮；B4 提前保存两轮/四轮状态、按节点释放；
B5 先两轮，在第二个完整 step 后从真实候选再更新两轮，第三/第四步复用同一状态。
普通原始 text K/V 缓存机制不被覆盖，规划 K/V 当前每次重新计算。

预算注意：回调记录的五次 CFG 分支评估不包含额外一次锚点 prefill，因此当前
完整请求实际有六次 Generator forward，以及四轮共享规划核心计算。
不能将这里的小模型接口测试等同于公开模型 cache-on/off 等价验收。
推理目前单 rank，训练支持四 rank；这两个支持范围不同。

## 合法配对训练数据

[RectifiedFlow.get_paired_interpolation](../packages/cosmos3/cosmos_framework/model/generator/diffusion/rectified_flow.py)
接收显式 sample IDs、clean、epsilon、两组 scalar sigma、condition/valid masks，
复用已有 `get_interpolation`。不重新采随机噪声，不改变 RNG；动作和视频分别调用，
同一模态的两个状态共用 epsilon，不同模态使用各自噪声。

要求 `0 < sigma_low < sigma_high <= 1`。conditioned token 保持 clean，
无效维置零；`velocity_target` 在有效 noisy 范围等于 `epsilon-clean`，
其他位置为零，`loss_mask` 明确排除这些位置。现有 action/video 损失权重与
有效维分母仍由训练入口处理。

首轮已接入的训练目标：

$$
\mathcal L=\tfrac12(\mathcal L_{\mathrm{anchor}}+\mathcal L_{\mathrm{bridge}}).
$$

两项只监督合法 FM 状态；bridge 使用高噪声候选作辅助条件，但监督 conditional
velocity，不监督 CFG 外推结果。真实采样回放不进入首轮恢复损失。
合法插值不等于消除了合成候选与部署候选的分布差异。

训练具体约定：来源固定为第二个模型评估节点（sigma约0.937266），低噪声监督从
第三/第四节点选择；沿同一个示范与同一份各模态epsilon构造所有状态。
anchor sigma沿用原训练分布。数据管线CFG丢弃设为0，在anchor分支内部做0.1的条件丢弃，
bridge始终为conditional。丢弃条件时直接禁止该sample的规划残差，unconditional不读规划slots。
来源前向为eval/no_grad；只detach候选，反馈编码器仍参与反向。
第三/第四节点使用同一更新sigma，不能因抽到不同监督节点改变反馈规划时刻。

每批四次Generator前向：锚点prefill、候选来源、anchor监督、bridge监督。
`planning_generator_forwards`、`planning_supervised_forwards` 和 `planning_core_rounds`
记录成本；等监督数据不意味着三组训练FLOPs相等。control也使用相同两项FM监督，
但没有planner，继续训练已有Generator LoRA与动作头。

`validation_step` 保存/恢复CPU、当前CUDA、Python、NumPy RNG以及网络train/eval状态，
同一批次和模型可重复。验证数据关闭图像增强、CFG丢弃和iterable shuffle；
复用现有episode级train/val划分（默认seed42，val比例0.03）。这是本次继续训练的
留出集，尚无独立scene/site/test划分，也不保证发布模型未见过这些episode。

## 规划组件边界

[recurrent_latent_planner.py](../packages/cosmos3/cosmos_framework/model/generator/mot/recurrent_latent_planner.py)
提供：

- `RecurrentLatentPlanner`：复制调用方给定 dense MoT Reasoner 层的投影、norm、FFN；
  原矩阵冻结，复用现有 LoRA 类，每轮共享核心，显式传入/返回固定数量 slots。
- `CandidateActionEncoder`：只 detach 候选动作输入，编码器输出保留梯度。
- `ActionPlanningResidual`：独立 attention 与低秩零输出投影；直接写入范围仅限
  属于 conditional sample 的 noisy action queries。其他位置返回零。

工厂支持从 meta 源层预注册结构；materialize 后显式调用 `initialize_from`，
从已加载的底座复制对应权重、重置新 LoRA 和零输出投影。
已有 Generator LoRA 会合并进复制的基底投影，不改变源模型。
该初始化只用于新分支 warm-start，不能在规划 checkpoint resume 时执行。
已支持DTensor权重复制：源参数临时聚合后按目标mesh/placement分发；逐参数执行，
不是把整个公开模型在每卡完整复制。规划模块在根FSDP前注册，规划计算也在根forward内执行。
EMA使用相同结构；新分支初始化后同步，resume不重置。
循环update_scale使用一维单元素参数以支持FSDP分片。

原生 `PackedAttentionMoT` 在原输出投影后加入独立残差，使用归一化 pre-RoPE query。
`PlanningState.bind` 将 packer 的 `action.mse_loss_indexes` 映射到 GEN 流，
排除条件 state 行和 padding；每层显式接收不可重赋值的 `PlanningAttentionContext`。
`denoise` 和 VFM forward 接收 `planning_state`；`return_planning_anchors=True`
返回完整 UND prefill 的 padded hidden states 和 mask，不对 gen-only 缓存路径伪造锚点。
GPU 小型两层 VFM 验证了锚点不随 noisy action 或规划残差变化。
`PlanningRequest` 把锚点、本体状态、噪声与候选显式送入已注册的 `ActionPlanningModule`。
可学习state投影补充当前本体信息；没有直接读取未来真值的反馈入口。
请求状态不跨环境控制周期复用，也不保存在全局可变slots中。
后续 joint attention 仍可能将动作变化传播至视频，action-only 不代表视频完全不变。

## 四卡普通 SFT

[Edge 配方](../packages/cosmos3/examples/toml/sft_config/action_policy_droid_edge.toml)
从公开策略的转换后 DCP warm-start，不恢复其优化器，也不跳过已有动作头。
默认 4-way FSDP、CP/CFG=1、LoRA rank/alpha=16、每 rank 一条样本、累积 8 次、
最多 20 次优化器更新；这是 smoke 起点，不是研究最终训练预算。
学习率为 LoRA `2e-5`、既有动作头 `1e-5`；原位置 embedding 保持冻结。

[ActionPolicySFTModel](../packages/cosmos3/cosmos_framework/model/generator/action_policy_sft_model.py)
在 LoRA 注入后、FSDP 前显式解冻 `action2llm`、`llm2action`、`action_modality_embed`。
缺少任何头则报错；参数集合单测确认原 Reasoner 和其他基底参数保持冻结。

启动脚本固定 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `--nproc-per-node=4`。
运行前必须设置 `DROID_ROOT`、`BASE_CHECKPOINT_PATH`、`WAN_VAE_PATH` 与独立
`IMAGINAIRE_OUTPUT_ROOT`。已通过的 dry-run 使用占位路径，只证明配置可解析，
不证明数据可读取、真实 checkpoint 可加载或显存足够。

```bash
DRYRUN=1 bash examples/launch_sft_action_policy_droid_edge.sh
```

公开模型索引引用1014个tensor、4个文件，safetensors结构检查通过；
已逐元素验证转换后的549个原生策略tensor。lazy vision tower和完整动作输出的
HF/DCP数值等价仍需单独验收，不能从参数一致性直接替代这些检查。

## 四卡规划训练

共同SFT完成后，`BASE_CHECKPOINT_PATH` 指向该SFT生成的DCP，而不是再次指向发布权重。
规划配方会保留共同SFT的LoRA；仅新planner/residual与EMA可跳过加载。
[规划 TOML](../packages/cosmos3/examples/toml/sft_config/action_policy_droid_edge_recurrent.toml)
默认启用feedback。每组使用不同的`IMAGINAIRE_OUTPUT_ROOT`，禁止误恢复其他组。

```bash
RECIPE=examples/toml/sft_config/action_policy_droid_edge_recurrent.toml \
  DRYRUN=1 bash examples/launch_sft_action_policy_droid_edge.sh

RECIPE=examples/toml/sft_config/action_policy_droid_edge_recurrent.toml \
  bash examples/launch_sft_action_policy_droid_edge.sh -- \
  experiment=action_policy_droid_edge_scheduled job.name=action_policy_droid_edge_scheduled
```

其他experiment名称为 `action_policy_droid_edge_static`、`action_policy_droid_edge_feedback`、
`action_policy_droid_edge_control`。仅改`model.planning.mode`不能切换到control模型类，
应按上例选择注册experiment。默认20次更新仅作smoke，正式比较需锁定统一曝光量。

恢复同一个输出目录的训练使用原命令，由DCP恢复model/optimizer/trainer/RNG，
模型钩子检测resume并跳过初始化。若从已有规划DCP显式warm-start到新目录，必须同时设置：

```text
model.planning.initialize_from_base=false checkpoint.keys_to_skip_loading=[]
```

保留默认`initialize_from_base=true`只适用于从共同SFT创建新规划分支，不能用于读取已训练的规划权重。
推理加载器现已在实际加载权重后调用可选`on_inference_weights_loaded`钩子；规划模型
要求`initialize_from_base=false`才进入就绪状态，普通模型不受影响。该钩子的单测
验证不会重置参数，但尚未用公开规划DCP通过独立policy server启动真实rollout。
普通HF导出／现有独立policy server是否保留和实例化新增结构尚未验收；
不得仅把一个规划DCP路径替换进原服务命令并声称模型自动加载了规划能力。

真实资源现已就绪并由本机smoke脚本配置。非dry-run启动仍检查数据目录、
DCP的model子目录与VAE文件存在性；使用子集时还必须保留episode_limit配置。

## 聚焦验证

在 `packages/cosmos3` 工作目录，使用仓库现有虚拟环境；系统 Python 缺训练依赖。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TEST_MAX_GPUS=4 LD_LIBRARY_PATH='' .venv/bin/python -m pytest -p no:env \
  cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py \
  cosmos_framework/model/generator/diffusion/samplers/step_feedback_test.py \
  cosmos_framework/model/generator/diffusion/rectified_flow_test.py \
  cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
  cosmos_framework/model/generator/mot/recurrent_latent_planner_test.py \
  cosmos_framework/model/generator/action_policy_sft_model_test.py \
  cosmos_framework/model/generator/recurrent_action_policy_model_test.py --num-gpus=0 --levels=0 -q
```

覆盖零开销关闭行为、候选三元组、批内隔离、restart-only、真实生成入口的 CFG 次数、
合法 FM 标签恒等式、条件/padding mask、零残差、更新后的核心/编码器梯度和模块权重恢复。
GPU测试覆盖原生attention、两层VFM、三种模式的完整training_step/validation及
三种模式的四步生成（四轮规划、五次采样分支评估）。四rank测试运行真实小型VFM，
覆盖规划器DTensor初始化、activation checkpoint重算、3次更新、EMA与DCP模型/优化器恢复。
当前聚焦集合为70项CPU测试、9项单卡GPU测试全部通过，四rank测试每个rank通过。
单卡集合还验证了等监督control只更新原LoRA/动作头，不创建planner。
这些不是公开 Edge 权重等价或真实任务成功率证据。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TEST_MAX_GPUS=4 LD_LIBRARY_PATH='' \
  .venv/bin/python -m pytest -p no:env \
  cosmos_framework/model/generator/mot/recurrent_latent_planner_test.py \
  cosmos_framework/model/generator/recurrent_action_policy_model_test.py --num-gpus=1 --levels=1 -q

CUDA_VISIBLE_DEVICES=0,1,2,3 TEST_MAX_GPUS=4 LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest -p no:env \
  cosmos_framework/model/generator/recurrent_action_policy_model_test.py::test_four_rank_recurrent_training_initialization_and_ema \
  --num-gpus=4 --levels=1 -q
```

`-p no:env` 很重要：仓库 pytest-env 默认删除 `CUDA_VISIBLE_DEVICES`，而 torchrun
下的 fixture 不会重新设置它。禁用插件可保留本任务的 0–3 卡限制；
`TEST_MAX_GPUS=4` 使用现有测试框架开关，不需要修改全局 GPU 枚举。

## 下一实施顺序

1. 本轮独立B0-CONT/B4同曝光对照已完成，暂不追加B5训练或调整loss权重。
2. 以B0-CONT为基线，固定多噪声种子/episode窗口，补真实采样轨迹的反馈与restart-only干预。
3. 扩展公开权重的预处理/denoiser/四步等价和缓存一致性检查（逐参数转换已通过）。
4. 验收独立server及RoboLab闭环协议，再考虑更大数据、多seed和非循环容量对照。
5. 扩展严格测试划分与统计汇总，开展32/32、32/8仿真；不提前声称长时域或实时部署收益。

静态循环无收益是诊断信号，不是拒绝小预算反馈试验的硬门槛。
本轮不包含正式训练、真实回放恢复目标、异步控制或长窗口扩展。