# 面向长时域动作生成的去噪反馈式循环潜在规划

> **英文候选题目：** Recurrent Latent Planning with Denoising Feedback for Long-Horizon Action Generation  
> **整理日期：** 2026-09-17  
> **研究主线：** 长时域动作序列与机器人策略；视频生成作为辅助验证。  
> **文档性质：** 文献调研与研究方案，不是已经完成实验验证的论文结论。文中“建议”“假设”及稳定性推导，不能被表述为现有模型已获得的能力。

---

## 摘要

本研究拟在 Cosmos 3 的统一理解与生成架构中，引入固定容量的循环潜在规划状态，使其在动作去噪过程中读取候选动作及其预测后果，并通过保留原始条件的残差接口修订后续生成。

核心问题不是“重复运行 Reasoner 是否有效”，而是：

> 在相同真实观测和可比计算预算下，经过训练的循环规划器能否利用动作生成中的反馈，修正多阶段动作计划，并减轻预测时域加长后的性能下降？

该方向在架构与数学上可行，但有三项必须通过实验建立的前提：

1. 循环计算确实产生了有用的规划修订，而非表征漂移、塌缩或冗余计算。
2. 动态规划条件能够被 Generator 正确利用，而非破坏原有条件分布与动作信息聚合。
3. 收益能够转化为真实或仿真环境中的策略成功率，而不只是更平滑的动作或更连贯的视频。

初步推荐采用“固定原始缓存 + 循环规划 slots + 单次去噪反馈 + 逐层残差控制”的最小方案，先在原生动作时域验证，再扩展长时域。

## 1. 问题背景与架构核实

### 1.1 Cosmos 3 的原始信息流

根据 Cosmos 3 官方技术报告 §2.3、§5.3 [1]：

- Reasoner 处理语言及视觉理解 token。
- Generator 处理视频、音频、动作等连续模态 token，通过迭代去噪生成。
- 两个塔具有各自的归一化、投影及前馈参数，通过 joint attention 交互。
- Reasoner token 只在 AR 子序列中进行因果注意力，不读取正在去噪的 DM token。
- Generator token 可以同时读取 Reasoner 和 Generator 两组 K/V。

第 $\ell$ 层的 Generator 注意力可以写成：

$$
O_G^\ell
=
\operatorname{Attn}
\left(
Q_G^\ell,
[K_{R,0}^\ell;K_G^\ell],
[V_{R,0}^\ell;V_G^\ell]
\right).
$$

下标 $0$ 表示原始固定条件。Reasoner 的输出依赖输入条件，不依赖当前去噪状态，因此可以计算一次并缓存。

### 1.2 必须修正的四个初始判断

**判断一：固定 prefill 并不等于浪费推理计算。**

对固定输入 $o,l$，每次重新计算 $R(o,l)$ 得到相同结果。报告明确指出，缓存与每步重算在数学上等价。因此，关闭缓存并重复 forward 不会自动增加推理能力。

**判断二：固定 K/V 不意味着 attention 固定。**

Generator 的 queries 以及自身 K/V 随去噪变化，因此读取固定 Reasoner 记忆的方式本来就在变化。新机制应描述为“更新规划记忆及其形成过程”，而不是“首次让 attention 随去噪变化”。

**判断三：Cosmos 3 并非完全没有使用生成前推理。**

报告 §6.3.2 已描述 Reasoner prompt upsampling：将简短指令扩展为结构化的时空场景与事件计划。附录 E.1 也表明，更适合 Physical AI 的理解塔能够改善 Generator 的领域表现。

更准确的不足是：

> 现有生成路径主要依赖生成前形成的条件，缺少根据生成中的候选动作及其预测后果进行迭代修订的机制。

**判断四：Cosmos 3 不能简单理解为 Reasoner 连接一个 Wan DiT。**

报告中双塔由预训练 VLM／Reasoner 权重初始化，并使用 Wan 的视频 VAE。使用 Wan VAE 与使用 Wan 作为整个 Generator 不是同一件事。

### 1.3 与机器人实验直接相关的原生配置

报告 §4.2.5 中，Cosmos3-Nano-Policy-DROID：

- 输入当前本体状态与三视角视觉观察。
- 预测 32 个未来绝对关节位置动作。
- 动作频率为 15 Hz，对应约 2.13 秒的动作窗口。
- 联合预测辅助 RGB 视频。
- 推理采用 4 个去噪步骤、noise shift 5、CFG scale 3。
- 跳过视频 latent 解码以降低开销。

注意：跳过 VAE 解码，并不等于消除了 Generator 内部视频 token 的计算。

以上是报告中的具体配置，不应泛化为所有 Cosmos 3 变体的固定限制。

## 2. 文献调研与创新边界

### 2.1 Looped／recurrent-depth Transformer

| 工作 | 主要机制与证据 | 对本研究的启发 | 不能推出的结论 |
|---|---|---|---|
| Universal Transformers，ICLR 2019 [2] | 固定 token 位置，沿深度共享参数更新；支持自适应计算 | 固定规划状态、多轮更新的架构基础 | 普通预训练模型可以零样本任意循环 |
| Reasoning with Latent Thoughts，2025 [3] | 循环模型在若干推理任务上优于参数量匹配浅模型；提供表达能力分析 | 共享计算可以增加有效深度 | 理论存在性构造等于现有 checkpoint 的直接能力 |
| Huginn／Recurrent Depth，2025 [4] | 输入锚定、共享 recurrent core、随机循环训练；分析循环轨迹及失败模式 | 最重要的循环训练与表征分析参考 | 更多循环必然更好，或所有成功状态都必须收敛 |
| Relaxed Recursive Transformers，ICLR 2025 [5] | 预训练层共享化、结构初始化、逐深度 LoRA、继续训练 | 普通预训练模型可以经过适配改造成递归结构 | 只复制层就能恢复能力，或可以无限外推循环次数 |
| Coconut [6] | 将上一位置 hidden state 作为下一位置连续输入 | 潜在推理及其训练课程的参考 | 与固定 slots 的 recurrent depth 完全相同 |

#### Huginn 的关键启示

其架构可以概括为：

$$
e=P(x),\qquad
s_{r+1}=F_\phi(e,s_r),\qquad
y=C(s_R).
$$

每次循环既读取固定输入锚点 $e$，也读取上一轮计算状态 $s_r$。这与重复运行固定输入的普通 forward 不同。

论文观察到：

1. 失败训练中，不同 token 的表征可能越来越相似，最终塌缩。
2. 模型也可能忽略循环状态，使一次和多次循环的表现近似相同。
3. 成功模型的轨迹不一定单调收敛，也可能表现为轨道或定向漂移。
4. 训练采用长尾随机循环预算；测试超过平均循环次数，不等于严格测试了训练中从未出现过的深度。

因此，应分别评价：

- 数值是否有界；
- token 是否仍具有区分性；
- 读出是否稳定；
- 任务表现是否随计算增加而改善。

#### 深度编码没有统一正确答案

UT 使用位置与循环步信号；Huginn 主架构不显式输入循环步数；Relaxed Recursive Transformers 通过逐深度 LoRA 区分不同展开位置。

如果目标包含未见循环次数外推，应单独对照无深度编码、函数式深度编码和有限 learned embedding，而不是默认某一种方式必然正确。

### 2.2 与机器人潜在规划最接近的已有工作

**ThinkAct** [7] 已经研究视觉潜在规划与动作执行的连接。

**Fast-ThinkAct，CVPR 2026** [8] 的 §3.2–3.3 已经实现：

- 自回归产生少量连续 latent reasoning token。
- 生成空间／轨迹规划 token。
- 提取这些 token 对应的逐层 K/V。
- 将规划 K/V 接入扩散动作模型的条件侧。
- 冻结推理模型等部分组件，训练下游策略。

因此，以下内容不能单独作为创新：

> “使用 latent reasoning，并通过 K/V 指导 diffusion policy。”

建议明确比较：

| 维度 | Fast-ThinkAct 已核实的方法 | 拟议方案 |
|---|---|---|
| 状态组织 | 自回归产生连续 latent token | 固定数量 slots，反复修订 |
| 条件形成 | 动作生成前形成规划条件 | 去噪中继续计算并修订 |
| 信息方向 | 规划器到动作模型 | 规划器与候选动作双向交互 |
| 接口重点 | 潜在规划到策略的连接 | 保留原始条件，约束更新带来的干扰 |

固定 slots、循环深度和残差模块各自都有先例。本调研支持该组合的研究动机，但不能认证其为“首次”。

### 2.3 去噪过程中改变条件的相关工作

| 工作 | 实际干预 | 对本研究的启发与限制 |
|---|---|---|
| Wan，2025 [9] | Flow matching；文本条件通过 cross-attention 注入 DiT | 可作为视频迁移实例；Reasoner K/V 与 Wan 条件不天然兼容 |
| Prompt-to-Prompt，2022 [10] | 在选定去噪阶段控制 cross-attention map | 可以保留结构锚点、分阶段释放新语义；全程保持旧 attention 可能过约束 |
| CADS，ICLR 2024 [11] | 条件噪声退火、均值／尺度校准 | 时变条件可行，但尺度变化可能严重损害质量；多样性收益不等于策略可靠性 |
| Guidance in a Limited Interval，NeurIPS 2024 [12] | 仅在部分噪声区间增强 CFG | 不应默认全程加强控制；切换与数值步边界应对齐 |
| FreeNoise，ICLR 2024 [13] | 噪声重排、时序滑窗、分阶段及分帧 prompt 控制 | 长序列还涉及注意力范围、跨段关联；不能只靠增强语义条件 |

#### 对 Wan 的具体启示

Wan 使用 flow matching，而不是简单套用 DDPM 的 epsilon-prediction 公式。论文和代码可能使用不同时间方向，迁移时必须对齐速度符号及 scheduler 约定。

其原生文本条件来自文本编码器，经投影进入各层 cross-attention。Reasoner 的 self-attention K/V 与该接口在维度、head 基底、归一化、位置编码和语义角色上可能不同。

**张量尺寸相同，不等于接口分布匹配。**

#### 对条件调度的具体启示

- Prompt-to-Prompt 说明，维持结构与接受新语义之间存在权衡。
- CADS 直接展示了推理时修改条件的可行性，也显示条件统计校准非常重要。
- Guidance Interval 改变的是 CFG 外推力度，区间外仍可使用普通 conditional prediction，并非删除全部条件。
- FreeNoise 将去噪时间与视频帧位置分开处理，提醒本研究不能混淆“何时修改规划”和“动作应在未来何时发生”。

这些结果不是针对四步动作生成器的通用最优调度。它们提供设计依据，具体噪声窗口仍需实验。

## 3. 建议研究命题与假设

### 3.1 核心命题

> 在保留原始 Reasoner 条件缓存的基础上，学习一个固定容量的循环潜在规划状态，使其利用动作去噪中的候选反馈修订计划，并通过噪声感知的残差接口影响后续生成。

### 3.2 可证伪的研究假设

| 假设 | 所需证据 |
|---|---|
| H1：循环计算产生更有用的规划 | 相同观测与动作时域下，训练后的循环优于静态条件及计算匹配模块 |
| H2：候选反馈具有独立收益 | 优于全部预先计算、仅定时释放条件的对照 |
| H3：保留锚点的接口减少生成干扰 | 相比直接覆写 K/V，短任务退化、flow 扰动或失效率更低 |
| H4：方法减轻长预测时域的性能下降 | 在各方法长窗口训练条件一致时，随 horizon 增长保持更好的执行表现 |
| H5：新增计算值得其延迟成本 | 在匹配端到端延迟／计算预算下，优于增加 Generator 评估等简单方案 |

不建议主张：

- 固定 prefill 表明原模型完全没有推理。
- 更多循环自动意味着更强 reasoning。
- 动态条件天然破坏扩散模型的数学合法性。
- 小 residual 或 LayerNorm 自动保证稳定。
- 长动作预测能够替代真实环境反馈。
- latent reasoning 加 K/V 条件化本身是新的研究范式。

## 4. 方法设计

### 4.1 符号与四个独立时间轴

| 符号 | 含义 |
|---|---|
| $\tau$ | 真实环境中的控制时刻 |
| $\sigma$ | 去噪噪声水平，本文采用 $1\to0$ |
| $k$ | 离散采样／反馈节点 |
| $r$ | 某节点内的循环深度 |
| $\ell$ | Transformer 层编号 |
| $H_{\mathrm{plan}}$ | 一次预测的动作长度 |
| $H_{\mathrm{exec}}$ | 获取新观测前实际执行的动作长度 |
| $T_{\mathrm{task}}$ | 整个任务时长 |
| $M$ | 固定规划 slot 数量 |

增加循环深度 $r$ 不会自动增加输出长度 $H_{\mathrm{plan}}$，更不等于可以增加无反馈执行长度 $H_{\mathrm{exec}}$。

### 4.2 固定的原始条件锚点

正常运行 Reasoner：

$$
C_0=
\left\{
K_{R,0}^{\ell},V_{R,0}^{\ell}
\right\}_{\ell=1}^{L}.
$$

第一阶段不覆写 $C_0$，也不改变原始 AR attention mask。

这样可以：

- 保持原模型已有的语义与视觉条件；
- 保留原始缓存路径；
- 避免将原 AR 路径直接改造成需要读取噪声 DM token 的新网络；
- 清楚地关闭新增机制并恢复基线。

### 4.3 固定容量的循环潜在规划器

令：

$$
Z_{k,r}\in\mathbb R^{M\times d}.
$$

采用共享更新块：

$$
Z_{k,r+1}
=
Z_{k,r}
+
\eta_r F_\phi
\left(
\operatorname{LN}(Z_{k,r}),E,B_k,\sigma_k
\right).
$$

其中：

- $E$：固定的观察、指令和 Reasoner 语义锚点；
- $B_k$：候选动作及预测后果的反馈摘要；
- $F_\phi$：跨循环共享的计算模块；
- $\eta_r$：更新步长或门控。

**实现建议：**

1. 优先复用或初始化自 Reasoner 的部分层，结合输入映射与共享 LoRA 训练。
2. 保留固定数量 slots，不随循环轮数不断追加条件 token。
3. 物理未来时间锚点与噪声、循环编号分别编码。
4. 不将最终层 hidden state 直接替换所有层 K/V；逐层读出需要显式适配。
5. 第一版先在单次动作生成内部延续状态；跨真实控制周期的 warm-start 作为独立实验。

如果仅在最终 embedding 后连接随机小 Transformer，即使有效，也不能单独证明“原 Reasoner 推理能力被进一步利用”。需要使用初始化、冻结／解冻和规划监督等对照厘清来源。

### 4.4 保留基线的残差注意力接口

保留原始 joint attention：

$$
O_{\mathrm{base}}^\ell
=
\operatorname{Attn}
\left(
Q_G^\ell,
[K_{R,0}^\ell;K_G^\ell],
[V_{R,0}^\ell;V_G^\ell]
\right).
$$

新增独立规划分支：

$$
O^\ell
=
O_{\mathrm{base}}^\ell
+
g_\ell(\sigma)
W_{\mathrm{out}}^\ell
\operatorname{Attn}
\left(
Q_G^\ell,
K_P^\ell(Z),
V_P^\ell(Z)
\right).
$$

投影需要匹配对应层的 head 及输出维度。该设计不要求把新的规划记忆直接加入原 joint-attention 的同一个 softmax 分母。

初始可将 $W_{\mathrm{out}}^\ell$ 置零，其他分支参数保持可学习且 gate 非零，确保初始新增输出为零，又保留学习入口。不要同时将所有输出通路和 gate 全部置零。

冻结且未改动的底座上，新增分支为零应恢复基线；若之后训练了底座 LoRA，要恢复原始模型还需要同时关闭这些改动。

### 4.5 候选动作反馈

采用 Cosmos 3 的 flow 参数化：

$$
X_\sigma=(1-\sigma)X_0+\sigma\epsilon,
\qquad v^*=\epsilon-X_0.
$$

当前速度预测给出：

$$
\widehat X_0
=
X_\sigma-\sigma v_\theta(X_\sigma,\sigma,C).
$$

从该估计构造反馈：

$$
B_k
=
E_{\mathrm{fb}}
\left(
\operatorname{stopgrad}(\widehat X_{0,k}),
o,l,\sigma_k
\right).
$$

可优先提取：

- 候选动作轨迹与末端执行器路径；
- 可达性及关节约束；
- 子目标完成顺序；
- 抓取／接触状态预测；
- 辅助视频中与任务有关的对象关系。

第一版建议优先使用动作与几何摘要，避免每轮完整视频解码和视觉重编码。

注意事项：

- $\widehat X_0$ 是估计，不是真实未来。
- 高噪声候选可能存在较大不确定性，不能直接当成可靠视觉观察。
- 不应把原始噪声 latent 无适配地交给只训练过正常图像输入的 Reasoner。
- 可使用前一步已经产生的候选结果，在下一完整采样步生效，避免同一步重复运行完整 Generator。

### 4.6 第一版交互流程

```text
真实观察 + 本体状态 + 指令
            |
            v
      原始 Reasoner
            |
       固定条件 C0 --------------------------+
            |                               |
            v                               v
     循环潜在规划 Z ------ 残差接口 ------ Generator
            ^                               |
            |                               v
            +------ 候选动作／预测后果摘要 ---+
                                            |
                                            v
                                     最终动作序列
                                            |
                                     执行前 H_exec 步
                                            |
                                        获取新观测
```

建议先只在四步采样中安排一次反馈更新，按实际 $\sigma$ 网格选择位置，而不是默认每步交互。

如果 Reasoner 从不读取 Generator 状态，循环结果原则上可提前全部计算，再按噪声阶段释放。这属于开环条件调度，不应包装成候选反馈闭环。

## 5. 数学分析

本节为方法分析与条件性推导，不是已验证的整体稳定性定理。

### 5.1 循环表征的误差传播

对：

$$
Z_{r+1}=F(Z_r),
$$

局部扰动满足：

$$
\delta Z_R
\approx
J_{R-1}J_{R-2}\cdots J_0\,\delta Z_0,
\qquad
J_r=\frac{\partial F}{\partial Z}(Z_r).
$$

一次映射有效，不代表自组合仍然有效。普通预训练模型的单次训练目标，没有自动约束该 Jacobian 乘积。

原始层从某一深度分布映射至另一深度分布；再次把输出送回该层，可能改变输入的语义坐标和统计分布。LayerNorm 只能部分调节统计，不能保证这些角色一致。

### 5.2 小残差不自动产生收缩

对残差循环：

$$
Z_{r+1}=Z_r+\eta F(Z_r),
$$

局部 Jacobian 为：

$$
J_{\mathrm{loop}}=I+\eta J_F.
$$

若 $J_F$ 存在正实特征值 $\lambda>0$，则该方向的放大因子为：

$$
1+\eta\lambda>1.
$$

因此，减小 $\eta$ 可以减缓增长，却不能仅凭这一点保证收缩。

实践上应同时监测：

- 状态范数与相邻轮变化；
- token 相关性、有效秩；
- 规划读出与动作读出的变化；
- 增加循环后的任务表现。

不建议把“所有 hidden states 都收敛”作为唯一目标。对决策有影响的读出方向稳定，可能比所有内部方向静止更重要。

### 5.3 修改 Reasoner keys 会改变两组信息的竞争

对某个 Generator query，令两组 logits 的归一化分母贡献为：

$$
Z_R=\sum_i e^{a_i},
\qquad
Z_G=\sum_j e^{b_j}.
$$

joint attention 可以精确分解为：

$$
O=\beta O_R+(1-\beta)O_G,
\qquad
\beta=\frac{Z_R}{Z_R+Z_G},
$$

其中 $O_R,O_G$ 分别使用各自组内归一化后的权重。

因此，更新 Reasoner keys 同时改变：

1. Reasoner token 之间的读取路由；
2. Reasoner 条件与 Generator 自身信息的总占比；
3. Generator 内部时空／动作信息的有效贡献。

即使内容相同，将 Reasoner K/V 复制 $m$ 份也会使：

$$
\beta'
=
\frac{mZ_R}{mZ_R+Z_G}.
$$

故 token 数量是必须控制的混杂因素。追加零 value 的 token 也不一定无扰动，因为其 key 仍可能改变 softmax 分母。

### 5.4 K/V 插值与输出插值不同

记：

$$
A(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt d)V.
$$

通常：

$$
A(Q,(1-g)K_0+gK_1,(1-g)V_0+gV_1)
\ne
(1-g)A(Q,K_0,V_0)+gA(Q,K_1,V_1).
$$

只有固定 $Q,K$、仅线性改变 $V$ 时，才有对应的线性关系。该结论是局部的；value 改动仍可能影响后续层和去噪步骤的 queries。

因此，输出端残差更容易定义局部扰动，但仍不是最终动作稳定性的保证。

### 5.5 时变条件改变速度场，不意味着数学上非法

原采样：

$$
\frac{dX_\sigma}{d\sigma}
=v_\theta(X_\sigma,\sigma,C_0).
$$

动态条件采样：

$$
\frac{dX_\sigma}{d\sigma}
=v_\theta(X_\sigma,\sigma,C_\sigma).
$$

在适当的可测性、局部 Lipschitz 和增长条件下，新速度场仍可定义解，并将初始噪声推送到一个输出分布。

不能保证的是：

- 输出仍对应原模型在某个固定条件下的目标分布；
- 更新后的条件处在训练分布内；
- 新分布的动作更好或更可靠。

如果规划器保留历史，应将 $(X,Z)$ 作为联合状态分析。不能只分析独立 Reasoner 的稳定性，再假定耦合系统同样稳定。

直接把 $C_\sigma$ 作为速度网络输入，不需要额外向速度添加 $dC/d\sigma$。链式项会出现在速度时间导数、复合能量导数等不同问题中。

### 5.6 局部小改动如何累积

将采样时间统一为从噪声到数据的 $u\in[0,1]$。假设：

$$
\dot X=f(X,u),\qquad
\dot{\widetilde X}=f(\widetilde X,u)+r(\widetilde X,u).
$$

若相关区域内：

$$
\|r(x,u)\|\le\delta(u),
$$

且 $f$ 对状态的 Lipschitz 常数为 $L(u)$，则同初值时有：

$$
\|\widetilde X_1-X_1\|
\le
\int_0^1
\delta(s)
\exp\left(\int_s^1L(u)\,du\right)ds.
$$

这说明：

- 小 gate 不等于小 attention 输出扰动；
- 小 feature residual 不等于小 flow residual；
- 小 flow residual 可能被后续动力学放大；
- 小动作距离也不等于相同任务结果，尤其在接触与抓取边界附近。

对于具有状态反馈的控制器，必须计入反馈对联合系统敏感性的影响，不能将上式直接当作未经条件检查的全局保证。

### 5.7 CFG 与低步数采样的影响

若：

$$
v_{\mathrm{CFG}}
=v_{\mathrm{uncond}}
+w(v_{\mathrm{cond}}-v_{\mathrm{uncond}}),
$$

且只改变 conditional 分支，则同一状态上：

$$
\Delta v_{\mathrm{CFG}}=w\,\Delta v_{\mathrm{cond}}.
$$

原生策略 $w=3$ 时，该局部速度变化被放大三倍；最终动作变化不一定恰好三倍，因为后续轨迹也会变化。

用一阶更新说明步长作用：

$$
X_{k+1}=X_k+\Delta\sigma_k v_k,
\qquad
\Delta X_{k+1}=\Delta\sigma_k\Delta v_k.
$$

这不是假定原 sampler 必然采用 Euler，而是提醒应关注：

$$
|\Delta\sigma_k|\,\|\Delta v_k\|.
$$

四步采样的每次更新占据较大区间，因此调度应绑定实际噪声水平及步长，而不是照搬五十步视频模型中的固定“中期”。

### 5.8 求解器与条件切换

建议：

- 硬切换发生在完整数值步边界；
- 同一 predictor/corrector 内保持一致的控制模式，或一致评估明确定义的平滑时间函数；
- 对多步求解器，比较保留历史、切换点重启历史、暂时降阶等处理；
- 以相同模型评估预算比较，而不是默认某种处理一定更好。

时变条件不必然使 solver 失效，但不连续切换可能降低跨切换点的高阶外推精度。

## 6. 训练方案

### 阶段 A：静态循环规划与接口适配

目标：先验证额外循环是否形成有用规划。

1. 冻结原始 Reasoner、Generator 底座作为起点。
2. 训练循环模块、输入适配器、逐层读出和残差输出。
3. 循环先全部在动作生成前完成。
4. 采样不同循环预算，例如 $R\in\{1,2,4\}$。
5. 验证新增分支关闭时的基线恢复。

若阶段 A 收益不明确，先排查表征、监督和接口；不要立即把复杂动态闭环当作补救。

### 阶段 B：加入单次去噪反馈

基础目标：

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E
\left[
\left\|
v_{\theta,\phi}(X_\sigma,\sigma,C_0,Z)
-(\epsilon-X_0)
\right\|^2
\right].
$$

可加入：

$$
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\lambda_{\mathrm{plan}}\mathcal L_{\mathrm{plan}}
+
\lambda_{\mathrm{budget}}\mathcal L_{\mathrm{budget}}.
$$

其中：

- $\mathcal L_{\mathrm{plan}}$：子目标顺序、关键路点、对象关系等监督。
- $\mathcal L_{\mathrm{budget}}$：对新增机制引起的速度变化施加软预算。

预算示例：

$$
\mathcal L_{\mathrm{budget}}
=
\left[
\|v_{\mathrm{aug}}-v_{\mathrm{base}}\|-b(\sigma)
\right]_+^2.
$$

比较应在同一采样状态、同一条件配置下进行，并明确是否使用最终 CFG 输出。该约束是局部控制启发式，不是全局安全保证。

不能把所有场景都强行蒸馏回原模型，否则必要修正也会被压制。

#### 闭环训练分布必须覆盖

独立前向加噪得到的 $X_\sigma$，未必覆盖实际闭环采样访问的状态与规划记忆。

应考虑：

- 短采样展开；
- 采样轨迹回放；
- 训练中匹配反馈节点、循环预算、gate 和状态延续方式；
- 初期对反馈支路使用 stop-gradient，降低耦合训练复杂度。

规划标签可以来自示范未来，但真实未来只能作为监督，不能作为部署时不可获得的输入。

### 阶段 C：扩展动作时域

从 $H_{\mathrm{plan}}=32$ 扩展至 $64,128$，同步检查：

- 动作 token 形状与时间位置编码；
- 更长窗口的数据覆盖；
- 辅助视频的长度和时间密度；
- Generator 的长序列适配；
- 归一化与控制接口；
- 端到端延迟。

各基线必须获得相同的长窗口训练条件。不能将“长窗口训练的收益”归因于循环 reasoning。

## 7. 长时域动作与真实闭环

### 7.1 长预测不等于长时间开环执行

必须分别报告：

$$
H_{\mathrm{plan}},\qquad
H_{\mathrm{exec}},\qquad
T_{\mathrm{task}}.
$$

Diffusion Policy [14] 已使用 receding-horizon control：预测一段动作，但不一定全部执行后才观察。

建议首先固定 $H_{\mathrm{exec}}$，研究更长预测是否改善当前动作选择；再单独增加 $H_{\mathrm{exec}}$，测量减少环境反馈的代价。

### 7.2 内部反馈不等于新增真实信息

Reasoner 与 Generator 的交互可能发现候选计划不一致，但不能获得现实中尚未观测到的物体滑动、抓取失败或外部扰动。

需要区分：

1. **模型内部闭环：** 规划状态与候选动作互相修订。
2. **执行环境闭环：** 动作执行后获取新观察并重规划。

新增循环也可能使观察在动作下发前变得更陈旧。离线规划质量提升，不必然转化为在线控制收益。

### 7.3 动作误差与状态误差

若环境动力学 $s_{t+1}=f(s_t,a_t)$ 在相关区域分别对状态和动作满足 Lipschitz 条件，则同初始状态下：

$$
\|\delta s_{t+H}\|
\le
L_a
\sum_{j=0}^{H-1}
L_s^{H-1-j}\|\delta a_{t+j}\|.
$$

该条件性界说明动作误差可能沿时域累积。接触、碰撞或离散任务阶段切换附近，平滑性假设可能不成立，更不能从小动作差异直接推导任务成功。

## 8. 实验设计

### 8.1 核心对照

| 编号 | 方法 | 回答的问题 |
|---|---|---|
| B0 | 原始固定 prefill | 基础策略表现 |
| B1 | 生成前显式规划 | 是否已有文本规划就足够 |
| B2 | Fast-ThinkAct 式静态 latent 规划接口 | 是否只是已有 latent-to-policy 范式的收益 |
| B3 | 固定 slots 循环，全部在生成前完成 | 循环计算本身的收益 |
| B4 | 预先计算各轮状态，在去噪中定时释放 | 条件调度的收益 |
| B5 | 循环 + 真实候选动作反馈 | 内部反馈的独立收益 |
| B6 | 同范数随机／错配反馈 | 排除扰动或模块容量解释 |
| B7 | 同预算增加 Generator 评估 | 是否优于普通计算增量 |

另加入参数量／计算量匹配的非循环模块，以分离权重共享、深度和容量。

### 8.2 消融变量

- 循环次数：训练内预算 $1,2,4$；额外测试 $8$ 时明确标为外推。
- 规划 slots：固定 $M$，与增加 slots 的方案分开。
- 反馈：无反馈、正确反馈、错配反馈。
- 更新节点：不同实际 $\sigma$ 边界；先单次，再多次。
- 接口：直接覆盖 K/V、value-only 更新、独立输出残差。
- 锚点：保留与移除固定条件。
- 训练：固定循环数与随机循环数。
- 状态：reset 与 warm-start。
- 时域：$H_{\mathrm{plan}}=32,64,128$，固定执行窗口后再单独改变。

不用一次展开所有组合；按“规划是否有效 → 反馈是否有效 → 接口是否稳健 → 长时域是否受益”的顺序进行。

### 8.3 主指标

**执行表现：**

- 多阶段任务成功率；
- 子目标完成率与顺序正确性；
- 失败后恢复能力；
- 外部扰动鲁棒性；
- 碰撞／约束违反；
- 无效动作与动作抖动；
- 随预测时域、执行时域增长的退化曲线。

**效率：**

- 端到端规划延迟；
- 控制时限内完成比例；
- Generator 模型评估次数；
- Reasoner／循环模块计算量；
- 反馈编码与同步开销；
- KV 与激活显存。

**机制诊断：**

- 每层 AR／DM attention mass；
- 新分支输出范数；
- 最终 guided flow 的变化；
- 规划状态有效秩与 token 相关性；
- 不同循环深度的子目标读出；
- 正确／错配反馈的因果对照。

视频一致性、预测轨迹平滑度和潜在状态可解码性均为辅助指标，不能替代策略成功率。

### 8.4 必做的信息不足对照

比较：

1. 在旧观察上增加循环；
2. 获取新观察后使用较少循环；
3. 获取新观察后 warm-start 循环。

该实验区分“计算不足”与“信息过时”，也检查状态复用是否引入旧计划干扰。

### 8.5 公平性要求

- 相同观察输入、任务指令及动作表示。
- 相同训练数据与长窗口覆盖。
- 比较规划机制时固定真实环境反馈频率。
- 如需从 DROID 关节动作迁移到其他基准，所有方法使用一致的任务适配接口。
- 显式计入 CFG、反馈编码和采样器子阶段的计算。
- 固定 seed 做配对比较，并报告多次 rollout 的不确定性。
- 同时包含短任务，检查长时域收益是否牺牲基本控制能力。

## 9. 最小可行研究路线

### 第一阶段：最小原型

1. 保持原生 $H_{\mathrm{plan}}=32$。
2. 使用固定数量规划 slots。
3. 训练 $R=1,2,4$ 的循环预算。
4. 保留原始缓存。
5. 使用逐层残差分支。
6. 四步采样中仅一次反馈更新。
7. 先完成 B0、B3、B4、B5、B7 对照。

**阶段判断：** 若 B5 无法优于 B3／B4，应优先检查反馈是否可靠、是否真正被使用，不宜直接声称闭环规划有效。

### 第二阶段：表征与稳定性分析

- 比较直接覆写 K/V 与残差分支。
- 检查 token 数量和 attention mass 的影响。
- 检查不同循环次数的表征变化。
- 测试错配反馈、噪声节点及速度扰动预算。

### 第三阶段：长时域验证

- 所有基线一致扩展到 $64,128$ 动作窗口。
- 固定执行窗口，测量长预测的收益。
- 单独增加执行窗口，测量反馈减少的代价。
- 加入多阶段任务与外部扰动。

### 第四阶段：视频辅助验证

在动作主结论成立后，研究预测视频是否更符合动作、对象关系和事件顺序。Wan 可作为不同生成架构上的迁移实例，而非替代主策略实验。

## 10. 风险与缓解

| 风险 | 可观察症状 | 优先缓解方式 |
|---|---|---|
| 原 Reasoner 不适应循环 | 增加循环后性能下降或状态漂移 | 输入锚定、循环适配训练、共享 LoRA |
| 循环被忽略 | 多轮输出与单轮近似相同 | 循环预算与状态干预消融、规划监督 |
| 条件接口失配 | 生成质量退化、flow 扰动明显 | 逐层适配、零输出初始化、保留底座 |
| 新 keys 挤占 DM 信息 | AR attention mass 异常增大 | 固定 token 数、独立残差分支 |
| 反馈不可靠 | 重复修正、错误子目标、计划反复 | 噪声感知摘要、单次更新、不确定性分析 |
| 四步内更新过激 | 个别步骤引起大动作变化 | 按实际步长诊断、控制 guided flow 变化 |
| 内部循环延迟过高 | 离线收益无法转为在线成功 | 小规模共享模块、缓存锚点、减少反馈节点 |
| 长时域结论混杂 | 仅更长训练或更频繁观察带来收益 | 同训练窗口、同执行反馈频率对照 |
| 视频指标误导 | 视频更静止却未完成任务 | 以执行成功、子目标顺序为主指标 |

## 11. 建议论文贡献表述

可采用如下三点：

1. **方法：** 面向低步数动作 flow 模型，构建固定容量、候选动作反馈驱动的循环潜在规划状态。
2. **接口与分析：** 保留原始条件锚点，通过逐层残差分支控制规划更新，并分析 joint-attention 竞争、CFG 和采样步长对干预的影响。
3. **实验：** 分离循环深度、条件调度、内部反馈与动作时域的贡献，证明其在可比计算与真实反馈预算下的执行收益。

推荐摘要式表述：

> 现有统一理解与生成模型通常使用生成前形成的条件来指导连续动作去噪，缺少利用候选动作反馈修订潜在计划的机制。我们研究一种保留原始条件缓存的循环潜在规划框架，以固定容量的规划状态读取动作生成中的预测反馈，并通过噪声感知的残差接口影响后续生成。通过分离循环计算、条件释放时机和反馈路径的对照，检验该机制是否能够在可比计算预算下改善多阶段动作执行，并缓解预测时域扩展带来的性能下降。

上述文字仍是研究目标；实验完成前不应改写成已实现的性能结论。

## 12. 总体判断

- **架构上可实现。**
- **动态条件在数学上可以成立，不必然破坏采样器。**
- **普通预训练 Reasoner 不能直接任意循环，需要接口与循环训练。**
- **小残差是保守起点，不是全局稳定性证明。**
- **循环深度与长动作输出是不同问题，必须分开验证。**
- **内部候选反馈不能替代真实环境观察。**
- **最有价值的贡献，是证明反馈驱动的规划修订具有独立、同预算且可执行的收益。**

---

## 参考文献与原始来源

[1] NVIDIA. **Cosmos 3: Omnimodal World Models for Physical AI.** 2026.  
[官方技术报告](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) · [官方代码与模型入口](https://github.com/NVIDIA/Cosmos)  
重点：§2.3 架构；§4.2 flow 目标；§4.2.5 DROID 策略；§5.3 缓存；§6.3.2 prompt upsampling；附录 E.1 理解塔消融。

[2] Dehghani et al. **Universal Transformers.** ICLR 2019.  
https://arxiv.org/abs/1807.03819

[3] Saunshi et al. **Reasoning with Latent Thoughts: On the Power of Looped Transformers.** 2025.  
https://arxiv.org/abs/2502.17416

[4] Geiping et al. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.** 2025.  
[论文](https://arxiv.org/abs/2502.05171) · [官方代码](https://github.com/seal-rg/recurrent-pretraining)  
重点：循环架构与训练、失败模式、状态轨迹、预算泛化与停止条件。

[5] Bae et al. **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.** ICLR 2025.  
https://arxiv.org/abs/2410.20672

[6] Hao et al. **Training Large Language Models to Reason in a Continuous Latent Space.** 首次公开于 2024.  
[论文](https://arxiv.org/abs/2412.06769) · [官方代码](https://github.com/facebookresearch/coconut)

[7] Huang et al. **ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning.** 2025.  
https://arxiv.org/abs/2507.16815

[8] Huang et al. **Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning.** CVPR 2026.  
[论文](https://arxiv.org/abs/2601.09708) · [HTML 正文](https://arxiv.org/html/2601.09708)  
重点：§3.2 连续潜在推理；§3.3 逐层规划 KV 与扩散动作模型的连接。

[9] **Wan: Open and Advanced Large-Scale Video Generative Models.** 2025.  
[论文](https://arxiv.org/abs/2503.20314) · [官方代码](https://github.com/Wan-Video/Wan2.1)

[10] Hertz et al. **Prompt-to-Prompt Image Editing with Cross Attention Control.** 2022.  
[论文](https://arxiv.org/abs/2208.01626) · [官方项目](https://prompt-to-prompt.github.io/)

[11] **CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling.** ICLR 2024.  
https://arxiv.org/abs/2310.17347

[12] **Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models.** NeurIPS 2024.  
[论文](https://arxiv.org/abs/2404.07724) · [官方实现](https://github.com/kynkaat/guidance-interval)

[13] **FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling.** ICLR 2024.  
[论文](https://arxiv.org/abs/2310.15169) · [官方代码](https://github.com/AILab-CVC/FreeNoise)

[14] Chi et al. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.** RSS 2023.  
[论文](https://arxiv.org/abs/2303.04137) · [官方项目](https://diffusion-policy.cs.columbia.edu/)

### 引用与使用说明

- 本文献集合面向当前研究问题，不是穷尽式的新颖性检索。
- 动态网页和技术报告可能更新；正式论文应固定实际使用的版本、访问日期及代码提交。
- 图像／视频编辑论文提供接口与采样设计依据，不能直接作为机器人执行收益的证据。
- 本文数学界均有相应假设，不能被引用成对完整机器人闭环系统的无条件安全保证。
