# Literature Cortex 种子库关系指南

_本指南解答一个问题：49 个种子节点不是孤立标签，而是一张有方向、有层级、有依赖的知识网络。_

---

## 1. 种子库总览

| 层级 | 节点数 | 本质 | 类比 |
|---|---|---|---|
| **L0 Meta** | 6 | 系统如何思考 | 操作系统的调度策略 |
| **L1 Axioms** | 12 | 数学的根基 | CPU 指令集架构 |
| **L2 Math** | 10 | 数学的工具箱 | 编程语言的标准库 |
| **L3 Methods** | 13 | 算法的核心范式 | 设计模式 |
| **L4 Physics** | 8 | 物理的约束条件 | 硬件的物理极限 |

**核心原则：** 上层依赖下层，下层不依赖上层。L1 不 care L3 的具体算法；L3 的所有算法最终必须回到 L2 的数学工具，再追溯到 L1 的公理。

---

## 2. 纵向依赖链

### L0 → L1：元理论指导公理选择

L0 不是数学，是**关于数学的数学**。它决定什么时候该换一套公理。

```
meta-1 (Double-Loop Learning)
    → 当系统发现 L1-L4 的框架无法解释新论文时，触发框架重构
    → 直接关联: axiom-5 (Gödel不完备) — 为什么框架必然有边界

meta-2 (Association & Causation)
    → 决定 L2 math-4 (概率推断) 和 L4 phys-6 (因果边界) 什么时候该升级关系

meta-3 (Dimensionality Reduction)
    → 指导 L3 method-11 (降维) 和 L2 math-5 (谱分解) 的抽象层级
    → 本质: "信息约束"是 epistemic 选择，不是技术优化

meta-6 (Epistemic Boundary)
    → 直接关联: axiom-5 (Gödel) + axiom-4 (CH独立性)
    → 含义: 有些问题在现有公理下不可判定，必须扩展公理系
```

### L1 → L2：公理生成数学工具

| L1 公理 | 催生哪些 L2 工具 | 为什么 |
|---|---|---|
| axiom-1 (ZFC) | 全部 L2 | 所有数学对象都是集合的构造 |
| axiom-2 (Peano/归纳) | math-2 (动力系统迭代解)、math-3 (优化收敛) | 归纳法是迭代算法的证明引擎 |
| axiom-5 (Gödel不完备) | math-4 (概率 — 处理不可判定性)、math-6 (信息论 — 描述已知与未知) | 形式系统有边界，概率填补间隙 |
| axiom-7 (范畴论) | math-8 (群论/表示论)、math-10 (拓扑) | 范畴论是群论和拓扑的统一语言 |
| axiom-8 (HoTT) | math-10 (拓扑)、math-8 (代数结构) | 类型 = 空间，等价 = 路径 |
| axiom-9 (Turing) | math-3 (优化复杂度)、math-7 (图论算法) | 可计算性决定哪些优化问题可解 |
| axiom-10 (Noether) | math-9 (变分法)、math-8 (群论) | 对称性 → 守恒律 → 变分原理 |
| axiom-11 (线性化) | math-2 (动力系统局部分析) | Taylor 展开是局部分析的根基 |
| axiom-12 (守恒) | math-9 (变分法)、math-2 (ODE能量分析) | Lyapunov 函数来自能量守恒 |

### L2 → L3：数学工具支撑算法范式

| L2 数学工具 | 支撑哪些 L3 范式 | 典型链路 |
|---|---|---|
| math-1 (函数逼近) | method-9 (数据驱动) | 神经网络 = 选择基函数 + 优化系数 |
| math-2 (动力系统/ODE) | method-4 (自适应更新)、method-6 (反馈)、method-8 (仿真) | 反馈控制 = ODE + 稳定性分析 |
| math-3 (优化) | method-1 (搜索)、method-3 (DP)、method-4 (自适应) | 梯度下降 = 优化理论的一阶方法 |
| math-4 (概率/推断) | method-5 (随机化)、method-9 (数据驱动)、method-7 (前馈预测) | Kalman 滤波 = 贝叶斯更新 |
| math-5 (谱分析) | method-10 (谱分解)、method-11 (PCA降维) | FFT = 正交基展开；PCA = 特征值筛选 |
| math-6 (信息论) | method-11 (降维)、meta-3 (信息约束) | MDL 原则 = 模型复杂度 vs 拟合度 |
| math-7 (图论) | method-1 (搜索)、method-12 (并行) | A* = 图上的启发式搜索 |
| math-9 (变分法) | method-3 (DP/Bellman)、method-13 (多尺度) | Bellman 方程 = 离散的 Euler-Lagrange |
| math-10 (拓扑) | method-11 (TDA降维)、axiom-8 (HoTT) | 持久同调 = 拓扑不变量的多尺度提取 |

### L3 → L4：算法范式落地到物理约束

| L3 范式 | 受哪些 L4 物理约束 | 约束表现 |
|---|---|---|
| method-4 (自适应更新) | phys-5 (噪声下限)、phys-6 (延迟) | 步长不能小于噪声 floor；更新率受因果延迟限制 |
| method-6 (反馈) | phys-1 (振荡/共振)、phys-6 (延迟) | 高增益 → 相位裕度不足 → 振荡发散 |
| method-7 (前馈) | phys-6 (因果边界)、phys-5 (SNR) | 参考信号必须因果先行；预测精度受噪声限制 |
| method-8 (仿真) | phys-6 (延迟/传播)、phys-2 (热扩散尺度) | 时间步长受 CFL 条件约束 |
| method-9 (数据驱动) | phys-5 (噪声)、axiom-5 (Gödel) | 数据有噪 + 模型不可完全确定 |
| method-10 (谱分解) | phys-1 (模态耦合) | 非线性模态耦合 → 谱分解失效 |
| method-12 (并行) | phys-6 (光速延迟)、phys-3 (EM干扰) | Amdahl 极限 + 通信延迟 + 串扰 |
| method-13 (多尺度) | phys-7 (相变/临界)、phys-2 (热扩散) | 尺度分离失效时粗粒化模型崩溃 |

---

## 3. 横向关联矩阵（跨层核心链路）

以下 10 条链路是论文→种子映射时最常被触发的：

### 链路 1：反馈控制
```
method-6 (Feedback)
    → math-2 (动力系统: ODE + 稳定性)
    → math-8 (群论: 对称性分解)
    → phys-1 (机械振荡: 阻尼与共振)
    → phys-6 (因果边界: 延迟导致相位滞后)
    → axiom-10 (Noether: 对称性 → 守恒 → 稳定判据)
```

### 链路 2：自适应算法
```
method-4 (Adaptive Update)
    → math-1 (函数逼近: 参数化表示)
    → math-3 (优化: 梯度下降收敛性)
    → math-4 (概率: 估计误差协方差)
    → phys-5 (噪声: 步长 vs 噪声 floor 的权衡)
    → axiom-11 (线性化: 局部收敛保证)
```

### 链路 3：前馈补偿
```
method-7 (Feedforward)
    → math-2 (动力系统: 预测需要模型)
    → math-4 (概率: Wiener 滤波 / 因果预测)
    → phys-6 (因果边界: 参考信号必须先于扰动)
    → meta-2 (因果理论: 前馈是干预验证的一种)
```

### 链路 4：谱分解 / 模态分析
```
method-10 (Spectral)
    → math-5 (谱分析: Fourier / SVD / 特征值)
    → math-8 (群论: 表示论 = 对称性的谱分解)
    → phys-1 (振荡: 模态 = 特征值问题)
    → phys-7 (相变: 临界 = 特征值穿越虚轴)
```

### 链路 5：降维与粗粒化
```
method-11 (Dimensionality Reduction)
    → math-5 (SVD/PCA: 线性子空间投影)
    → math-6 (信息论: 保留多少信息？)
    → math-10 (拓扑: 持久同调识别拓扑特征)
    → meta-3 (信息约束: 降维是 epistemic 选择)
    → phys-4 (材料: 有效介质理论 = 粗粒化)
```

### 链路 6：多尺度耦合
```
method-13 (Multiscale)
    → math-2 (ODE/PDE: 不同尺度对应不同方程)
    → math-9 (变分法: 均质化 = 渐近展开)
    → phys-7 (相变: 临界现象 = 多尺度涨落)
    → meta-5 (层级与涌现: 粗粒化模型的涌现性质)
```

### 链路 7：搜索与优化
```
method-1 (Search)
    → math-3 (优化: 凸/非凸、局部/全局)
    → math-7 (图论: 搜索空间 = 图)
    → axiom-9 (Turing: 可计算性边界)
    → meta-6 (认知边界: 有些问题不可解)
```

### 链路 8：数据驱动映射
```
method-9 (Data-Driven)
    → math-1 (逼近: 通用逼近定理)
    → math-4 (概率: 泛化误差 = 偏差-方差)
    → math-6 (信息: MDL = 模型选择)
    → axiom-5 (Gödel: 数据永远不足以上完整个理论)
    → phys-5 (噪声: 信号提取的物理极限)
```

### 链路 9：动态规划与预测
```
method-3 (Dynamic Programming)
    → math-3 (优化: Bellman = 离散变分)
    → math-2 (动力系统: 序列决策 = 轨迹优化)
    → math-9 (变分法: Hamilton-Jacobi-Bellman)
    → phys-6 (延迟: 预测地平线受因果限制)
```

### 链路 10：随机化与采样
```
method-5 (Randomization)
    → math-4 (概率: 蒙特卡洛 = 大数定律)
    → phys-5 (噪声: 物理噪声 vs 算法随机性)
    → axiom-5 (Gödel: 随机化绕过确定性不可判定)
    → meta-2 (因果: 随机对照试验 = 干预验证)
```

---

## 4. 论文 → 种子映射指南

当 Phase F 提取一篇论文的结构时，按以下顺序匹配：

### Step 1: 识别核心范式（L3）
问：这篇论文的**核心操作**是什么？

| 论文描述关键词 | 匹配 L3 范式 |
|---|---|
| "更新权重/参数/增益" | method-4 (Adaptive Update) |
| "误差驱动/闭环/反馈" | method-6 (Feedback) |
| "预测/前馈/参考信号" | method-7 (Feedforward) |
| "搜索/优化/最小化" | method-1 (Search) |
| "分解/模态/频域/PCA" | method-10 (Spectral) |
| "降维/压缩/特征提取" | method-11 (Dimensionality Reduction) |
| "多尺度/多物理场/耦合" | method-13 (Multiscale) |
| "并行/GPU/分布式" | method-12 (Parallelism) |
| "随机/蒙特卡洛/采样" | method-5 (Randomization) |
| "递归/分治/子问题" | method-2 (Recursion) |
| "序列决策/值迭代/MPC" | method-3 (Dynamic Programming) |
| "仿真/数字孪生/预测" | method-8 (Simulation) |
| "数据驱动/神经网络/训练" | method-9 (Data-Driven) |

### Step 2: 追溯数学基底（L2）
问：这个范式用到了**什么数学工具**？

如果论文用了梯度下降 → 链接到 math-3 (Optimization)  
如果论文用了 Fourier/SVD → 链接到 math-5 (Spectral)  
如果论文用了概率模型 → 链接到 math-4 (Probability)  
如果论文讨论收敛性 → 链接到 math-2 (Dynamical Systems)

### Step 3: 锚定物理约束（L4）
问：这个系统受**什么物理限制**？

传感器噪声 → phys-5  
通信/计算延迟 → phys-6  
机械共振 → phys-1  
热漂移 → phys-2  
材料非线性 → phys-4

### Step 4: 检验元理论触发（L0）
问：这篇论文是否挑战了现有框架？

如果论文指出之前的方法在某种条件下完全失效 → 可能触发 meta-1 (Double-Loop)  
如果论文建立了新的因果链条（不只是相关）→ 升级 meta-2 (Association→Causation)  
如果论文大幅压缩了模型复杂度 → 关联 meta-3 (Dimensionality Reduction)

---

## 5. 完整示例：FxLMS 论文的映射

**论文：** "Latent FxLMS for Active Noise Control" (2507.03854)

### 5.1 内容扫描
- **方法**：前馈自适应控制，用参考信号预测并抵消噪声
- **数学**：LMS 梯度下降 + 次级路径传递函数估计
- **物理**：声学传播、扬声器-麦克风传递函数
- **创新**：引入 latent 表示（降维）来提升收敛性

### 5.2 种子映射

```
L3 核心范式:
├── method-7 (Feedforward) — 论文是前馈架构
├── method-4 (Adaptive Update) — LMS 权重更新
└── method-11 (Dimensionality Reduction) — latent 表示 = 降维

L2 数学工具:
├── math-2 (动力系统) — 收敛性分析
├── math-3 (优化) — 梯度下降
├── math-4 (概率) — 估计误差分析
├── math-5 (谱分析) — 窄带谐波处理
└── math-1 (函数逼近) — 用 FIR 滤波器逼近次级路径

L1 数学公理:
├── axiom-11 (线性化) — 小步长假设下的局部线性
├── axiom-12 (守恒) — 能量不灭，噪声能量被转化为控制能量
└── axiom-5 (Gödel) — 次级路径估计永远不完全准确

L4 物理约束:
├── phys-1 (振荡) — 声学驻波/模态
├── phys-5 (噪声) — 传感器噪声限制性能
├── phys-3 (电磁) — 扬声器驱动 (压电/电磁)
└── phys-6 (延迟) —  causality: 参考信号必须先于误差

L0 元理论:
└── meta-3 (降维约束) — latent 表示是信息约束的 epistemic 选择
```

### 5.3 生成的图谱子图

```
[Paper: 2507.03854]
    │─uses→ method-7 (Feedforward)
    │─uses→ method-4 (Adaptive Update)
    │─uses→ method-11 (Dimensionality Reduction)
    │
    │─based_on→ math-2 (动力系统)
    │─based_on→ math-3 (优化)
    │─based_on→ math-5 (谱分析)
    │
    │─constrained_by→ phys-5 (噪声下限)
    │─constrained_by→ phys-6 (因果延迟)
    │
    └─epistemic_choice→ meta-3 (降维约束)
```

### 5.4 可生成的关联洞察

1. **近迁移**：与 method-7 相关的其他论文（其他前馈控制器）
2. **数学链路**：向上追溯至 math-3 → math-9 (变分法) → axiom-10 (Noether)
3. **物理链路**：phys-6 (延迟) 是 feedback 和 feedforward 的共同瓶颈
4. **双循环触发**：如果 latent 表示显著改变了收敛理论框架，可能触发 meta-1

---

## 6. 使用这份指南

### 对人
阅读时先掌握**3 条核心链路**：
1. 反馈控制链路 (method-6 → math-2 → phys-1)
2. 自适应更新链路 (method-4 → math-3 → phys-5)
3. 降维压缩链路 (method-11 → math-5 → meta-3)

其余链路按需查阅。

### 对系统
Phase F 的 LLM prompt 应把这份指南作为 few-shot context，指导 LLM 做论文→种子的映射。提示词核心：

> "识别论文的核心范式（L3），然后向上追溯数学工具（L2），向下锚定物理约束（L4）。如果论文挑战了现有框架，标记元理论触发（L0）。"

---

## 附录：快速索引表

| 想知道... | 看哪层 |
|---|---|
| 这篇论文用的方法属于什么范式？ | L3 |
| 这个范式的数学原理是什么？ | L2 |
| 数学工具的根基在哪？ | L1 |
| 系统有什么物理限制？ | L4 |
| 这篇论文是否颠覆现有认知？ | L0 |
| 两篇论文是否同构？ | 比较它们的 L3 范式 + L2 数学 |
| 为什么算法在这里失效？ | 检查 L4 物理约束是否被违反 |
