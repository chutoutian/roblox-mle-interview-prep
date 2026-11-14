# 04 · 面试流程与答题节奏

完成三个 CV design practice 后，用这一页把知识组织成一场可以交流、可以追问的面试回答。中文解释用于复习；英文模板用于练习口述。

**核心主线：业务目标决定标签 → 标签决定任务与 loss → 数据和预算决定模型 → 指标检验目标 → 服务约束限制可用方案。**

## 1. 与 Roblox preview 的对应关系

根据你提供的 **[Full Time Only] ML Modeling Knowledge Interview** preview，这轮可以理解为以模型设计、理论和训练为核心的端到端 ML system design。官方给出的时间是 **5 分钟介绍 + 35 分钟技术讨论 + 5 分钟 Q&A**，并明确推荐以下流程：

```text
Problem Formulation
  → Data & Feature Engineering
  → Model Selection
  → Loss & Optimization
  → Offline/Online Evaluation
  → Latency & Scalability
```

| Preview 明确要求的内容 | 三个练习中的准备 |
| --- | --- |
| ML Theory & Model Design | CNN / ViT、pooling、fusion、sigmoid / softmax、embedding 与架构取舍 |
| Data Translation & Feature Engineering | 业务目标、样本单位、标签规范、缺失标签、输入处理、业务代理指标 |
| Training & Optimization | loss、optimizer、冻结与微调、偏差、过拟合、负采样或数据增强 |
| Trade-off Evaluation | 离线指标、延迟、内存、扩展性，以及更复杂模型是否值得 |

我们的七步把官方的 **Model Selection** 拆成 **Baseline + Model Design**，因此与 preview 的主线匹配。七步的具体时间、三个题目、假设规模和 21 个 follow-up 都是练习安排，不是官方规定的题单或评分表。实际讨论可能由面试官选择重点，也可能穿插理论或伪代码追问。

预训练模型是合理起点。Preview 同时强调 theoretical depth 和 beyond standard out-of-the-box approaches，所以需要继续说明：怎样构造业务数据、怎样训练或适配、为什么选择这个目标、出现什么错误后如何改进。CV / multimodal 是本套练习的准备方向；实际分配到的 vertical 以 recruiter 的信息为准。

## 2. 35 分钟技术讨论的节奏

下面的时间从技术讨论开始计算，不包含前面的介绍和最后的 Q&A；**包含面试官追问，不是 35 分钟连续独白。**

| 技术时间 | 阶段 | 这一段必须交代的结果 |
| --- | --- | --- |
| 0–3 分钟 | Problem formulation | 任务、样本、输入输出、重要约束 |
| 3–8 分钟 | Data & labels | 标签来源、处理、缺失与偏差、防泄漏切分 |
| 8–10 分钟 | Baseline | 一个可训练或直接评估的完整起点 |
| 10–16 分钟 | Model design | 输入如何变成预测，选择理由与替代方案 |
| 16–23 分钟 | Loss & training | loss、batch、更新参数、优化与泛化 |
| 23–29 分钟 | Evaluation | 主要指标、阈值、分组表现、线上验证 |
| 29–35 分钟 | Serving & trade-offs | 一次预测、预计算、延迟、内存与更新 |

这是建议预算。原来三个 walkthrough 的时间按教学内容分组；实际口述可以缩短 baseline 介绍，把时间留给理论深挖和追问。

拿到题目后，先用约 30 秒说出整体方向。介绍复杂改进之前，先让面试官看见一个能训练、能评估、能服务的 baseline。每段先讲核心决定和理由，再根据追问展开公式、shape 或替代方案。被追问占用时间后，压缩次要细节，仍然把评估与服务路径交代完整。

## 3. 开头需要 clarify 什么

**需要先澄清 requirements 和规模，但优先问会改变模型设计的问题。** 题目已经提供的信息直接使用；未给出的数字先询问数量级，必要时提出明确假设，并解释它影响哪个决定。

| 顺序 | 澄清的问题 | 为什么需要知道 |
| --- | --- | --- |
| 1 | 预测什么，结果用来做什么？ | 决定任务、输出、标签和 loss |
| 2 | 预测时有哪些输入，已有多少可靠标签？ | 决定特征、预训练和微调策略 |
| 3 | 什么错误最贵，怎样算成功？ | 决定指标、阈值、拒答和人工审核 |
| 4 | 在线还是异步，规模、延迟、更新要求是什么？ | 决定模型预算、预计算和检索方式 |

**Moderation：** 多个违规类别能否同时成立？只判断是否存在，还是需要定位？输出用于自动限制还是人工审核？严重违规的漏报和正常内容的误限制，哪个约束更严格？

**Genre：** 一个游戏能否有多个 genre？是否要求刚发布就给标签？当时有哪些截图和文字？类别定义和标注规范是否已确定？

**Retrieval：** 找完全相同的资产，还是满足需求的替代品？支持文字和参考图吗？一个 query 能否有多个正确结果？目录多大、展示多少条、延迟和新资产可搜索时间有什么要求？

规模询问要服务于决策：目录大小决定是否需要 ANN；输入分辨率影响小目标识别和计算；异步发布标签允许离线编码。先明确任务，再根据需要展开容量估算。

**英文开场：**

> “Before choosing a model, I’d clarify the prediction target, what inputs and labels are available at prediction time, and the main error costs. I’d also confirm whether inference is online or asynchronous and the latency and scale constraints. Then I’ll walk through data, a baseline, model and training choices, evaluation, and serving.”

## 4. 每一步讲什么

### 4.1 Problem formulation：把业务目标变成学习问题

说明一个样本是什么、预测时的输入、目标标签、输出怎样影响业务。明确分类还是检索、单标签还是多标签、是否允许不确定结果，并连接误报与漏报的代价。

> “One example is ___. At prediction time, we have ___. The target is ___. The output will be used to ___.”

例如 genre：一个样本是某个游戏的发布版本，输入是当时可用的截图和描述，输出是多个 genre 分数，供发现和筛选功能使用。如果 genre 可以同时成立，后面的模型和 loss 都要遵守这个定义。

### 4.2 Data：先解释标签，再解释处理与切分

按 **标签如何定义 → 如何获得 → 噪声或缺失 → 如何切分** 展开。区分人工标签、weak labels、举报、点击；解释这些来源的偏差。未知标签不能直接当负例。

说明预测时可用的信息，避免未来行为泄漏。按资产复制家族、游戏版本或 creator 等相关组切分，并根据目标加入未来时间或新 creator 测试。说明 resize、crop、normalization、多视角、文字和缺失模态的处理；数据增强必须保留标签语义。

> “I’d define an annotation protocol, distinguish unknown labels from negatives, and split related content before training. Evaluation would include a representative sample and separate difficult slices.”

### 4.3 Baseline：给出一个完整、便宜、可验证的起点

讲清 **输入 → 表示 → 输出 → 指标**。Moderation 可从冻结视觉 encoder 加分类 head 开始；genre 可比较简单文字模型和冻结图像特征；retrieval 可从兼容的预训练文字/图像 embedding 开始。

接着说 baseline 可能在哪些地方失败，以及什么证据会推动升级。例如材质区分失败后，检查渲染是否提供材质证据，再考虑 domain adaptation。先修数据或管线错误，再判断模型是否需要改变。

### 4.4 Model design：沿数据流解释，而不是只报模型名

```text
输入 → encoder → pooling / fusion → head 或相似度 → 输出
```

说明单图、多图或多模态如何组合，关键 representation 和最终输出是什么。必要时写关键 shape，例如多视角输入如何变成每个 asset 的一个向量。架构细节围绕当前瓶颈展开。

每个重要选择使用同一个表达方式：**选择 X，因为约束 Y；代价是 Z；用实验 W 判断是否值得。**

> “I’d start with mean pooling because it is simple and cheap. If error analysis shows that distinctive views are diluted, I’d compare attention pooling on held-out relevance and serving cost.”

### 4.5 Loss & training：准备好解释一个核心公式

说清 loss 为什么匹配任务，一个 batch 包含什么，哪些参数冻结或更新，optimizer 和 learning rate 如何选择，以及如何处理不平衡、标签噪声、过拟合与增强。

| 任务 | 核心训练选择 | 准备解释的追问 |
| --- | --- | --- |
| Moderation / genre | 多标签 sigmoid + masked BCE | 为什么不用 softmax，mask 怎样避免把未知当负例，类别权重有什么代价 |
| Retrieval | Query–asset contrastive objective | 正负例怎样构造，如何处理多正例和假负例，normalization 与 temperature 的作用 |

有限标签下可先冻结 encoder，训练 head / projection，再验证部分解冻或轻量适配。AdamW 是可解释的起点，是否更好需要验证。用 validation 选择 checkpoint、学习率和正则化，保留独立 test。

对于 query → asset 服务目标，单向 contrastive loss 已能更新两边可训练 encoder；反向 loss 是可比较的辅助目标，需要正确处理两个方向的有效替代匹配。

### 4.6 Evaluation：让指标对应具体业务问题

选择 **一个主要离线目标 + 必要约束 + 线上业务结果**，解释为什么适合任务。

| 任务 | 离线重点 | 线上重点 |
| --- | --- | --- |
| Moderation | 在误限制和审核容量约束下的严重违规 recall；逐类表现 | 审核与随机审计得到的违规漏检、误限制、申诉和处理时效 |
| Genre | 每类 precision / recall、macro 指标、阈值与长尾覆盖 | 使用标签后是否改善筛选或发现，错误标签反馈 |
| Retrieval | 候选 Recall@K、最终 NDCG@20；ANN 与精确搜索另行比较 | 成功选择、插入或使用资产，放弃率和延迟 |

模型分数用于排序，人工标签用于评价；上述检索排序指标不要求把相似度映射到 0–3。阈值和模型在 validation 上选择，test 用于最终确认。按文字/图片、新 creator、稀有类别等切片，避免平均值掩盖退步。

离线提升需要在线验证。可能失败的原因包括数据不代表实际流量、标签标准与可用性不一致、历史曝光偏差、延迟变大、线上预处理或索引版本差异，以及重复结果太多。点击增长也不自动等于成功使用增长。

### 4.7 Serving：走一次真实预测，再讲更新

说明 **预处理 → 模型执行 → 后处理或检索 → 业务输出**，并指出哪些工作可以预计算。

Moderation 需要及时产生分数并路由到相应动作；genre 可以在发布或更新后异步计算并保存标签；retrieval 可以预计算 asset 向量，在线编码 query 后检索与重排。

只估算关键成本，并解释数字来源。例如检索向量存储量为 **资产数 × 每个资产的向量数 × 维度 × 每维字节数**；200 万资产、每个一个 512 维 float16 向量，原始向量约 2.048 GB，不含 ANN 结构与元数据。

解释新内容、删除、更新和回滚。模型改变后检查缓存是否有效，query encoder 是否仍与 asset 向量兼容。测量完整请求路径的 p95 / p99；分阶段 p95 不能直接相加当作端到端 p95。

## 5. 三个题的主线如何切换

| 练习 | 任务与输入 | 最值得深入的设计问题 |
| --- | --- | --- |
| [01 · Visual moderation](01_visual_moderation_detailed.md) | 上传图像 → 多个违规类别分数 → 动作 | 标签可靠性、稀有违规、误报漏报成本、阈值与审核容量 |
| [02 · Game genre](02_game_genre_detailed.md) | 游戏版本的截图和文字 → 多标签 genre | 多图聚合、多模态融合、冷启动、缺失标签与模态 |
| [03 · Multimodal retrieval](03_multimodal_asset_retrieval_detailed.md) | 文字或参考图 → 相关资产排序 | 共同表示空间、正负例、多正例、召回与重排、ANN 和索引兼容 |

七步结构相同，时间应围绕具体题目的难点调整。Follow-up 是准备知识的参考库，实际面试无需按 Q1–Q21 顺序主动讲完。

## 6. 如何应对追问与时间不足

先回答追问的结论，再给必要的机制、公式或例子，最后连接回设计选择。例如解释温度后，说明会在 held-out retrieval 上调参并观察噪声敏感性，而不是停留在 softmax 公式。

如果面试官深入某个模块，顺着讨论并保留其余路径的简短交代。可以用过渡句恢复整体结构：

> “Given that training setup, I’d validate the model using ___. For serving, the main constraint is ___, so I’d ___.”

重要选择可以用实验判定：同一验证集、明确变量、质量与成本一起比较。例如 text-only / image-only / fusion，mean / attention pooling，冻结 / 部分微调。更大的模型、更多视角或更高分辨率都需要收益证据。

最后 5 分钟留给 Q&A，可以询问团队如何定义模型成功、当前数据或标注的主要瓶颈、研究模型怎样进入生产及接受评估。

## 7. 面试前一分钟速查

- **任务：** 我能否用一句话定义一个样本、输入、目标和业务动作？
- **澄清：** 哪些缺失信息会改变标签、模型或服务方式？
- **数据：** 标签哪里来，未知和负例如何区分，怎样防泄漏？
- **Baseline：** 是否已有一个完整、可评估的起点？
- **模型：** 输入如何变成输出，为什么选这套结构？
- **训练：** 能否解释核心 loss、batch、更新参数和泛化策略？
- **评估：** 主指标对应什么用户问题，阈值在哪里选，如何在线验证？
- **服务：** 哪些能预计算，主要延迟和内存在哪里，模型与内容怎样更新？

**每项决定都说明 why，并给出可验证的改进条件。** 来源是本次对话提供的 Roblox preview 和三个本地练习；本页的时间预算与口述模板是准备建议，不代表新增官方要求。
