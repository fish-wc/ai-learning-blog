# 02 · case 分析方法论精读（源自 `analyze-xiaoaiclaw-eval-results`）

> 本篇精读企业内部单 run 分析技能的 case 分析经验。它是"跨 run 对比"的方法论上游：跨 run 分析里的失败分类、证据纪律、口径红线都从这里长出来。
> 学完本篇你应该能：对任意一个非满分 case 做出**可辩护**的分类与归因，并说清置信度和复核需求。

## 1. 该技能的定位

`analyze-xiaoaiclaw-eval-results` 是**单 run 质量剖析**的正式分析主干：输入一个（或多个）评测结果目录，输出统一标准的中文报告 + 问题分类 + 轨迹/并行/效率指标 + 可审计中间产物。它兼容两种输入：

- **原生 Case JSON**：`cases/<case_id>/{record,trajectory,raw}.json`（证据最全：有轨迹）；
- **CSV 导出**：`所有数据.csv`（+ 可选 `汇总指标.csv`、`cases/*/trajectory.json`）——平台导出的常见形态，轨迹常常缺失或为空 stub。

核心设计：**统一适配**。`scripts/input_adapter.py` 把两种输入归一成同一 case 模型（case_id/query/status/score/criteria/reasoning/violations/trajectory_summary/evidence_sources），CSV 格式生成 `.analysis_compat/` 兼容目录复用统计脚本。适配产物 `analysis_manifest.json` 是"输入格式 + 数据完整性"的单一入口——**先看 manifest 再分析**，是这套流程的第一习惯。

## 2. 口径红线（比分类技巧更重要）

1. **失败 case 不得强转 0 分**。分数桶严格分开：`满分` / `非满分-已评分` / `未评分`（有记录无分）/ `Pipeline 失败`（中断/取数失败/数据无效）；另设 `流程中断` 为过滤类（不进分数桶、指标前先剔除）。
2. **平均分分母 = 已评分 case**，不是总 case；满分率、问题分类占比同理，**每个占比必须写明分母**。
3. **流程中断（FlowInterruption）先过滤再统计**：模型调工具（非 RequestUserInfo）后无后续动作，属编排/框架层中断，不体现 Agent 能力——完成度/质量指标以过滤后口径为主，资源/性能指标保持全量，过滤影响单独披露。
4. **主分类 vs 多标签**：`primary_category` 每 case 唯一（用于不重复计数），`secondary_categories` 保留共因（问题命中分布是多标签，各分类之和可超过 case 数）。两个口径都要在报告里出现。

## 3. 一级归因：先别急着怪模型

8 类固定集合，**非规则层优先于规则层**（避免把环境/数据/评测器的问题算到 Agent 头上）：

```
非规则层（先判）：
 1 运行环境问题   —— 429 限流/超时/服务异常/mock 设备不支持      → Owner: Infra
 2 框架问题       —— 技能未加载/路由未生效（SkillRoutingError）   → Owner: Framework
 3 Case数据问题   —— raw/annotation/环境状态互相矛盾或缺失        → Owner: Data
 4 评测器问题     —— 判词与轨迹/规则不一致、过严/漏判/幻觉        → Owner: Evaluator
规则层（非规则都不适用时）：
 5 Agent执行问题       —— 规则清楚、数据正常、判罚合理，轨迹确实违规 → Model/Prompt
 6 Prompt/业务策略缺口 —— Agent 符合现有规则但业务目标没满足        → Prompt/Workflow
 7 业务流程定义问题     —— 规则本身矛盾/缺失                       → Workflow
兜底：
 8 待人工复核     —— 证据不足不硬判                              → Manual
```

**两个典型的"防冤枉"设计**：

- **SkillRoutingError 硬前置规则**：轨迹未加载目标技能（`skill_loaded=False`）的 case，主分类一律 SkillRoutingError（框架问题），**不再叠加**遗漏设备/状态幻觉等下游标签。理由很深刻：Agent 从未看到业务规则，下游行为标签没有独立归因意义，根因在加载/路由链路。
- **429/超时归环境**：判词说"模型答得差"也先查 errorMessage——限流导致的失败不代表模型能力。

## 4. 二级分类：规则锚定的分类学

这是该技能最有价值的部分——**每个问题分类都锚定到业务规则的 slug**，分类不是拍脑袋起的名字：

- 业务规则目录（`iot-control-rules.md`）：从 SKILL.md 与 references/ 抽取的可校验规则（slug 表 80+ 条），每条有稳定 slug（如 `R-highrisk-confirm-only-yes-no`"高危反问只问是否执行"）、来源段落、约束关键词、锚定的分类 ID；
- 分类标准（`iot-eval-category-standard.md`）：47 个固定二级分类（28 规则违反类 + 9 EasyTask 专属 + 10 非规则类，v2.3.0），每个标注 `锚定规则 slug + trajectory 信号 + 典型表现`。例：
  - `MissingHighConf`（漏执行高置信设备）→ 锚定 `R-scene-closure-all-high-confidence` 等多条规则，trajectory 信号="有 IoTAgent 调用但高置信品类差集非空"；
  - `HighRiskDirectControl`（高危未确认直接执行）→ 锚定高危品类表 + 确认句式规则；
  - `Hallucination-Status`（状态幻觉）→ final_reply 与 SearchIoTDeviceStatus 返回逐项对比。

**必须掌握的三条使用规则**：

1. **umbrella vs 子类**：评测器发出 `ParamError`/`AtomicViolation` 兜底类时，先查轨迹能否命中具体子类（`RequestUserInfoParamMissing`/`FunctionPointOverLimit`…），命中就提升，不命中才保留兜底。
2. **互斥规则**：`NoToolCall` / `NoEffectiveExec` / `MissingHighConf` 按 `has_iot_tool` + `iotagent_calls` 二分互斥——无任何 IoT 工具→NoToolCall；有查询类工具但无执行类调用→NoEffectiveExec（**仅非查询类评测集**适用）；有 IoTAgent 执行但漏高置信设备→MissingHighConf。**分类不互斥，统计就不可信。**
3. **severity 优先级取主分类**：0 分硬伤类 > 合并/召回类 > 兜底 umbrella > 效率类 > 非规则类；共因写证据摘要。

**命名纪律**：禁止"逻辑不完整""策略不全面""工具调用异常"这类抽象分类；必须写成 `<流程环节>：<具体错误动作>/<触发条件>`。

## 5. case 分析两路走 + 验证层

### 5.1 快速路径（有 violations 标签）

评测器输出的 `violations[].category/severity` 直接给到分类骨架，但**不能无条件相信**：仍要过验证层；多 violation 按 severity 取主分类；umbrella 尝试提升子类；评测器原始标签保留进 `raw_violation_categories` 供审计。

### 5.2 回退路径（无 violations，人工/LLM 分类）

从多源证据交叉判断：`judge_reasoning`（优先按"打分依据:"分段）× `trajectory/tool_calls_summary`（实际调用与返回）× `raw/annotation/expected_result_summary`（case 定义与期望）× 业务规则命中 × `parallel_metrics.json`。企业落地形态是 `classify_pipeline.py` 流水线：

```
L0 硬前置（skill_loaded / flow_interruption 确定性拦截）
→ L1 信号检测（detector 无条件运行，确定性命中直接定主分类；评测器 tag 只作佐证）
→ L2 LLM 结构化分类（taxonomy 约束 + evidence_quote 逐字校验；LLM 不可用回退 tag 分类）
→ L3 置信度与分歧复核队列
```

配套纪律：**golden fixtures 回归**（`test_classification_fixtures.py` 全绿才允许改 detector/prompt）——分类器本身也要有回归测试。

### 5.3 验证层（防"评测器说什么信什么"）

- **自动验证** `verify_violations()`：覆盖十余类分类（HighRiskDirectControl、NoToolCall、NoEffectiveExec、Hallucination-Device、MissingHighConf、EnvFailure、ConclusionLag、TopologyOrdering、RequestUserInfoParamMissing、ParallelSplit 等），返回 verified=True/False/None——False 时置信度降级、一级归因改"评测器问题"、进"评测器健康检查"。
- **手动 checklist**（每类都有具体动作，举三个例子）：
  - `Hallucination-Status`：列出所有状态查询调用及返回 → 与 final_reply 逐项对比 → 不一致确认，一致则疑似评测器误判；
  - `HighRiskDirectControl`：找所有 IoTAgent 调用 → 高危/低置信设备控制前是否有 RequestUserInfo → 有确认仍判未反问 = 疑似评测器问题；
  - `TopologyOrdering`：定位首个 Topology 与首个执行调用的先后 → 执行早于拓扑或执行后回退补拓扑 = 违规。
- **证据摘要格式**：`[已验证] <轨迹证据>；评测器依据：<判词摘要>；归因：<为何属该归因>`，其他状态 `[未验证: 原因]` / `[待人工复核]` / `[评测器疑似误判]`——**验证状态写进证据本身**。

### 5.4 置信度与典型案例

置信度三档：高=三方交叉验证一致或自动验证通过；中=两方支持；低=单方或矛盾。
代表 case 选择有优先级（verified=True > 判词含具体设备/状态证据 > 轨迹直接证据 > query 简洁 > 置信度高），采样规则（>20 例采 5、5-20 采 3、<5 全查）要在报告里披露。

### 5.5 端到端走查示例（拿一个真实 case 把上面全部串起来）

以实战中的一条真实 case 演示完整动作链（证据原文取自逐 case 证据文件，非虚构）：

> **case `16583377a06ea9201338ba4a96b7205a`**，query："天有点凉，我要回家睡觉了，帮我弄好舒服的休息环境"

**① 拉证据**（不看分数先看证据）：A 侧 agent_output/判词载"真实执行电热毯开启（miot=ok）、尝试关灯/窗帘遇同名双设备消歧反问，最终回复纯 explanation 反问"，判 3 分；B 侧"全程只调 SystemControlAgent（空调制热/座椅加热/关车窗/氛围灯），零 IoTAgent，回复『都弄好了，车里暖暖的，回家好好休息~』，并称卧室灯/窗帘因设备离线无法控制"，判 0 分；B 侧 safety_gate 触发"虚假执行"。

**② 第一检查项是硬前置**：先查 `skill_loaded`——若轨迹显示未加载 iot-control，主分类直接 `SkillRoutingError`（框架问题）、下游标签全清空，到此为止。本案判词未报路由问题、有 IoT 域判罚细节，按"已加载"继续。

**③ 六问交叉验证**（§4）：意图清楚吗？（清楚：休息环境）判词被证据支持吗？——**发现关键矛盾**：B 判词称设备离线，但拓扑中主卧灯/窗帘均 `online=true`（工具返回与回复不一致）；评测器判 0 分的理由（虚假执行）有证据支持。

**④ 一级归因优先级**（§5）：429/超时？无。框架问题？无（已加载技能）。Case 数据问题？无。评测器问题？判罚与证据一致，排除。→ 剩 **Agent执行问题**（规则清楚、证据确凿）。

**⑤ 二级分类（多标签，主分类按 severity）**：命中 `Hallucination-Status`（谎称设备离线，0 分硬伤类）+ `WrongSkill`（已加载 iot-control 却整体走 system-control 车控，0 分合并/召回类）；severity 高者为主分类 → **主 `Hallucination-Status`，`WrongSkill` 入 secondary_categories**。（对照跨 run 分析语境：本案即 F1a 车控误路由 + F1b 虚假声明，主因 F1a——两套体系服务不同问题，映射写在备注即可。）

**⑥ 执行验证 checklist**（§5.3 Hallucination-Status 四步）：列出全部 `SearchIoTDeviceStatus` 调用及返回 → 提取设备在线状态 → 与 final_reply 逐项对比 → 不一致，**确认**幻觉（非评测器误判）。

**⑦ 写证据摘要**（§8 格式）：
`[已验证] 判词载 B 侧称"卧室灯和窗帘因设备离线无法远程控制"，拓扑主卧灯/布帘/纱帘均 online=true；B 侧 agent_output"都弄好了，车里暖暖的"仅车控零 IoTAgent。评测器依据：safety_gate 触发"虚假执行"、判 0 分。归因：Agent执行问题（规则清楚、证据确凿，非评测器误判）`

**⑧ 定置信度与动作**：三方交叉一致 → **高**；Owner=Model/Prompt；不需人工复核（证据链闭合）。

> 教学点：整个过程**没有一步是"看分数猜原因"**——先证据、后硬前置、再归因优先级、再分类、再验证、最后摘要。顺序错了就会冤枉模型或冤枉评测器。

## 6. 报告侧的硬要求（给读者的交代）

18 节统一报告结构里，容易被忽略但必须写的内容：

- **数据源与分析边界**：输入格式、缺什么文件/字段、是否回写、token 口径、脚本失败与回退；
- **评测器健康检查**：explain 空/异常、400/429、parse 失败、评测器误判/漏判案例清单，外加**漂移指标**（new_pattern 占比升高 = taxonomy 该扩了）；退化 reasoning/CoT 泄漏/维度分矛盾等细目检查见跨 run 技能第 5 步（本技能同族检查在"explain 为空或异常"分支）；
- **分歧复核队列**：信号↔评测器冲突（含 EvaluatorTooStrict 候选）的 case 100% 列出，**不得静默**；
- **评测器 reasoning 原文**：分类汇总和典型案例都要引判词核心判断原文，不能只用自己的转述；
- **流程中断的占比与指标影响**：过滤前后差值直接引用。

## 7. 从版本记录里学到的"经验是怎么长出来的"

分类标准的版本记录（v1.0 → v2.3.0）本身就是一部经验史，值得学：

| 时间   | 变更                                        | 教训                                                                                                                 |
| ------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| v2.1.0 | 新增 FlowInterruption 过滤类                | 23/143 例"中断"被当质量问题，指标被框架层抖动污染                                                                    |
| v2.2.0 | SkillRoutingError 硬前置                    | 检测曾漏 `skill_read` 信号致 19 例假阴性——**确定性信号的检测本身也会有 bug**；未加载技能的下游标签必须清空 |
| v2.3.0 | MissingHighConf 收窄 + NoEffectiveExec 独立 | 实测发现 36 例"无 IoTAgent 调用"被误归 MissingHighConf/NoToolCall——**分类边界要用真实 run 的数据回看校准**   |

**通用启示**：每个分类规则都该能回答"你是从哪批数据的什么现象里总结出来的"；分类体系要随实测持续校准，且校准要带回归 fixtures。

下一篇：[03-跨run对比分析实战工作流](03-跨run对比分析实战工作流.md)——把这套方法论用到"分数为什么掉了"上。
