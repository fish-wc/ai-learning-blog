# 08 · [二开] SaaS 架构与部署运维

> 目标：读完本章，你了解公司侧（小爱）在这个代码基座上加了什么：多租户 SaaS、K8s 部署、云端记忆、设备场景 skills/agents、以及与上游同步的纪律。
> 本章是指路图 + 心智模型；实操细节全部指向 `xiaoaiclaw-ops/` 与 `docs/xiaoai/` 的权威文档（那些文档很长，按需下钻）。

## 0. 一句话定位

- **上游** = 开源个人 AI 助手（Gateway/Channel/Agent 主体在 `src/`、`extensions/`）。
- **小米二开** = 在 Gateway 之上叠加 **SaaS 多租户扩展层**（`src/saas/`，Java 侧 ai-claw-proxy 接入），配套 **K8s/Matrix 部署体系**（`xiaoaiclaw-ops/`）、**云端记忆插件**（`plugins-local/memory-cloud`）、**小爱设备场景 skills/agents**（`xiaoaiclaw-ops/skills/`、`xiaoaiclaw-ops/xiaoai-agents/`）。
- 物理隔离原则：`xiaoaiclaw-ops/` 与上游不冲突；根 `AGENTS.md` rebase 时冲突属预期，用我方版本覆盖。

## 1. SaaS 多租户架构

权威文档：`xiaoaiclaw-ops/saas-architecture.md`（617 行，"代码为准"）、`docs/xiaoai/saas-multi-user-plan.md`、`docs/xiaoai/saas-gateway-proxy-gaps.md`。

### 1.1 核心设计

- **多用户共享同一 Gateway 进程，数据完全隔离**；总开关 `saas.enabled`（`src/saas/config.ts` 的 `isSaasEnabled()`）。
- **不侵入 get-reply 核心路径**：靠 `ctx.SaasUserId` 注入做路径切换（`src/auto-reply/reply/get-reply.ts`）。
- **三条 RPC 路径**：
  - `chat.send` —— TUI/WebUI，管理员本地 workspace；
  - `chat.send.saas` —— ai-claw-proxy（独立 Java 服务，`com.xiaomi.ai.claw.proxy`）专用，必传 `xiaoaiMaskUid`；
  - `channel.send.saas` —— 飞书，含 owner/visitor 角色限制。
- **SaaS TUI**（`./manage.sh saas-tui --mask-uid <uid>`）= 用 TUI 壳模拟 proxy 调用，**testing-only**（双端 `XIAOAICLAW_SAAS_TUI_ACCEPT` 门控）。

### 1.2 per-user 隔离

- 目录：`{saasWorkspaceDir}/{userId}/agents/{agentId}/…`（memory/skills/cron/logs/sessions），见 `src/saas/workspace.ts`（`resolveSaasAgentWorkspace()`）。
- SessionKey 格式：`agent:{agentId}:{channel}:{groupId}:u-{userId}`（`src/saas/session-key.ts`）。
- 并发：`SaasUserQueue`（`{userId}:{groupId}` 串行）+ per-user lane 并发上限（`src/saas/user-queue.ts`）。
- exec 安全双层：`saas.exec.blockedCommands`（命令名拦截）+ `saas.workspaceGuard`（strict 路径白名单/blockPaths，见 `src/agents/pi-tools.ts`）。
- Agent 前缀协议：消息开头 `agent=xxx;` 选 agent；`create new agent <name>` 创建。

### 1.3 `src/saas/` 模块清单（以代码为准）

config / workspace / middleware / session-key / user-queue / memory / system-prompt / agent-template / preset-agent-templates / main-agent-template-cache / **device-model-routing** / **test-traffic**（含 `resolveVipModelOverride`）/ tool-audit-log / diagnostic-allowlist / context-readable / agent-md-refresh。

> 注意：`saas-architecture.md` 的模块表略滞后（文档自己声明"出入以代码为准"），看代码不要只信文档表格。

### 1.4 部署形态演进：单 Pod per-user

- 早期：多用户共享 Gateway Pod + 共享 JuiceFS PVC。
- 现架构（`docs/xiaoai/session-migration-single-pod-per-user.md`）：**每用户独占一个 Pod**，JuiceFS 卷用 **subPath 硬隔离**（Pod 只能看到本用户数据）；迁移时靠 Gravitino fileset (gvfs) 绕过 mount 层隔离读全量。
- 灰度路由：双 Nacos / saas41 见 `docs/xiaoai/feature-proxy-dual-nacos-saas41-routing.md`、`session-migration-sop.md` §4.4。

### 1.5 设备场景与模型路由

- **设备形态**：Phone / SoundBox（音箱）/ TV / Automotive（座舱）/ Wearables-Headless。每个 xiaoai agent 有分设备 prompt 变体：`xiaoaiclaw-ops/xiaoai-agents/main/AGENTS-{Phone,SoundBox,TV,Automotive}.md`、`IDENTITY-*.md`；skill 同理（如 `xiaoaiclaw-ops/skills/music-search-play/SKILL.SoundBox.md`）。分设备 skill 可见性过滤规范：`xiaoaiclaw-ops/docs/device-specific-skill-loading-tech-standard.md`（appId 优先、deviceType 兜底两级过滤）。
- **按端模型/思考路由**：`src/saas/device-model-routing.ts` + `docs/xiaoai/feature-device-model-routing.md`。按 `deviceType`（如 `Automotive#Large`）**请求级**改写模型与 thinking 档位（`ctx.DeviceRouteOverride`，不写 session 不留残留）；thinking 四级优先级：路由 > 模型 extra_params > 用户 /think > 默认。配置 `saas.deviceModelRouting`（Apollo 热更新）。
- **VIP 模型流量路由**：`src/saas/test-traffic.ts` 的 `resolveVipModelOverride()` + `docs/xiaoai/feature-vip-model-traffic-routing.md`。白名单机制：targetModel 命中 + maskUid 不在 `vipUidPatterns` + 时间窗口内 → 静默把非 VIP 用户重定向到公共集群模型（缓解私部集群压力）。与 `testTrafficRedirect`（测试流量黑名单）相对。

## 2. 部署与运维（Ops Map）

### 2.1 镜像

- `xiaoaiclaw-ops/Dockerfile.base`：Node 22 + Bun + pnpm + Chromium + CJK 字体，推 MICR，tag 规范 `base-YYYYMMDD`（**禁止 `:latest` 作默认引用**，`branch-strategy.md` §7）。
- `xiaoaiclaw-ops/Dockerfile`：FROM 钉基础镜像版本，`USER node`、`--bind lan`；**Extensions 裁剪**：上游 33 个只保留 7 个（feishu / memory-core / memory-lancedb / lobster / llm-task / diffs / diagnostics-otel）—— **合并上游时必须保留此定制**。

### 2.2 K8s / Matrix（生产 CD）

权威文档：`xiaoaiclaw-ops/k8s-deployment.md`（1055 行，非常实操）+ 参考清单 `xiaoaiclaw-ops/k8s/*.yaml`（Matrix 平台是封装的 K8s，YAML 仅作字段对照）+ `xiaoaiclaw-ops/flow-matrix-guide.md`。

- CI = Flow（读 `xiaoaiclaw-ops/Dockerfile`，镜像 tag `$CI_COMMIT_SHORT_SHA`），CD = Matrix。
- 配置注入：`xiaoaiclaw.json` 经 SCM subPath 挂到 `/app/xiaoaiclaw-config/xiaoaiclaw.json` + `XIAOAICLAW_CONFIG_PATH`；`${ENV}` 语法引用 Secret 注入的 API key。
- **实例隔离**：shell wrapper `export XIAOAICLAW_HOME=/workspace/xiaoaiclaw/$POD_NAME`（`XIAOAICLAW_STATE_DIR` 不够，必须 `XIAOAICLAW_HOME`）。
- Session：`session.storeMode=redis`（P4T Redis，`mibrain:claw:` 前缀）+ `session.memoryCloud` hooks；灰度/回滚 SOP（`SESSION_STORE_MODE=filesystem` 回滚）见 k8s-deployment.md §七、`xiaoaiclaw-ops/dev/redis-session-grey-release-selection.md`。
- 日志：stdout（Matrix 采集）+ 内置滚动文件 `/tmp/xiaoaiclaw/xiaoaiclaw-<date>.log`（500MB/24h，`logging.file` 可改）。
- 已踩坑速查：代理变量透传致 LLM 连不上、models.json 缓存、ConfigMap 只读 EROFS、healthz 被 Control UI 拦截（上游 bug，见 upstream-tracker）。

### 2.3 本地开发环境

- 入口：`xiaoaiclaw-ops/local-setup.md`（Ubuntu/WSL2，3 步 Quick Start）；`podman-install.md`、`macos-local-setup.md`、`native-local-setup.md`、`native-overlay-guide.md`（构建加速）、`docker-compose.local.yml`。
- **`xiaoaiclaw-ops/manage.sh` 统一入口（勿 sudo）**：start/up/stop/down/restart/logs/status/fix-perms/onboard/cli/saas-tui/build/build-base/rebuild；macOS/Ubuntu 变体 `manage_*_local.sh`。

### 2.4 日志 / 监控 / 事故 Runbook

| 要找什么 | 去哪 |
|---|---|
| 日志开关全表、TTFT 排查、模型交互记录（cache-trace.jsonl / anthropic-payload.jsonl / commands.log） | `xiaoaiclaw-ops/logging-guide.md`（628 行） |
| OTel Trace/Metrics/Logs | `extensions/diagnostics-otel`；本地全家桶 `xiaoaiclaw-ops/o11y/`（collector + Jaeger + Prometheus + Grafana） |
| Grafana 看板（成功率/首包 P95/Compaction/Redis/Cache 命中/Shadow 模型…） | `xiaoaiclaw-ops/monitoring/grafana/` |
| 线上排查 SOP | `xiaoaiclaw-ops/incidents/debug-sop-proxy-no-response.md`（小爱 APP 无回复分步排查）等 |
| 事故复盘 | `xiaoaiclaw-ops/incidents/`、`xiaoaiclaw-ops/analysis/`（如 prod-kata-oom-20260429.md） |
| bug 归档 | `xiaoaiclaw-ops/bugs/`（18 个，含 memory-saas-isolation-bug.md） |
| E2E 用例 | `xiaoaiclaw-ops/e2e/`（33 case / 7 tier，`python3 xiaoaiclaw-ops/e2e/run.py`） |

## 3. 记忆与会话体系（本仓库最复杂的部分之一）

### 3.1 三层记忆

1. **内置 RAG 记忆（上游）**：`xiaoaiclaw-ops/memory-architecture.md`（735 行）。SQLite + FTS5 BM25 + 向量（6 种 embedding provider；`auto` 无 Key 时**不建索引**），Hybrid 70/30 + MMR + 时间衰减；真相源是 Markdown（`MEMORY.md` bootstrap 注入 system prompt + `memory/*.md`）；`memory_search`/`memory_get` 两个 Tool；compaction 前 Memory Flush。**坑**：tool profile 为 `messaging` 时无 memory（新装"失忆"）。
2. **云端记忆插件（二开）**：`plugins-local/memory-cloud/**`（"方案 E：plugin 完全自治"，主仓净侵入 +51 行）。经 proxy `/internal/skills` 调 API 平台（Pull/Search/Store/Update/Delete…）；bootstrap 拉 LongTerm + Daily 各 500 条注入 system prompt；禁本地文件记忆。设计：`xiaoaiclaw-ops/memory-plugin/memory-cloud-plugin-changelog.md`、`docs/xiaoai/feature-memory-cloud-to-main.md`（sensitivity L1-L4 分级）。
3. **设备差异化记忆**：车载三开关（记忆开关/主驾匹配/声纹昵称）+ `<memory-isolation-strict>`；音箱家庭共享记忆（FamilyMemory）+ 声纹隔离 + L4 敏感过滤（`docs/xiaoai/feature-memory-cloud-soundbox.md`）。

SaaS 隔离：后端按 maskUid+maskDeviceId 双键；guard 决策 `decideCloudMemoryGuardFromSaasContext`；`xiaoaiclaw-ops/memory-plugin/internal-hooks/memory-saas-session-guard/`（非 owner 时 veto session-memory 写）。

### 3.2 Session 存储与同步

- **Redis 存储**：`session.storeMode=redis`，key 前缀 `mibrain:claw:`，5 类 key（meta hash / list zset / msgs list / sids zset / skills string，TTL 30 天）—— `xiaoaiclaw-ops/dev/redis-session-storage-contract.md`、`docs/xiaoai/session-migration-sop.md` §六。
- **快系统同步**：ai-claw-proxy 侧"快系统"经 `chat.inject.session` / `sessions-write-http.ts` 注入 transcript。
- **Redis/JFS 双写排障（强烈推荐精读）**：`docs/xiaoai/session-sync-redis-jfs-double-write.md`（349 行）——三条独立写路径、跨用户 cwd 错位（数据隔离级事故）、sessionId 三源对齐、跨天 session 分裂、Redis 模式下 compaction 写 JFS 副产物等，每项带 commit 号。学"这个系统哪里会写状态"，读这一篇最快。
- **迁移 SOP**：`docs/xiaoai/session-migration-sop.md`（JFS→Redis，脚本 `scripts/migrate-sessions-jfs-to-redis.mjs`）、`session-migration-single-pod-per-user.md`（gvfs fileset 全量迁移）、`session-migration-incremental.md`、`session-migration-saas-kata-resync.md`、`session-write-log-checklist.md`。
- SaaS vs 上游 session：本地 `~/.xiaoaiclaw/agents/<agentId>/sessions/` vs SaaS `{saasWorkspaceDir}/{uid}/agents/{agentId}/sessions/`；`src/gateway/session-utils.ts` 的 loadSessionEntry 按 `xiaoaiMaskUid` 分支 storePath —— **该分支曾两次在上游 sync 中丢失**（见 §5）。

## 4. 二开 Skills 与 Agents

### 4.1 Skills（`xiaoaiclaw-ops/skills/`，31 个）

小爱设备端业务技能：iot-control、iot-automation、cross-device-control、car-automation、music-search-play、navigation、phonecall、reminder、vqa、deep-research、sport-health、ai-search、ai-weather 等。

- **加载机制**：ConfigMap `skills.load.extraDirs: ["/app/xiaoaiclaw-ops/skills"]`（source=`xiaoaiclaw-extra`）。为什么用 extraDirs 而不是往上游 `skills/` 里放？—— 会污染上游目录导致 sync 冲突。六级加载源优先级与门控见 `xiaoaiclaw-ops/preinstall-skills-design.md`（推荐方案 D）。
- 分设备变体 `SKILL.SoundBox.md` / `SKILL.Wearables-Headless.md`：`xiaoaiclaw-ops/docs/device-specific-skill-{loading,content}-tech-standard.md`。
- 快照热更新（重启后旧会话不感知新技能问题）：`xiaoaiclaw-ops/skill-snapshot-refresh-design.md`；开发指南 `xiaoaiclaw-ops/skill-dev-guide.md`。

### 4.2 Agents（`xiaoaiclaw-ops/xiaoai-agents/`，4 个）

`main`（分 Phone/SoundBox/TV/Automotive 的 AGENTS/IDENTITY 变体 + SOUL.md）、`deep-research-agent`、`liveai`、`video-qa`。指令分层设计（根 AGENTS.md vs ops AGENTS.md 的 rebase 影响）：`xiaoaiclaw-ops/agent-instructions-design.md`；skill 仓库映射：`xiaoaiclaw-ops/docs/knowledge/01-architecture/modules/module-agent-repos.md`。

## 5. 上游同步纪律（每个开发者都要懂）

- **分支策略**：`xiaoaiclaw-ops/branch-strategy.md`。remotes：origin（git.n.xiaomi.com）+ upstream（github.com/openclaw/openclaw）。`main`（二开主干，每日 20:00 统一部署 Prev）、`deploy/p4t`（测试环境，push 即部署）、`feat/*`、`skill/*`、`hotfix/*`。**MR→main 强卡：单测 100% 通过 + 增量覆盖率 30%**。
- **每周同步流程**（附录 A）：`upstream/main → sync-upstream-XXXX`（`scripts/rebrand.sh` 重命名）→ merge 到 `sync-target-XXXX` → **分组解冲突**（A 组 `xiaoaiclaw-ops/`+`AGENTS.md` 用 `--ours`；B 组非二开清单用 `--theirs`；C 组二开清单内手动 3-way）→ build/test → MR。Playbook：`xiaoaiclaw-ops/sync-history/upstream-sync-playbook.md`。
- **已知陷阱**（附录 B）：三路 merge 时 import 区整块丢失（merge 前后 `pnpm tsgo` 对比 + grep 验证 saas import）；rebuild 后仍跑旧镜像。
- **upstream-tracker.md**（327 行）：上游 bug workaround 清单 + "哪些可以移除了"。重点：**SaaS Session Store Routing 的 storePath 分支在 sync 时两次丢失**（`4412e2614f`），rebase 后必查；slug-generator 不走 model alias（要用完整 `provider/model`）；skill snapshot stale patch；供应链风险（`stageRuntimeDependencies` 裸 npm install）。

## 6. 安全（`xiaoaiclaw-ops/security/`）

- `security-check.py`：安全基线自动审计（版本/token/端口/DM 策略）。
- 沙箱系列：多租户共享 Gateway 的安全缺口 → DinD/gVisor/Kata 演进；从 `sandbox-testing-overview.md` 读起，实测见 `kata-sandbox-testing.md`。
- 威胁分析全景：`security-hardening-design.md`（多租户隔离、exec 沙箱、FS 防护、网络出口、供应链、Gateway 认证、日志审计）；专项设计在 `security/scheme/`（17 份，如 exec-seven-mechanisms-deepdive.md、workspace-fs-boundary、callback 签名）。
- **贡献者红线**：本地 bind 必须 loopback、K8s 必须 `--bind lan`；Gateway token 只配 `xiaoaiclaw.json`（`openssl rand -hex 32`，别放 `.env`）；不提交真实 API key/手机号；触碰观测面文件走 review checklist（见第 7 章）。

## 7. 动手试试

```bash
./manage.sh status              # 看本地二开环境状态（先读 local-setup.md）
./manage.sh saas-tui --mask-uid <fake-uid>   # 体验 SaaS 多用户调用（testing-only）
```

- 精读 `docs/xiaoai/session-sync-redis-jfs-double-write.md`，画一张"这个系统所有会写 session 状态的位置"清单 —— 这是理解二开最值钱的一张图。
- 读 `xiaoaiclaw-ops/upstream-tracker.md`，体会"与上游共存"的真实代价。

## 8. 深入阅读（推荐顺序）

1. `xiaoaiclaw-ops/README.md` + `xiaoaiclaw-ops/AGENTS.md` —— 全局定位、目录地图、安全红线
2. `xiaoaiclaw-ops/architecture.md` —— Gateway/Channel/Node 与 25 个核心工具清单
3. `xiaoaiclaw-ops/saas-architecture.md` —— SaaS 多租户全解
4. `xiaoaiclaw-ops/k8s-deployment.md` + `branch-strategy.md` + `flow-matrix-guide.md` —— 部署与 CI/CD
5. `xiaoaiclaw-ops/memory-architecture.md` → `memory-plugin/memory-cloud-plugin-changelog.md` → `docs/xiaoai/feature-memory-cloud-{to-main,soundbox}.md` —— 记忆三层
6. `docs/xiaoai/session-sync-redis-jfs-double-write.md` + `session-migration-*.md` —— 会话存储实战
7. `docs/xiaoai/feature-device-model-routing.md` + `feature-vip-model-traffic-routing.md` —— 模型路由
8. `xiaoaiclaw-ops/preinstall-skills-design.md` + `skill-dev-guide.md` —— skills 体系
9. `xiaoaiclaw-ops/upstream-tracker.md` + `branch-strategy.md` 附录 A/B —— 上游同步
10. `xiaoaiclaw-ops/security/sandbox-testing-overview.md` → `security-hardening-design.md` —— 安全
11. 排障：`logging-guide.md`、`incidents/debug-sop-proxy-no-response.md`、`monitoring/grafana/`

二开知识库总入口：`xiaoaiclaw-ops/docs/knowledge/README.md`（8 领域 343+ 文件）。
