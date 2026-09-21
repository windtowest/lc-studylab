# 面向编程学习与代码理解的检索增强智能体系统：技术设计

> 状态：实施前技术设计  
> 目标：在**本人实现的 LC-StudyLab 学习与研究智能体系统**既有 Chat、文档 RAG、学习工作流和深度研究能力基础上，新增可部署的代码检索与代码学习能力。  
> 非目标：本文件不替代代码检索论文实验设计；不删除或重写当前学习助手能力；不把离线评测脚本直接暴露为线上服务。

## 1. 设计定位

系统面向两类任务：

1. **编程学习与代码理解**：用户提问“某业务流程在哪里实现”“应按什么顺序阅读仓库”“某接口如何调用”；系统结合源代码检索与研发文档 RAG 输出带证据的学习路径和解释。
2. **代码定位与工程排查**：用户输入 issue、需求、错误日志或异常堆栈；系统返回候选文件/函数、代码片段、检索依据和后续排查建议。

代码检索不是通用文档 RAG 的替代品，而是 Agent 可调用的专用工具：

```text
用户问题
  ├─ 文档问题 ───────────────→ 文档 RAG（现有能力）
  ├─ 代码定位问题 ───────────→ Code Locator（新增核心能力）
  └─ 复合问题 ──→ 任务编排 ─→ Code Locator + 文档 RAG + Agent 解释
```

## 2. 现有基础与新增边界

### 2.1 保留的现有能力

| 层次 | 当前模块 | 现有落点 | 在目标系统中的职责 |
|---|---|---|---|
| HTTP 服务 | FastAPI 入口 | `backend/api/http_server.py` | 生命周期、统一 `/smartfarm-ai` 前缀、路由注册、健康检查 |
| 通用对话 | Chat Agent | `backend/agents/base_agent.py`、`api/routers/chat.py` | 对话、工具调用、SSE 输出 |
| 文档知识库 | 文档 RAG | `backend/rag/`、`api/routers/rag.py` | 研发文档、课程资料、API/部署规范问答 |
| 有状态学习流程 | Workflow | `backend/workflows/`、`api/routers/workflow.py` | 计划、练习、评分、反馈、会话持久化 |
| 深度研究 | Deep Research | `backend/deep_research/`、`api/routers/deep_research.py` | 多步研究/工具编排，后续可接入代码证据 |
| 小程序入口 | AI 助手 | `smartfarm-front-applets/app/packageA/ai_assistant/` | Chat/RAG/Workflow 三页签、SSE、历史和配置 |

### 2.2 必须新增的能力

| 模块 | 目标 |
|---|---|
| 代码仓库注册与索引管理 | 把本地/授权代码仓变为可版本化、可重建、可查询的代码索引 |
| 在线 Code Locator | 把 HybridG、反馈补检、A3 重排从离线实验封装为稳定服务 |
| 代码—文档协同工作流 | 统一调用代码检索和文档 RAG，生成带代码和文档来源的回答 |
| 代码助手小程序页 | 让用户选择仓库、输入问题、查看候选代码、提交反馈 |
| 私有数据集与评测管线 | 构建 LabCodeLoc 与 LabDevRAG，验证真实私有仓库上的方法和系统效果 |
| 反馈自适应（后期） | 收集反馈并离线优化检索参数/重排策略；训练不进入在线请求路径 |

### 2.3 明确不做的事

- 不修改现有 `/rag/*` 的文档索引格式，也不让代码检索伪装成文档 RAG。
- 不将 `paper/paper/src/eval_hybrid.py`、SWE-bench 评测脚本或带 gold 的评测逻辑直接用于线上服务。
- 不在小程序请求路径启动 RL/RLHF 训练、模型微调或长时索引构建。
- 不把“Agent 自进化”描述为无监督自动改写生产系统；只做可审计的反馈收集、离线评测和人工确认的参数/模型升级。

## 3. 目标架构

```text
┌──────────────────────────────────────────────────────────────────┐
│ 微信小程序 smartfarm-front-applets                                │
│ Chat │ 知识检索 │ 学习工作流 │ 代码助手（新增）                   │
└───────────────────────────────┬──────────────────────────────────┘
                                │ REST / SSE
┌───────────────────────────────▼──────────────────────────────────┐
│ LC-StudyLab FastAPI / LangGraph 后端                              │
│                                                                    │
│ 现有：Chat、RAG、Workflow、Deep Research                          │
│ 新增：Code Search Router ─ Repository Index Service               │
│                         └─ Code Locator                           │
│ 新增：Code Learning Workflow / Code Locator Tool                  │
└───────────────┬─────────────────────┬────────────────────────────┘
                │                     │
┌───────────────▼───────────────┐ ┌───▼────────────────────────────┐
│ 代码检索运行时                 │ │ 文档 RAG 运行时（现有）          │
│ BM25 / Dense_fn / Dense_file   │ │ FAISS 文档向量索引               │
│ 调用图 post-fusion bonus       │ │ README、设计、API、规范、手册     │
│ 缺失类型补检 / A3 / 重排       │ └────────────────────────────────┘
└───────────────┬───────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│ 索引和数据资产                                                     │
│ 授权代码仓、代码符号/文件/图索引、LabCodeLoc、LabDevRAG、反馈日志  │
└──────────────────────────────────────────────────────────────────┘
```

## 4. 后端模块设计

新增目录建议为 `agent/lc-studylab/backend/code_retrieval/`：

```text
code_retrieval/
├── schemas.py                  # Pydantic/内部 DTO：仓库、符号、命中、反馈
├── repository_registry.py      # 已注册仓库、访问路径、语言、commit、索引版本
├── index_builder.py             # 异步/CLI 索引构建编排，不在查询接口中执行
├── artifact_store.py            # 索引资产加载、版本校验、缓存和释放
├── lexical_retriever.py         # BM25 适配
├── dense_retriever.py           # Embedding + FAISS 适配
├── file_retriever.py            # 文件层摘要和映射
├── graph_retriever.py           # 调用图、结构候选与图评分
├── hybrid_service.py            # 迁移 HybridRetriever 融合逻辑
├── feedback_service.py          # 缺失类型分类、图/重写支路候选扩展
├── reranker.py                  # A3 路由与 listwise 重排适配，含超时/回退
├── locator.py                   # 对外统一查询编排：CodeLocator
├── feedback_store.py            # 审计型用户反馈记录
└── jobs.py                      # 索引构建任务状态；首版可使用进程内任务
```

### 4.1 代码索引数据模型

线上必须使用稳定、可追溯的 ID，不使用 CodeSearchNet 行号作为业务 ID。

```json
{
  "repository_id": "lab-java-backend",
  "index_version": "git:<commit_sha>:<build_time>",
  "symbol_id": "lab-java-backend:src/main/java/com/lab/device/AlarmService.java:AlarmService#dispatchOfflineAlert",
  "language": "java",
  "path": "src/main/java/com/lab/device/AlarmService.java",
  "qualified_name": "com.lab.device.AlarmService#dispatchOfflineAlert",
  "kind": "method",
  "start_line": 86,
  "end_line": 121,
  "snippet": "...",
  "file_id": "lab-java-backend:src/main/java/com/lab/device/AlarmService.java"
}
```

每个索引版本至少保存：仓库 ID、来源 commit/版本、构建时间、解析语言、符号数、文件数、embedding 模型名、构建配置和校验和。

### 4.2 检索执行路径

```text
query + repository_id
  → 参数校验和仓库/版本解析
  → BM25 Top-K、函数 Dense Top-K、文件 Dense Top-K 并行召回
  → 文件结果展开为符号候选
  → HybridG 融合：加权排序 + graph post-fusion bonus
  → LLM 一次输出 complexity 与 missing_types（可配置）
  → 根据 A3 路由决定是否做反馈扩池与深度重排
  → 输出 Top-K 文件/符号、代码片段、证据与可解释元数据
```

**迁移原则：** 从论文项目迁移 `hybrid_retriever.py` 中的融合原语和已验证配置；将重型数据加载、编码、索引构建从 `eval_hybrid.py` 中拆出。线上查询不得依赖评测集标注、gold 文件或预测保存逻辑。

### 4.3 A3 在线策略边界

A3 的具体阈值必须通过验证集/私有开发集标定，并存入索引或服务配置版本。线上响应必须记录：

- `route`、`complexity`、`dense_top_conf`；
- 是否进行了反馈扩池、查询重写、图扩展、重排；
- 每步耗时、LLM 调用次数、重排回退与原因；
- `index_version` 与检索配置版本。

这样可支持后续效率实验和反馈优化。任何“重排保证不退化”的表述必须由固定评测协议验证，不能作为系统绝对承诺。

## 5. API 设计

新建 `backend/api/routers/code_search.py`，由 `http_server.py` 按现有方式注册：

```python
from api.routers import code_search
app.include_router(code_search.router, prefix=API_PREFIX)
```

### 5.1 仓库与索引管理

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/smartfarm-ai/code-search/repositories` | 获取可查询仓库及索引状态 |
| POST | `/smartfarm-ai/code-search/repositories` | 注册已授权仓库；仅管理端使用 |
| POST | `/smartfarm-ai/code-search/repositories/{repository_id}/index-jobs` | 创建索引构建/更新任务；仅管理端使用 |
| GET | `/smartfarm-ai/code-search/index-jobs/{job_id}` | 查询任务进度、错误、索引版本 |

### 5.2 查询与反馈

`POST /smartfarm-ai/code-search/query`

```json
{
  "repository_id": "lab-java-backend",
  "query": "设备离线后为什么没有发送告警通知？",
  "top_k": 5,
  "mode": "locate",
  "filters": {"language": ["java"]},
  "include_snippet": true
}
```

```json
{
  "query_id": "uuid",
  "repository_id": "lab-java-backend",
  "index_version": "git:abc123:2026-09-21T10:00:00Z",
  "route": "C",
  "latency_ms": 820,
  "llm_calls": 2,
  "hits": [
    {
      "rank": 1,
      "symbol_id": "...:AlarmService#dispatchOfflineAlert",
      "path": "src/main/java/com/lab/device/AlarmService.java",
      "qualified_name": "com.lab.device.AlarmService#dispatchOfflineAlert",
      "kind": "method",
      "span": {"start_line": 469, "end_line": 516},
      "snippet": "...",
      "score": 0.0,
      "evidence": {
        "channels": ["dense", "file"],
        "graph_related": false,
        "feedback_added": false,
        "reranked": true
      }
    }
  ],
  "warnings": []
}
```

`POST /smartfarm-ai/code-search/feedback`

```json
{
  "query_id": "uuid",
  "repository_id": "smartfarm-front-applets",
  "index_version": "git:abc123:...",
  "feedback_type": "helpful",
  "selected_symbol_id": "...",
  "comment": "可选，长度受限"
}
```

反馈只追加记录，保留时间、匿名用户/会话哈希、查询/索引版本、候选排序和显式反馈。涉及源码内容时，不记录超过必要长度的代码片段。

### 5.3 Agent 工具接口

代码检索功能同时封装为 LangChain Tool：

```text
code_locator(query, repository_id, top_k=5, language_filter=None)
```

工具返回结构化 JSON，不直接生成结论。Chat/Deep Research/代码学习工作流负责：

1. 判断是否需要调用工具；
2. 用工具返回的路径、符号、片段和文档来源组织解释；
3. 在回答中区分“检索证据”“模型推断”“需要人工确认的内容”。

## 6. 小程序设计

在 `packageA/ai_assistant` 中新增第四页签：`代码助手`。

```text
components/
├── CodeSearchView.vue           # 新页面主体
├── RepositorySelector.vue        # 仓库选择与索引状态
├── CodeResultCard.vue            # 文件/函数候选卡
├── CodeEvidencePanel.vue         # 通道/路由/片段解释
└── CodeFeedbackBar.vue           # 准确、有用、无用反馈
```

前端 API 集中放到 `packageA/ai_assistant/api/ai_assistant.js`，与现有 Chat/RAG/Workflow 一致：

```text
getCodeRepositories()
queryCode(params)
createCodeIndexJob(repositoryId, params)       # 管理端或开发环境
getCodeIndexJob(jobId)
sendCodeSearchFeedback(params)
```

代码助手首版交互：

```text
选择仓库 → 输入问题/日志 → 查询 → 查看 Top-K 代码卡片
  ├─ 查看上下文
  ├─ 基于此讲解（带候选调用 Chat Agent）
  ├─ 作为学习起点（带候选进入学习工作流）
  └─ 准确/部分准确/不准确反馈
```

小程序不保存完整私有源码；本地仅保存会话 ID、查询历史摘要、用户配置和可清除的反馈状态。

## 7. 代码—文档协同工作流

首版不强制上多个自治 Agent，使用可观测的 LangGraph 编排即可：

```text
意图分类
  ├─ 仅文档问题 → 文档 RAG → 回答
  ├─ 仅代码定位 → Code Locator → 代码解释
  └─ 复合问题   → 并行 Code Locator + 文档 RAG
                       ↓
                 证据融合与回答生成
                       ↓
                 用户反馈/会话持久化
```

后续可将其纳入现有 Deep Research 的子任务体系，但前提是先完成独立的 Code Locator API 和单元/集成测试。

## 8. 数据与实验设计

### 8.1 LabCodeLoc：私有代码定位集

| 项目 | 设计 |
|---|---|
| 目标 | 验证真实实验室代码仓上的文件/函数定位能力 |
| 仓库 | 实验室 Java 后端仓库为主；可加入 `smartfarm-front-applets` 等前端仓作为跨语言补充。代码统一纳入可用数据范围，不做按用户角色区分 |
| 规模 | 先完成 100–300 条高质量样本；扩展时单独版本化 |
| 样本来源 | 已关闭 issue、Git 修复提交、测试失败记录、经审查的真实需求；人工构造仅单独标识 |
| 标注 | `query / repo / commit / gold_files / gold_symbols / task_type / difficulty / provenance` |
| 质量 | 抽样双人复核；记录分歧与裁决；测试集固定后不可用于阈值调参 |
| 隐私 | 不公开源代码；如需发布，只发布脱敏 ID、统计、协议和评测脚本 |

### 8.2 LabDevRAG：研发文档知识库

可含 README、架构/接口文档、部署说明、编码规范、故障手册、数据库/设备协议。它用于文档问答与代码解释，不与 LabCodeLoc 混合计分。

### 8.3 评测层级

| 层级 | 对象 | 对照 | 指标 |
|---|---|---|---|
| 方法层 | CSN、SWE-bench Lite | BM25、Dense、HybridG、FB/A3 消融 | R@1/5/10、MRR、NDCG、延迟、LLM 调用数 |
| 私有检索层 | LabCodeLoc 固定测试集 | BM25、Dense、文档 RAG 检索、完整方法 | 文件/函数 R@K、MRR、索引与查询延迟 |
| 系统层 | 30–50 个代码理解/排查任务 | Chat-only、文档 RAG、文档 RAG+Code Locator | 任务完成率、证据覆盖率、人工正确性、有用率、成本 |
| 自适应层 | 固定 holdout | 初始策略、反馈调优策略 | 指标增量、校准集与测试集隔离、反馈成本 |

## 9. 安全、权限和运维约束

1. 首版不区分“谁可以使用哪个仓库”：实验室纳入系统的代码仓统一可查询、统一可用于数据集构建；但仓库注册仍只能由服务端配置或管理员命令完成，生产端不能接受任意本地文件路径。
2. 代码索引和日志按仓库版本组织；`repository_id` 不能仅靠前端传入而不校验，以避免误查询不存在或过期的索引。
3. 即使代码统一可用，仍必须在索引和数据集构建前自动排除密钥、账号、令牌、证书、`.env`、生产连接串、构建产物和依赖目录；模型 API Key、数据库密码、服务端地址全部来自环境变量或密钥管理，不写入前端、代码或文档示例。
4. 将 FastAPI 宽松 CORS 改为环境化白名单；小程序生产环境需使用已配置的 HTTPS 合法域名。
5. 索引更新是后台任务，失败可重试且不覆盖上一个可用索引版本。
6. LLM 发送的代码片段需限制长度并最小化；是否允许向外部模型发送片段按实验室统一策略配置，并记录审计信息。
7. self-reward/RLHF 训练与线上推理解耦，训练产物须在固定验证/测试集通过后才能人工发布。

## 10. 实施决策与待确认项

实施前必须确认：

- 首批可授权的代码仓、语言和 Git 可访问方式；
- 后端部署机器的 CPU/GPU、内存、磁盘、模型下载网络；
- 目标小程序是否可访问 HTTPS AI 服务域名；
- PostgreSQL/SQLite 的实际可用配置和用户/会话隔离要求；
- 是否允许将私有代码片段发送到外部 LLM；
- LabCodeLoc 标注负责人、代码脱敏方式与数据保留周期。

具体落地顺序、检查点和验收条件见同目录 [`task.md`](task.md)。
