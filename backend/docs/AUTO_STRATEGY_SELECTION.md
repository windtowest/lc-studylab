# 自动检索策略选择

## 概述

自动策略选择功能根据用户查询的特征，智能选择最合适的检索策略，提升 RAG 系统的检索效果。

## 三种检索策略

| 策略 | 适用场景 | 特点 |
|------|---------|------|
| **similarity** | 具体问题、事实查询、定义解释 | 返回最相关的文档，速度快 |
| **mmr** | 宽泛探索、对比分析、需要多样性 | 在相关性和多样性之间平衡 |
| **similarity_score_threshold** | 推荐请求、需要高质量结果 | 只返回高相关度文档，宁缺毋滥 |

## 三种选择模式

### 1. 基于规则（rule-based）

**优点：** 速度快，无需调用 LLM，适合高并发场景

**原理：** 通过正则表达式匹配关键词，识别查询意图

**示例：**
```python
from rag.strategy_selector import create_strategy_selector

selector = create_strategy_selector(mode="rule")
strategy = selector.select_strategy("介绍一下机器学习")
# 返回: RetrievalStrategy.MMR
```

**意图识别规则：**
- 宽泛探索：`介绍.*?一下`、`概述`、`有哪些` → MMR
- 具体问题：`.*?是什么`、`.*?的定义` → similarity
- 对比分析：`.*?和.*?的区别`、`对比` → MMR
- 推荐请求：`最好的`、`推荐`、`高质量` → threshold
- 操作指南：`如何`、`怎么`、`步骤` → similarity
- 事实查询：`什么时候`、`在哪里`、`谁` → similarity

### 2. 基于 LLM（llm-based）

**优点：** 更智能，能理解复杂语义

**缺点：** 需要调用 LLM，有延迟和成本

**示例：**
```python
selector = create_strategy_selector(mode="llm")
strategy = selector.select_strategy("比较深度学习和传统机器学习的优缺点")
# LLM 分析后返回: RetrievalStrategy.MMR
```

**工作原理：**
1. 将查询和策略说明发送给 LLM
2. LLM 分析查询特征，选择最合适的策略
3. 解析 LLM 输出，映射到策略枚举

### 3. 混合模式（hybrid，推荐）

**优点：** 兼顾速度和准确性

**策略：**
1. 先用规则快速判断（覆盖 80% 常见场景）
2. 如果规则无法判断，再调用 LLM（处理复杂场景）

**示例：**
```python
selector = create_strategy_selector(mode="hybrid")
strategy = selector.select_strategy("什么是神经网络？")
# 规则匹配成功，返回: RetrievalStrategy.SIMILARITY
```

## 使用方法

### 方式一：在创建检索器时自动选择

```python
from rag import get_embeddings, load_vector_store, create_retriever

# 加载向量库
embeddings = get_embeddings()
vector_store = load_vector_store("data/indexes/my_docs", embeddings)

# 自动选择策略
retriever = create_retriever(
    vector_store,
    auto_select=True,           # 开启自动选择
    query="介绍一下机器学习",    # 提供查询文本
    k=4
)

# 执行检索
docs = retriever.invoke("介绍一下机器学习")
```

### 方式二：先选择策略，再创建检索器

```python
from rag.strategy_selector import create_strategy_selector

# 创建选择器
selector = create_strategy_selector(mode="hybrid")

# 选择策略
query = "深度学习和传统机器学习有什么区别？"
strategy = selector.select_strategy(query)

# 创建检索器
retriever = create_retriever(
    vector_store,
    search_type=strategy.value,
    k=4
)
```

### 方式三：在 RAG Agent 中集成

```python
from rag import create_rag_agent
from rag.strategy_selector import create_strategy_selector

# 创建策略选择器
selector = create_strategy_selector(mode="hybrid")

# 在查询前选择策略
query = "如何训练神经网络？"
strategy = selector.select_strategy(query)

# 创建对应策略的检索器
retriever = create_retriever(
    vector_store,
    search_type=strategy.value
)

# 创建 RAG Agent
agent = create_rag_agent(retriever)
result = agent.invoke(query)
```

## 配置选项

### 全局配置

在 `config/settings.py` 中设置默认策略：

```python
# 默认检索策略（当不使用自动选择时）
retriever_search_type: str = "similarity"

# 是否默认启用自动选择
auto_select_strategy: bool = False
```

### 运行时配置

```python
# 创建混合选择器，但不使用 LLM 兜底
selector = HybridSelector(use_llm_fallback=False)

# 使用自定义 LLM 模型
from core.models import get_chat_model
custom_model = get_chat_model("openai:gpt-4")
selector = create_strategy_selector(mode="llm", model=custom_model)
```

## 性能对比

| 模式 | 平均延迟 | 准确率 | 成本 | 推荐场景 |
|------|---------|--------|------|---------|
| rule | ~1ms | 85% | 无 | 高并发、低延迟要求 |
| llm | ~200ms | 95% | 有 | 准确性要求高 |
| hybrid | ~10ms | 92% | 低 | 通用场景（推荐） |

## 测试

运行测试脚本：

```bash
cd backend
python scripts/test_auto_strategy.py
```

## 扩展

### 添加新的意图规则

编辑 `rag/strategy_selector.py`：

```python
class RuleBasedSelector:
    INTENT_PATTERNS = {
        # 添加新意图
        QueryIntent.TROUBLESHOOTING: [
            r"为什么.*?不工作", r".*?报错", r".*?问题",
            r"如何.*?解决", r".*?失败"
        ],
        # ...
    }
    
    INTENT_TO_STRATEGY = {
        QueryIntent.TROUBLESHOOTING: RetrievalStrategy.SIMILARITY,
        # ...
    }
```

### 自定义 LLM 提示词

```python
class LLMBasedSelector:
    STRATEGY_SELECTION_PROMPT = """
    你是检索策略专家。根据查询选择策略。
    
    [自定义你的提示词...]
    """
```

## 最佳实践

1. **默认使用混合模式**：兼顾速度和准确性
2. **高并发场景用规则模式**：避免 LLM 调用成本
3. **复杂查询用 LLM 模式**：提升准确性
4. **定期分析日志**：优化规则匹配模式
5. **A/B 测试**：对比不同策略的实际效果

## 监控指标

建议记录以下指标：

- 策略选择分布（similarity/mmr/threshold 各占比）
- 规则命中率（规则模式 vs LLM 兜底）
- 检索结果质量（用户反馈、点击率）
- 平均检索延迟

## 故障排查

**问题：规则无法匹配我的查询**

解决：
1. 检查查询是否包含规则中的关键词
2. 添加新的正则表达式规则
3. 使用 LLM 或混合模式

**问题：LLM 选择的策略不合理**

解决：
1. 优化提示词，提供更多示例
2. 使用更强的 LLM 模型（如 GPT-4）
3. 添加后处理逻辑，修正明显错误的选择

**问题：自动选择比手动指定慢**

解决：
1. 使用规则模式而非 LLM 模式
2. 缓存常见查询的策略选择结果
3. 对于已知场景，直接手动指定策略
