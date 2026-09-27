# LangChain 练习册生成 & init_chat_model 用法

> 对话日期：2026-09-26
> 环境：langchain 1.3.17 / langchain-core 1.6.0 / langgraph 1.2.11 / Python 3.13.5
> 模型接入：智谱 GLM（`glm-4.5-air`）的 OpenAI 兼容接口，`.env` 配置 `API_KEY_ZHIPU` 与 `API_URL_ZHIPU`（形如 `https://open.bigmodel.cn/api/paas/v4/`）

---

## Q1：想熟练掌握 LangChain，生成一套详细的练习 notebook（基本使用 + demo 练习）

### 交付内容

在仓库根目录生成了 8 章中文练习 notebook + README 学习指南，与已有的 chapter01 衔接。每章结构统一：**学习目标 → 概念讲解 → 可运行 Demo → 动手练习（文末附可运行的参考答案）**。

| 章节 | 主题 | 核心内容 |
|---|---|---|
| chapter01-model.ipynb（已有） | 连接大模型 | ChatOpenAI / ChatZhipuAI / init_chat_model |
| chapter02-prompts.ipynb | 提示词与消息 | 4 种消息类型、ChatPromptTemplate、MessagesPlaceholder、few-shot、partial |
| chapter03-structured-output.ipynb | 结构化输出 | Pydantic schema、with_structured_output、枚举/嵌套/列表抽取、JsonOutputParser |
| chapter04-lcel.ipynb | LCEL 链式编排 | `\|` 管道、RunnableLambda/Parallel/Passthrough、条件路由、重试与降级 |
| chapter05-tools.ipynb | 工具与函数调用 | @tool、bind_tools 闭环、手写工具循环（Agent 原理）、Pydantic 参数 schema |
| chapter06-agents.ipynb | 智能体与记忆 | create_agent、消息流转、checkpointer 多轮记忆、response_format、链 vs Agent 选型 |
| chapter07-rag.ipynb | 检索增强生成 | 文档切分、embedding-3 向量化、InMemoryVectorStore、RAG 链、引用溯源 |
| chapter08-streaming-async.ipynb | 流式与异步 | stream、AIMessageChunk、batch、asyncio.gather 并发、astream_events |
| chapter09-projects.ipynb | 综合项目实战 | 智能客服 Agent / 知识库问答 / 内容生产流水线（3 个完整项目） |

### 生成前做的验证（保证代码可跑）

1. **静态校验**：全部代码格通过语法编译、导入解析、nbformat 校验。
2. **离线 dry-run**：用假模型（`GenericFakeChatModel`）/假 embedding（`DeterministicFakeEmbedding`）真跑模板格式化、LCEL 组装、工具执行、RAG 检索链、create_agent 装配、流式事件等关键链路，不产生 API 费用。
3. **少量真实 API 验证**，发现并规避了 3 个兼容性问题（见下）。

### ⚠️ 实测发现的兼容性问题（重要，写代码时注意）

**① 智谱不支持 OpenAI 原生 `json_schema` 模式**

langchain-openai 1.6 中 `with_structured_output` 默认走 `json_schema`（`response_format` 参数），智谱兼容层会无视该参数、返回自由文本，随后解析报错：

```
pydantic_core._pydantic_core.ValidationError: Invalid JSON: trailing characters ...
```

解决办法：**显式指定走 function calling**（已实测可用）：

```python
llm.with_structured_output(MyModel, method="function_calling")
```

而 `create_agent` 的 `response_format=MyModel` 参数实测直接可用，无需额外处理。

**② langgraph 1.x 的 `ToolNode` 不能脱离图运行时单独 invoke**

旧教程常见的 `ToolNode([tools]).invoke({"messages": [...]})` 独立调用在 langgraph 1.1 下会报：

```
ValueError: Missing required config key 'N/A' for 'tools'.
```

它被设计为在 LangGraph 图运行时内工作，实际开发中直接用 `create_agent`（内部已内置这套调度）即可。

**③ `init_chat_model` 必须传 `base_url`**

chapter01 第 4 格的 `ConnectTimeout` 就是漏传它导致默认连了 `api.openai.com`，详见 Q2。

### 其他注意事项

- `.env`（含 API key）曾被暂存进 git，已用 `git rm --cached .env` 移出暂存区（文件保留）。`.gitignore` 已包含 `.env`。
- chapter01 第 1 格 `print(api_key)` 的输出里带真实 key，分享 notebook 前记得清掉该格输出。
- 所有 Demo/练习都会发起真实 API 调用（输入短、费用极低），避免无脑重复运行整章。
- 本练习册基于 LangChain 1.x；网上大量 0.x 旧教程（`LLMChain`、`create_react_agent` 等）API 已移入 `langchain_classic` 包，写法不同，注意甄别。

---

## Q2：init_chat_model 用法

`init_chat_model` 是一个**模型工厂函数**：按模型名字符串自动选择并实例化对应 provider 的 ChatModel 类，核心价值是**代码与 provider 解耦**。

### 基本用法（智谱场景，已实测通过）

```python
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "openai:glm-4.5-air",                      # provider 前缀:模型名
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url=os.getenv("API_URL_ZHIPU"),   # 关键：指向智谱的 OpenAI 兼容端点
    temperature=0.6,
)
print(llm.invoke("你好").content)          # 返回的就是 ChatOpenAI 实例
```

`openai:` 前缀只决定"实例化 `ChatOpenAI` 类"；不传 `base_url` 时客户端默认连 `api.openai.com`，智谱 key 在那里必然 ConnectTimeout——这就是常见报错根因。

### 三种用法形态

**① 前缀形式（最常用）**：`"openai:gpt-4o"`、`"anthropic:claude-sonnet-4"`、`"deepseek:deepseek-chat"`。

**② 分开传参**（等价）：

```python
llm = init_chat_model(model="glm-4.5-air", model_provider="openai", base_url=..., api_key=...)
```

**③ 部分实例化**（工程化核心卖点：先声明、后补全，适合写库/配置化切换模型）：

```python
# 库/模块作者只声明"需要一个模型"，不关心是谁家的
placeholder = init_chat_model(temperature=0.6)

# 使用者运行时才决定
llm = placeholder.bind(model="openai:glm-4.5-air", base_url=..., api_key=...)
```

### 自动推断与例外

内置按模型名前缀推断 provider：`gpt-*` → openai、`claude-*` → anthropic、`deepseek-*` → deepseek 等。

**注意 `glm-*` 不在推断列表**，必须显式写 `openai:` 前缀。智谱没有专属 provider，走 `openai:` 兼容路线是标准做法（moonshot/kimi、qwen 等同理）。

### 要点速记

- **内置 provider**（1.3.17 版实测 28 个）：`openai`、`anthropic`、`deepseek`、`google_vertexai`、`google_genai`、`ollama`、`mistralai`、`groq`、`xai`、`perplexity` 等。provider 名写错会报 `ValueError` 并列出全部可选值。
- **需要对应集成包**：`openai:` 要求安装 `langchain-openai`，缺失报 `ImportError`。
- **参数透传**：`temperature`、`base_url`、`max_tokens` 等原样传给底层 `ChatOpenAI`。
- **返回对象无阉割**：就是标准 `ChatOpenAI`，`bind_tools`、`with_structured_output(method="function_calling")`、流式等全部可用。

### 什么时候用 init_chat_model，什么时候直接 ChatOpenAI

- 直连 `ChatOpenAI(base_url=智谱)`：确定只用智谱的简单场景（练习章节约够用）。
- `init_chat_model`：需要**配置化切换模型**的场景——模型名放配置文件/命令行参数，同一套代码今天跑 GLM、明天跑 DeepSeek 做对比；或写可复用库时把模型选择权交给调用方。
