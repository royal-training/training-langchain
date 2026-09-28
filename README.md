# LangChain 学习练习册

一套基于 **LangChain 1.x**（`langchain 1.3.17` + `langgraph 1.2.11`）的中文练习 notebook，从模型连接一路练到综合项目。每章结构统一：**学习目标 → 概念讲解 → 可运行的 Demo → 动手练习（附参考答案）**。

模型通过智谱 GLM 的 OpenAI 兼容接口调用（`.env` 中的 `API_KEY_ZHIPU` / `API_URL_ZHIPU`），换模型只需改各章"环境准备"cell 里的 `model` 参数。

## 章节目录（建议按顺序学习）

| 章节 | 主题 | 核心内容 |
|---|---|---|
| [chapter01-model.ipynb](chapter01-model.ipynb) | 连接大模型 | ChatOpenAI / ChatZhipuAI / init_chat_model |
| [chapter02-prompts.ipynb](chapter02-prompts.ipynb) | 提示词与消息 | 4 种消息类型、ChatPromptTemplate、MessagesPlaceholder、few-shot、partial |
| [chapter03-structured-output.ipynb](chapter03-structured-output.ipynb) | 结构化输出 | Pydantic schema、with_structured_output、枚举/嵌套/列表抽取、JsonOutputParser |
| [chapter04-lcel.ipynb](chapter04-lcel.ipynb) | LCEL 链式编排 | `\|` 管道、RunnableLambda/Parallel/Passthrough、条件路由、重试与降级 |
| [chapter05-tools.ipynb](chapter05-tools.ipynb) | 工具与函数调用 | @tool、bind_tools、ToolMessage 回传闭环、手写工具循环、Pydantic 参数 schema |
| [chapter06-agents.ipynb](chapter06-agents.ipynb) | 智能体与记忆 | create_agent、消息流转、checkpointer 多轮记忆、response_format、链 vs Agent 选型 |
| [chapter07-rag.ipynb](chapter07-rag.ipynb) | 检索增强生成 | Document、文本切分、embedding、InMemoryVectorStore、RAG 链、引用溯源 |
| [chapter08-streaming-async.ipynb](chapter08-streaming-async.ipynb) | 流式与异步 | stream、AIMessageChunk、batch、asyncio.gather 并发、astream_events |
| [chapter09-projects.ipynb](chapter09-projects.ipynb) | 综合项目实战 | 智能客服 Agent、知识库问答、内容生产流水线（3 个完整项目） |

## 知识点笔记（阅读版）

不想打开 Jupyter 时，可以直接阅读 [notes/](notes/) 目录下的 Markdown 版知识整理（概念讲解 + 核心示例代码，不含练习）：

[ch02 提示词与消息](notes/chapter02-prompts.md) · [ch03 结构化输出](notes/chapter03-structured-output.md) · [ch04 LCEL 链式编排](notes/chapter04-lcel.md) · [ch05 工具与函数调用](notes/chapter05-tools.md) · [ch06 智能体与记忆](notes/chapter06-agents.md) · [ch07 RAG](notes/chapter07-rag.md) · [ch08 流式与异步](notes/chapter08-streaming-async.md) · [ch09 综合项目实战](notes/chapter09-projects.md)

## 如何使用

```bash
# 1. 激活虚拟环境（已包含全部依赖）
source .venv/bin/activate
pip list | grep langchain   # 确认 langchain >= 1.0

# 2. 确认 .env 配置（已被 .gitignore 忽略，不要提交）
#    API_KEY_ZHIPU=你的智谱 API Key
#    API_URL_ZHIPU=https://open.bigmodel.cn/api/paas/v4/

# 3. 启动 Jupyter，按章节顺序学习
jupyter lab
```

**学习建议**：每章先顺序运行 Demo 格，理解输出；做练习时**新建 cell 自己写**，写完再对照每章末尾的参考答案。参考答案也是可运行的，可直接执行对照结果。

## 注意事项

- **费用**：所有 Demo 和练习都会发起真实 API 调用（每次输入都很短，费用极低），但请避免无脑重复运行整章。
- **版本**：本练习册基于 LangChain 1.x。网上大量教程基于 0.x 旧版（`LLMChain`、`create_react_agent` 等已移入 `langchain_classic` 包），API 写法不同，注意甄别。
- **智谱兼容层已知行为**（练习册代码已规避）：
  - `with_structured_output` 必须显式传 `method="function_calling"`，默认的 `json_schema` 模式智谱不支持；
  - langgraph 1.x 的 `ToolNode` 需在图运行时内使用，不要脱离图单独 invoke。
