# 第 9 章 综合项目实战

> 运行环境：langchain 1.3.17 / langgraph 1.2.11 / Python 3.13，模型为智谱 `glm-4.5-air`（OpenAI 兼容接口）。
> 以下示例中的 `llm` 均指下述初始化：

```python
import os
from dotenv import load_dotenv
load_dotenv()  # 读取 .env 中的 API_KEY_ZHIPU / API_URL_ZHIPU

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.6,
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url=os.getenv("API_URL_ZHIPU"),  # https://open.bigmodel.cn/api/paas/v4/
)
```

> 💡 动手练习与参考答案见同章 `.ipynb`；智谱兼容层的注意事项见 [README](../README.md)。


**学习目标**
综合前 8 章的知识，独立完成 3 个贴近真实业务的项目：
1. **智能客服助手** —— 工具 + Agent + 多轮记忆（ch05/06）
2. **文档知识库问答** —— RAG 全流程 + 引用溯源（ch07）
3. **内容生产流水线** —— LCEL 并行 + 结构化输出（ch03/04/08）

每个项目先给**需求和验收标准**，建议自己实现后再看参考答案。

---

## 项目 1：智能客服助手「小星」

**需求**：为一家电商公司搭建客服 Agent「小星」，能：
- 查订单状态（订单号 SO 开头）
- 算优惠价格（用户给原价和优惠方式）
- 查退换货政策（政策内容自己编 3 条）
- 记住会话中用户说过的信息（姓名、订单号）

**验收标准**
- [ ] 使用 `create_agent` + 至少 3 个工具
- [ ] 带 `checkpointer`，同 thread_id 下第二轮不用重复提供订单号
- [ ] 模型算价格必须走工具，不允许心算（写进 system_prompt 验证）
- [ ] 问"你们公司老板是谁"这类无法回答的问题时，礼貌承认不知道而不是编造

**提示**：工具里用模块级 dict 存伪数据；"不编造"靠 system_prompt 约束 + 工具返回明确的"未找到"信息。

---

## 项目 2：文档知识库问答「内部wiki」

**需求**：把一份"新员工 FAQ"（下面给定）建成可问答的知识库：
- 切分、向量化、入库、检索、生成全流程
- 答案必须**附引用来源**（文档名）
- 知识库没有的内容明确说"无相关信息"

**验收标准**
- [ ] 用 `RecursiveCharacterTextSplitter` 切分（能说清 chunk_size 为何这么选）
- [ ] RAG 链基于 LCEL，`retriever` 以 Runnable 形式接入
- [ ] 至少 2 个"能答对"问题和 1 个"正确拒答"问题的测试记录

**提示**：参考 ch07 第 5 节的 `RunnableParallel(...).assign(answer=...)` 模式。

---

## 项目 3：内容生产流水线「一键成稿」

**需求**：输入一个产品名，产出一份结构化的营销包：
- 并行生成：标题、卖点（3 条）、短文案（50 字内）
- 用**结构化输出**把结果定形为 Pydantic 模型（不允许自由文本）
- 最后用代码把结构化结果渲染成 Markdown 模板

**验收标准**
- [ ] 三个生成任务**并行**执行（`RunnableParallel`）
- [ ] 生成阶段用 `with_structured_output` 保证 schema
- [ ] 渲染阶段是**纯代码**，不调用 LLM

**提示**：先 `RunnableParallel` 拿三路输出，再接一个 `RunnableLambda` 做代码渲染；结构化输出放在每条分支链的解析器位置。

---

## 下一步学习路线

完成本章后，你已经具备 LangChain 的核心能力。建议按这个顺序进阶：

| 方向 | 学什么 | 为什么 |
|---|---|---|
| **LangGraph** | StateGraph、条件边、human-in-the-loop | 复杂多步工作流、多 Agent 协作的事实标准（`create_agent` 内部就是它） |
| **LangSmith** | 追踪、评估、数据集 | 生产环境调试与质量回归的必备可观测平台 |
| **RAG 进阶** | 混合检索、重排序（rerank）、Parent-Document、多轮 query 改写 | 把 ch07 的玩具 RAG 变成能打的 RAG |
| **MCP** | Model Context Protocol | 工具生态标准协议，让 Agent 接入现成的外部工具 |
| **部署** | LangServe / FastAPI + SSE | 把 Notebook 里的链变成线上服务 |

> 一个检验学习成果的好办法：把本章 3 个项目**重构成 FastAPI 接口**（流式接口用 SSE），你就同时打通了开发和服务端两块拼图。

---

> 📝 本笔记由 [chapter09-projects.ipynb](../chapter09-projects.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
