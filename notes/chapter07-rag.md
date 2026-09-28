# 第 7 章 检索增强生成（RAG）

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
- 理解 RAG 五步流程：加载 → 切分 → 向量化 → 检索 → 生成
- 掌握 `Document` 与文本切分器 `RecursiveCharacterTextSplitter`
- 会用智谱 `embedding-3` 做向量化，`InMemoryVectorStore` 建库检索
- 用 LCEL 组装标准 RAG 链，并让答案**带引用来源**
- 了解生产级向量库的选型方向

> RAG 解决两个问题：模型**不知道你的私有知识**（训练数据里没有），以及**知识会过期**。做法是"开卷考试"：先检索相关资料，把它们塞进提示词，再让模型作答。

## RAG 全流程图

```
私有文档 ──切分──> 文本块 ──Embedding──> 向量库
                                          │
用户提问 ──Embedding────────────────> 相似度检索 Top-K
                                          │
                              检索到的文本块 + 用户问题
                                          ▼
                                    LLM 生成答案（附引用）
```

```python
# 额外准备：本章还需要 embedding 模型（走智谱 OpenAI 兼容接口，已验证可用）
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="embedding-3",   # 智谱的向量大模型
    dimensions=1024,       # 输出向量维度（embedding-3 支持 256~2048）
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url=os.getenv("API_URL_ZHIPU"),
)

v = embeddings.embed_query("测试一下")
print("embedding 维度:", len(v))
```

## 1. Document 与文本切分

`Document` 是 LangChain 的文档标准格式：`page_content`（正文）+ `metadata`（元数据，如来源、页码，引用时靠它溯源）。

**为什么要切分？**
- embedding 模型输入长度有限
- 块太大 → 检索不精准（一堆不相关内容混在一起）；块太小 → 上下文不足
- 经验值：`chunk_size=300~500`、`chunk_overlap=50`（块之间重叠一部分，避免句子被切断后两头都查不到）

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 构造一个迷你"公司知识库"（真实场景用 DocumentLoader 读 PDF/网页/Word）
docs = [
    Document(
        page_content=(
            "星辰科技员工手册（2026 版）。\n\n"
            "考勤制度：公司实行弹性工作制，核心工作时间为 10:00-16:00，"
            "员工可在 8:00-10:00 之间到岗，每日工作满 8 小时即可。"
            "迟到早退每月累计不超过 3 次不予追究。\n\n"
            "年假制度：入职满一年的员工享有 5 天带薪年假，满三年 10 天，满五年 15 天。"
            "年假需提前 3 个工作日在 OA 系统申请，当年有效，最多可结转 5 天至次年 3 月底。\n\n"
            "报销制度：差旅报销需在行程结束后 15 天内提交，单笔超过 5000 元需部门总监审批。"
            "餐补标准为出差期间每餐 50 元上限。"
        ),
        metadata={"source": "员工手册.pdf", "type": "制度"},
    ),
    Document(
        page_content=(
            "星辰科技产品 FAQ。\n\n"
            "问：云盒子（CloudBox）支持多大文件上传？答：单文件最大 20GB，"
            "免费版总容量 100GB，专业版 2TB。\n\n"
            "问：云盒子支持哪些平台？答：Windows、macOS、iOS、Android 全平台支持，"
            "网页版也有。\n\n"
            "问：如何开发票？答：在订单页面点击'申请开票'，电子发票 24 小时内发送到邮箱。"
        ),
        metadata={"source": "产品FAQ.md", "type": "FAQ"},
    ),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块目标长度（RecursiveCharacterTextSplitter 按长度计算，中文一个字约等于一个字符）
    chunk_overlap=30,     # 相邻块重叠长度
    separators=["\n\n", "\n", "。", "，", " ", ""],   # 优先按段落切，再按句子……
)
chunks = splitter.split_documents(docs)

print(f"2 篇文档 → {len(chunks)} 个文本块\n")
for c in chunks[:4]:
    print(f"[{len(c.page_content)} 字 | 来源: {c.metadata['source']}] {c.page_content[:50]}...")
    print("-" * 60)
```

## 2. 向量化与相似度检索

`InMemoryVectorStore` 是 LangChain 内置的内存向量库：零依赖、随建随用，最适合学习。把文本块存进去，检索时用**余弦相似度**找语义最接近的块。

```python
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embedding=embeddings)

# 入库（真实调用 embedding 接口，文档多时会稍慢）
vectorstore.add_documents(chunks)
print(f"已入库 {len(chunks)} 个文本块")

# 语义检索：注意问题里没有出现'年假'的字面说法，靠语义相似度命中
results = vectorstore.similarity_search("工作满一年能休几天？", k=2)
for d in results:
    print(f"[{d.metadata['source']}] {d.page_content[:60]}...")

# 带相似度分数的检索（分数越小越相似，是距离不是相似度）
scored = vectorstore.similarity_search_with_score("云盒子最多能传多大的文件？", k=1)
for d, score in scored:
    print(f"score={score:.3f} [{d.metadata['source']}] {d.page_content[:60]}...")
```

## 3. Retriever：统一的检索接口

`vectorstore.as_retriever()` 把向量库包装成 `Retriever`——它也是一个 `Runnable`（`invoke` 一个字符串，返回文档列表），所以能直接接进 LCEL 链。这是 RAG 链的接口标准。

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})   # 每次取 Top-2

docs = retriever.invoke("报销单据要多久内交？")
print(f"检索到 {len(docs)} 条")
for d in docs:
    print(f"- [{d.metadata['source']}] {d.page_content[:50]}...")
```

## 4. 组装 RAG 链

把三样东西接起来：
1. `retriever`：根据问题取相关文档
2. `format_docs`：把文档列表拼成提示词里的一段文字
3. `prompt | llm | parser`：让模型**只根据资料回答**

```
{"context": retriever | format_docs, "question": 原样透传}
        │  (RunnableParallel 并行准备两个字段)
        ▼
   prompt ── llm ── StrOutputParser
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"【来源: {d.metadata['source']}】\n{d.page_content}" for d in docs
    )

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是公司知识库助手。仅根据下面的资料回答问题；"
     "资料中没有的内容，直接回答'知识库中暂无相关信息'，不要编造。\n\n"
     "## 资料\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("入职三年能休几天年假？要怎么申请？"))
```

```python
# 测试"拒答"能力：问知识库里没有的问题
print(rag_chain.invoke("公司附近的健身房哪家便宜？"))
```

## 5. 带引用来源的 RAG

用户需要知道答案"出自哪里"。套路：用 `RunnableParallel` 同时保留检索到的文档列表和问题，答案生成后引用 metadata。

```python
from langchain_core.runnables import RunnableParallel

# 外层：检索原始文档 + 问题（供最终引用展示）
# 内层：answer 子链先把 docs 格式化成提示词需要的 context
rag_with_source = RunnableParallel(
    docs=retriever,
    question=RunnablePassthrough(),
).assign(
    answer=(
        {"context": lambda x: format_docs(x["docs"]),
         "question": lambda x: x["question"]}
        | rag_prompt | llm | StrOutputParser()
    )
)

result = rag_with_source.invoke("出差吃饭有补贴吗？标准是多少？")
print("答案:", result["answer"])
print()
print("引用来源:")
for d in result["docs"]:
    print(f"- {d.metadata['source']} : {d.page_content[:40]}...")
```

## 6. 从学习到生产：向量库选型

| 库 | 特点 | 适用 |
|---|---|---|
| `InMemoryVectorStore` | 零依赖，进程内 | 学习、原型 |
| Chroma | 轻量、可持久化到本地 | 个人项目、小团队 |
| FAISS | 高性能、Meta 出品 | 中等规模、单机 |
| Milvus / Qdrant / pgvector | 分布式/生产级 | 大规模、企业 |

换库只需换构造方式，`add_documents / as_retriever` 的用法完全一致——这就是 LangChain 抽象的价值。

---

> 📝 本笔记由 [chapter07-rag.ipynb](../chapter07-rag.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
