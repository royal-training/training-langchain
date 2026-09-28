# 第 4 章 LCEL：把组件串成链（LangChain Expression Language）

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
- 理解 LCEL 的核心：`|` 管道 + 所有组件都是 `Runnable`
- 掌握统一执行接口：`invoke / batch / stream / ainvoke`
- 会用 `RunnableLambda` 把自定义函数接入链
- 会用 `RunnableParallel` 并行分支、`RunnablePassthrough.assign` 透传扩展
- 会做条件路由、重试与降级（fallback）

> LCEL 是 LangChain 的"乐高积木语言"。第 2、3 章里已经零星用过 `prompt | llm | parser`，本章把它讲透。

## 1. Runnable：一切皆可管道

LCEL 的约定：
- 每个组件（prompt、llm、parser、retriever、自定义函数……）都实现了 `Runnable` 协议
- `|` 把它们串成 `RunnableSequence`：前一步的输出 = 后一步的输入
- 整条链本身也是 `Runnable`，因此自动拥有统一接口：

| 方法 | 作用 |
|---|---|
| `invoke` | 跑一次 |
| `batch` | 并发跑多次输入 |
| `stream` | 流式输出（第 8 章展开） |
| `ainvoke` / `abatch` | 异步版本 |

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是营销文案专家，输出不超过 50 字"),
    ("human", "为产品写一句广告语：{product}"),
])
chain = prompt | llm | StrOutputParser()

# invoke：单次调用
print("invoke:", chain.invoke({"product": "无线降噪耳机"}))

# batch：并发处理多个输入（一条语句，自动并发）
results = chain.batch([
    {"product": "儿童护眼台灯"},
    {"product": "电动牙刷"},
])
for r in results:
    print("batch:", r)
```

## 2. RunnableLambda：把任意函数接入链

链里的环节不只能是官方组件，用 `RunnableLambda` 可以把任何 Python 函数变成链的一环。

两种传参形态：
- 函数只接收一个参数 → 接收上一步输出
- 需要多个输入 → 上一步传 dict，函数里取

```python
from langchain_core.runnables import RunnableLambda

def make_report(d: dict) -> str:
    # 上一步传来了 dict，这里自己取字段
    return f"【{d['title']}】共 {d['chars']} 字"

chain = (
    RunnableLambda(lambda x: {"title": x})          # str -> dict
    | {"title": RunnableLambda(lambda d: d["title"]),  # RunnableParallel 的字典语法糖
       "chars": RunnableLambda(lambda d: len(d["title"]))}
    | RunnableLambda(make_report)
)

print(chain.invoke("使用 LCEL 组装你的 AI 应用"))
```

### 字典语法糖

上面的中间一步 `{"title": ..., "chars": ...}` 用了 LCEL 的语法糖：**一个 dict 也是 Runnable**，它的每个 value 会并行执行，最后聚合成 dict。这与 `RunnableParallel` 等价，是最常用的写法。

## 3. RunnableParallel：显式并行分支

当每个分支要调用**不同的 LLM 请求**时，并行能省总耗时。

```python
from langchain_core.runnables import RunnableParallel

joke_chain = ChatPromptTemplate.from_messages([
    ("system", "你是段子手"), ("human", "用一句话讲一个关于{topic}的冷笑话")
]) | llm | StrOutputParser()

poem_chain = ChatPromptTemplate.from_messages([
    ("system", "你是诗人"), ("human", "为{topic}写一句七言诗")
]) | llm | StrOutputParser()

parallel = RunnableParallel(joke=joke_chain, poem=poem_chain)

# 两个请求同时发出，总耗时≈最慢的那个，而不是相加
result = parallel.invoke({"topic": "程序员"})
print("冷笑话:", result["joke"])
print("七言诗:", result["poem"])
```

## 4. RunnablePassthrough：透传与 assign 扩展

`RunnablePassthrough()` 原样传递输入；它的 `.assign(**kwargs)` 在**保留原有字段**的基础上**新增字段**——这是组装"既要原始输入、又要加工结果"场景的利器。

```python
from langchain_core.runnables import RunnablePassthrough

add_summary = RunnablePassthrough.assign(
    summary=(ChatPromptTemplate.from_messages([
        ("system", "用不超过 20 字概括输入内容"),
        ("human", "{text}"),
    ]) | llm | StrOutputParser()),
    word_count=lambda x: len(x["text"]),
)

result = add_summary.invoke({"text": "LangChain 是一个用于开发大模型应用的开源框架，提供了模型调用、提示词管理、检索增强、智能体等能力，是目前 AI 应用开发的事实标准之一。"})
print(result)  # 原来的 text 保留，新增 summary 和 word_count
```

## 5. 条件路由：让链学会"分诊"

`RunnableLambda` 里写 if/else 返回**不同的下游链**，就实现了路由。下面根据问题类型选择不同的回答链。

```python
from langchain_core.runnables import RunnableLambda

tech_chain = (ChatPromptTemplate.from_messages([
    ("system", "你是资深工程师，回答要带代码示例"), ("human", "{q}")
]) | llm | StrOutputParser())

common_chain = (ChatPromptTemplate.from_messages([
    ("system", "你是聊天伙伴，回答轻松简短"), ("human", "{q}")
]) | llm | StrOutputParser())

def route(d: dict):
    # 简单规则路由；也可以先用 LLM 做意图分类（第 3 章技巧）再路由
    keywords = ("代码", "报错", "bug", "python", "实现")
    if any(k in d["question"].lower() for k in keywords):
        return tech_chain
    return common_chain

router_chain = RunnableLambda(route)

for q in ["Python 里怎么读取 CSV 文件？", "今天适合摸鱼吗？"]:
    print(f"Q: {q}")
    print(f"A: {router_chain.invoke({'question': q})[:80]}...")
    print("-" * 40)
```

## 6. 重试与降级

LLM 服务偶发超时/限流，生产代码必须考虑容错：
- `.with_retry()`：失败自动重试
- `.with_fallbacks([...])`：主链失败后换备用链
- 两者可以叠加

```python
# 先用纯函数离线理解 fallback 机制（不花 API 费用）
from langchain_core.runnables import RunnableLambda

call_count = {"n": 0}

def flaky_step(x: str) -> str:
    call_count["n"] += 1
    if call_count["n"] < 3:          # 前两次故意失败
        raise ValueError("服务暂时不可用")
    return f"第 {call_count['n']} 次调用成功，输入是: {x}"

def backup_step(x: str) -> str:
    return "备用服务响应"

resilient = RunnableLambda(flaky_step).with_retry(stop_after_attempt=2).with_fallbacks(
    [RunnableLambda(backup_step)]
)
print(resilient.invoke("hello"))
# 把 with_retry(stop_after_attempt=2) 改成 3，观察重试成功的效果
```

```python
# 真实模型场景：把"主模型"换成错误地址，fallback 换成正常 llm
from langchain_openai import ChatOpenAI

bad_llm = ChatOpenAI(
    model="glm-4.5-air",
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url="http://127.0.0.1:9",   # 一个必然连不上的地址
)

fallback_chain = bad_llm.with_fallbacks([llm]) | StrOutputParser()
print(fallback_chain.invoke("用一句话介绍 fallback 的作用"))
```

---

> 📝 本笔记由 [chapter04-lcel.ipynb](../chapter04-lcel.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
