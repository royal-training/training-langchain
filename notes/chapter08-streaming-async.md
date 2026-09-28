# 第 8 章 流式、异步与并发（Streaming & Async）

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
- 掌握 `stream`：让回答像 ChatGPT 一样"打字机"式输出
- 理解 `AIMessageChunk` 及其累加方式
- 整条 LCEL 链的流式（不是只有模型能流式）
- 会用 `asyncio` 并发调用多个 LLM 任务
- 会用 `astream_events` 观察链内部的事件流

> 流式决定**体验**（用户 2 秒内看到第一个字），异步决定**吞吐**（100 个请求并行处理）。两者都是生产应用必备。

## 1. stream：模型级流式

`llm.stream()` 返回一个生成器，逐段产出 `AIMessageChunk`。每个 chunk 的 `content` 只是增量片段，需要自己拼接。

```python
from langchain_core.messages import AIMessageChunk

print("打字机效果:")
final_chunk = None
for chunk in llm.stream("用三句话介绍什么是向量数据库"):
    print(chunk.content, end="", flush=True)
    final_chunk = chunk if final_chunk is None else final_chunk + chunk

print("\n\n--- 流结束后 ---")
print("累加后的完整内容类型:", type(final_chunk))
print("完整内容:", final_chunk.content)
```

> **chunk 对象可以相加**：`chunk1 + chunk2` 会把内容、tool_calls 等字段按规则合并，最后得到一个完整的 `AIMessageChunk`。需要"完整回复"做后续处理（如解析 JSON）时就是这么攒出来的。

## 2. 整条链的流式

LCEL 的杀手锏：只要链中每个环节都支持流式，`chain.stream()` 就是端到端流式——解析器会逐块透传，你拿到的就是增量文本，无需关心链里有多少环节。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是资深技术布道师，回答分 3 个要点，每点不超过 30 字"),
        ("human", "为什么{topic}值得学习？"),
    ])
    | llm
    | StrOutputParser()
)

for piece in chain.stream({"topic": "LangChain"}):
    print(piece, end="", flush=True)
```

## 3. batch 与 abatch：并发批处理

第 4 章已经用过 `batch`：传入输入列表，LangChain 自动并发执行（默认并发上限可配）。异步版 `abatch` 用法相同，适合已有事件循环的场景（如 FastAPI 服务里）。

```python
import time

questions = [
    {"topic": "向量数据库"}, {"topic": "微服务"}, {"topic": "函数式编程"},
]

t0 = time.time()
results = chain.batch(questions)
t1 = time.time()
print(f"batch 并发 3 个请求耗时 {t1 - t0:.1f}s（顺序执行约需 3 倍单次耗时）")
for r in results:
    print("•", r[:40], "...")
```

## 4. 异步基础：ainvoke / astream

Notebook（IPython）顶层直接支持 `await`。所有 Runnable 都有 `a` 前缀的异步版本：`ainvoke / astream / abatch`。

```python
import asyncio

# 异步流式
async def demo_astream():
    async for piece in chain.astream({"topic": "RAG"}):
        print(piece, end="", flush=True)

await demo_astream()
```

```python
# 并发调度多个异步任务：asyncio.gather
import time

async def answer(topic: str) -> str:
    return await chain.ainvoke({"topic": topic})

async def demo_concurrent():
    t0 = time.time()
    tasks = [answer(t) for t in ["并发", "缓存", "异步 IO", "消息队列"]]
    results = await asyncio.gather(*tasks)     # 4 个请求同时发出
    print(f"并发 4 个请求总耗时 {time.time() - t0:.1f}s\n")
    for r in results:
        print("•", r[:40], "...")

await demo_concurrent()
```

> **asyncio.gather 是异步并发的核心**：把多个协程打包同时执行，全部完成后按顺序返回结果列表。配合 `asyncio.Semaphore` 可限制并发数，防止触发 API 限流。

## 5. astream_events：观察链的"事件流"

比 `astream` 更底层的观测手段：链执行过程中每个环节（prompt、chat model、parser）都会发出事件。做调试、做 Web 应用的事件推送（SSE）都靠它。

```python
async def demo_events():
    async for event in chain.astream_events({"topic": "Agent"}, version="v2"):
        if event["event"] == "on_chat_model_stream":        # 模型吐字的每个 chunk
            print(event["data"]["chunk"].content, end="", flush=True)
        elif event["event"] == "on_chain_start":
            print(f"▶ 链开始: {event['name']}")
        elif event["event"] == "on_chain_end":
            print(f"\n✔ 链结束: {event['name']}")

await demo_events()
```

常用事件：
| 事件 | 含义 |
|---|---|
| `on_chain_start` / `on_chain_end` | 某个 Runnable（含整条链）开始/结束 |
| `on_chat_model_start` | 模型开始 |
| `on_chat_model_stream` | 模型流式吐出一个 chunk |
| `on_chat_model_end` | 模型完成（data 里有完整输出） |

---

> 📝 本笔记由 [chapter08-streaming-async.ipynb](../chapter08-streaming-async.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
