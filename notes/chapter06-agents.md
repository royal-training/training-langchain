# 第 6 章 智能体与记忆（Agent & Memory）

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
- 用 v1 的 `create_agent` 快速创建工具智能体
- 理解 Agent 的消息流转（比第 5 章的手写循环多了什么）
- 用 `checkpointer` + `thread_id` 给 Agent 加多轮记忆
- 掌握 `response_format` 让 Agent 输出结构化结果
- 建立"链 vs Agent"的选型判断

> **链 vs Agent**：链是固定的流程（你写好每一步），Agent 是动态的（模型自己决定下一步做什么）。简单流程用链，开放任务用 Agent。

## 准备：先定义几个练习工具

本章所有 Agent 共用这 3 个工具（伪数据，不产生真实副作用）：

```python
from langchain_core.tools import tool
from datetime import datetime

@tool
def calculator(expression: str) -> str:
    """计算一个四则运算表达式，如 '3 * (12 + 8)'。只支持数字和 + - * / ( )。"""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "错误：表达式包含不支持的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态。订单号形如 SO 开头。"""
    fake_orders = {
        "SO20260901": "已发货，物流显示运输中，预计明天送达",
        "SO20260815": "已签收",
    }
    return fake_orders.get(order_id, f"未找到订单 {order_id}")

@tool
def current_time() -> str:
    """获取当前的日期和时间。涉及'今天/现在'等时间问题时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, query_order, current_time]
print("工具注册完成:", [t.name for t in tools])
```

## 1. create_agent：三行代码一个 Agent

v1 版 LangChain 中，`create_agent` 从 `langchain.agents` 导入（旧版的 `create_react_agent` 已归入 `langchain_classic`）。它内部就是第 5 章的手写循环 + 更完善的调度（基于 LangGraph 实现）。

返回值是一个可编译执行的图（graph），用 `.invoke({"messages": [...]})` 触发。

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一位电商客服助手。回答问题前先判断是否需要调用工具；"
        "涉及计算务必用计算器工具，不要心算；回复简洁友好。"
    ),
)

result = agent.invoke({"messages": [{"role": "user", "content": "订单 SO20260901 到哪了？预计什么时候能到？"}]})
print(result["messages"][-1].content)
```

## 2. 观察 Agent 的消息流转

`result["messages"]` 是完整的消息轨迹：用户消息 → （带 tool_calls 的 AI 消息 → ToolMessage）× N 轮 → 最终 AI 回答。看懂这个序列，你就看懂了 Agent。

```python
result = agent.invoke({"messages": [{"role": "user", "content": "帮我算一下 3.8 折 4999 元的手机多少钱，再看看现在几点了"}]})

for m in result["messages"]:
    role = type(m).__name__
    if role == "AIMessage" and m.tool_calls:
        print(f"{role:12} | 调用工具: {[tc['name'] for tc in m.tool_calls]} 参数: {[tc['args'] for tc in m.tool_calls]}")
    elif role == "ToolMessage":
        print(f"{role:12} | 结果: {m.content}")
    else:
        print(f"{role:12} | {str(m.content)[:60]}")
```

## 3. 给 Agent 加记忆：checkpointer

Agent 默认每次 `invoke` 都是全新的（无状态）。给它一个 `checkpointer`（检查点存储器），并用 `thread_id` 标识会话，Agent 就能记住同一个会话里的所有历史——**这就是"多轮对话记忆"的全部实现**。

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

chat_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是电商客服助手，自然地记住用户在会话中告诉你的信息。",
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "user-001"}}   # 同一会话用同一 thread_id

# 第 1 轮：告诉它一些信息
r1 = chat_agent.invoke({"messages": [{"role": "user", "content": "你好，我是老王，我的订单号是 SO20260815"}]}, config)
print("Agent:", r1["messages"][-1].content)
print()

# 第 2 轮：不重复提供任何信息，看它是否记得
r2 = chat_agent.invoke({"messages": [{"role": "user", "content": "我的订单到哪了？"}]}, config)
print("Agent:", r2["messages"][-1].content)
```

```python
# 换一个 thread_id 就是另一个"人"的会话，互相隔离
config_b = {"configurable": {"thread_id": "user-002"}}
r3 = chat_agent.invoke({"messages": [{"role": "user", "content": "我的订单到哪了？"}]}, config_b)
print("新会话 Agent:", r3["messages"][-1].content)   # 它不认识这个用户

# 原会话的记忆仍在
r4 = chat_agent.invoke({"messages": [{"role": "user", "content": "我刚说我叫什么？"}]}, config)
print("原会话 Agent:", r4["messages"][-1].content)
```

> `InMemorySaver` 存在内存里，进程重启即丢失。生产环境可换用 `SqliteSaver`、`PostgresSaver` 等持久化实现（`langgraph-checkpoint-*` 系列包），**业务代码一行都不用改**——这就是 checkpointer 抽象的好处。

## 4. response_format：结构化收尾

让 Agent 在完成任务后，把最终结论整理成 Pydantic 对象。放在 `result["structured_response"]` 里。

```python
from pydantic import BaseModel, Field

class SupportReport(BaseModel):
    answer: str = Field(description="给用户的最终答复")
    tools_used: list[str] = Field(description="解决问题过程中用到的工具名")
    need_human: bool = Field(description="是否需要转人工")

report_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是客服助手，完成任务后按 schema 输出报告。",
    response_format=SupportReport,
)

res = report_agent.invoke({"messages": [{"role": "user", "content": "订单 SO20260901 大概还要多久？另外帮我算下 4999 打 8 折多少"}]})
print(res["structured_response"])
```

## 5. 链还是 Agent？选型速查

| 场景 | 选型 |
|---|---|
| 固定格式的翻译/摘要/分类 | 链（prompt \| llm \| parser） |
| 输入→固定几步加工→输出 | 链 + RunnableParallel/Lambda |
| 用户意图不定、可能要调工具、多轮交互 | Agent |
| 检索文档回答固定问题 | RAG 链（第 7 章） |
| 复杂多角色协作、审批流 | LangGraph（Agent 的进阶框架） |

经验法则：**先用最简单的链；只有当"步骤无法提前写死、需要模型临场决定"时才上 Agent。**

---

> 📝 本笔记由 [chapter06-agents.ipynb](../chapter06-agents.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
