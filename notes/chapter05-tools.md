# 第 5 章 工具与函数调用（Tools）

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
- 用 `@tool` 把普通 Python 函数变成 LLM 可调用的工具
- 理解工具调用的本质：**模型只"点名"，代码来执行**
- 掌握 `bind_tools` → `tool_calls` → `ToolMessage` 回传的完整闭环
- 会写多轮工具循环（这是 Agent 的底层原理）
- 会用 Pydantic 给工具定义精确的参数 schema

> 工具（Tool）让 LLM 突破"只能说话"的限制：查数据库、调 API、算数学、操作系统……它是 Agent 的基石。

## 1. 定义工具：@tool 装饰器

一个普通函数 + `@tool` 就成了工具。三个要素决定模型"会不会用、用得对不对"：
1. **函数名**：模型的调用凭据，命名要动词化、语义清晰
2. **docstring**：模型决定何时调用此工具的唯一依据，必须写清楚"干什么、什么时候用"
3. **类型注解**：每个参数的类型和含义会自动生成参数 schema

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。当需要乘法运算时使用。"""
    return a * b

@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气情况。当用户询问天气时使用。"""
    # 演示用假数据；真实场景这里会调用天气 API
    fake_db = {"北京": "晴，12°C，西北风 3 级", "上海": "多云，18°C", "广州": "小雨，25°C"}
    return fake_db.get(city, f"{city}：阴，15°C")

@tool
def get_word_length(word: str) -> int:
    """统计一个单词的字符数。"""
    return len(word)

print("name:", multiply.name)
print("description:", multiply.description)
print("args schema:", multiply.args_schema.model_json_schema()["properties"])
print("直接调用（绕过模型）:", multiply.invoke({"a": 6, "b": 7}))
```

> **最佳实践**：docstring 用中文写也完全可以（模型能理解），但描述要具体到"什么情况下用"，避免模型误判调用时机。

## 2. bind_tools：把工具交给模型

`bind_tools` 把工具的 schema 绑定到模型上。此后每次请求，模型都会"看到"这些工具，并可能返回 `tool_calls`。

**关键认知**：模型**不会执行**你的函数，它只返回一个"我想调用 multiply，参数是 a=37, b=45"的请求。执行永远发生在你的代码里。

```python
llm_with_tools = llm.bind_tools([multiply, get_weather, get_word_length])

response = llm_with_tools.invoke("请帮我算一下 37 乘以 45 等于多少")

print("content:", response.content)          # 通常为空或"我去算一下"
print("tool_calls:", response.tool_calls)    # 模型真正想调用的工具
```

`tool_calls` 是一个列表（模型可能一次点名多个工具），每项包含：
- `name`：工具名
- `args`：模型生成的参数字典
- `id`：本次调用的唯一 ID，回传结果时要用

## 3. 完整闭环：执行工具并回传结果

一轮工具调用分四步：
1. `invoke` → 模型返回带 `tool_calls` 的 AIMessage
2. 你的代码找到对应函数，**真的执行**
3. 把结果包成 `ToolMessage`（带上 `tool_call_id`）追加到消息列表
4. 再次 `invoke` → 模型结合工具结果生成最终回答

> **必须追加历史消息**（包括那条带 tool_calls 的 AIMessage 本身），模型才知道"我刚才调了什么"。

```python
from langchain_core.messages import HumanMessage, ToolMessage

tools_by_name = {t.name: t for t in [multiply, get_weather, get_word_length]}

messages = [HumanMessage(content="37 乘以 45 等于多少？")]

# 第 1 步：模型点名
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
print("模型想调用:", ai_msg.tool_calls)

# 第 2 步：代码执行
tool_call = ai_msg.tool_calls[0]
result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
print("工具执行结果:", result)

# 第 3 步：结果回传
messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# 第 4 步：模型总结
final = llm_with_tools.invoke(messages)
print("最终回答:", final.content)
```

## 4. 手写工具循环：Agent 的引擎盖下面

真实问题往往要连续调用多个工具。写一个循环，就是**一个最小的 Agent**（第 6 章的 `create_agent` 内部核心就是它）：

```
while 模型还想调用工具:
    执行所有 tool_calls，把 ToolMessage 追加进历史
    再次调用模型
```

```python
def run_agent(question: str, max_turns: int = 5) -> str:
    messages = [HumanMessage(content=question)]
    for _ in range(max_turns):                     # 防止死循环的保险丝
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:                  # 模型不再点名 → 结束
            return ai_msg.content
        for tc in ai_msg.tool_calls:               # 执行模型点名的每一个工具
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return "达到最大轮数限制"

print(run_agent("先算 37*45，再算结果乘 2，最后告诉我北京的天气怎么样"))
print()
print("中间过程（模型每一轮说了什么/调了什么）:")
messages = [HumanMessage(content="先算 37*45，再算结果乘 2")]
for _ in range(5):
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    if not ai_msg.tool_calls:
        break
    print(f"  [轮次] 模型调用: {[tc['name'] for tc in ai_msg.tool_calls]}")
    for tc in ai_msg.tool_calls:
        messages.append(ToolMessage(
            content=str(tools_by_name[tc["name"]].invoke(tc["args"])),
            tool_call_id=tc["id"],
        ))
```

## 5. 精确的参数 schema：Pydantic args_schema

参数复杂时（多个字段、强约束、需要字段级说明），用 Pydantic 模型定义 `args_schema`，字段描述会被完整传给模型，大幅提升参数质量。

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class TrainTicketArgs(BaseModel):
    departure: str = Field(description="出发城市，如：北京")
    destination: str = Field(description="到达城市，如：上海")
    date: str = Field(description="乘车日期，格式 YYYY-MM-DD")
    seat_class: Literal["二等座", "一等座", "商务座"] = Field(
        default="二等座", description="座位等级，用户没说则默认二等座"
    )
    passenger: Optional[str] = Field(default=None, description="乘车人姓名")

@tool(args_schema=TrainTicketArgs)
def book_train_ticket(departure: str, destination: str, date: str,
                      seat_class: str = "二等座", passenger: Optional[str] = None) -> str:
    """预订火车票。当用户明确想订票时调用。"""
    who = passenger or "乘客"
    return f"✅ 已为 {who} 预订 {date} {departure}→{destination} {seat_class}（演示，未真实下单）"

llm_booking = llm.bind_tools([book_train_ticket])
msg = llm_booking.invoke("帮我订明天从北京到上海的高铁，一等座，乘车人张三")
print(msg.tool_calls)   # 观察模型自动填充的字段、枚举和默认值
```

## 6. ToolNode：现成的"工具执行器"（了解即可）

第 4 步手写循环里"执行工具 + 包装 ToolMessage"这部分很模板化。LangGraph 提供了 `langgraph.prebuilt.ToolNode` 来替你完成：输入含 `tool_calls` 的消息，自动并行执行所有工具并返回 `ToolMessage` 列表。

注意：v1 版的 `ToolNode` 需要在 LangGraph 的图运行时中工作，不推荐像旧版那样脱离图单独 `invoke`。**你不需要手写它**——下一章的 `create_agent` 内部已经内置了这套调度。这里只需记住它的位置和作用，第 6 章会直接用到它的产物。

---

> 📝 本笔记由 [chapter05-tools.ipynb](../chapter05-tools.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
