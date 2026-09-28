# 第 3 章 结构化输出（Structured Output）

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
- 掌握 `with_structured_output()`：让模型直接返回 Pydantic 对象
- 会定义枚举、可选字段、嵌套模型、列表抽取
- 了解不依赖工具调用的备选方案：`JsonOutputParser`
- 能搭建"抽取 → 代码处理 → 再生成"的完整 pipeline

> 让 LLM 输出**程序可处理的结构**（而不是自由文本），是大多数真实应用的第一步：信息抽取、分类打标、路由分发、Agent 参数传递都依赖它。

## 1. with_structured_output：基本用法

核心套路三步：
1. 用 `pydantic.BaseModel` 定义想要的数据结构（**字段描述写得越清楚，抽取越准**）
2. `llm.with_structured_output(MyModel)` 得到一个"抽取器"
3. `invoke()` 任意文本，直接得到填充好的 Pydantic 实例

底层原理：模型被绑定了 schema，因此能保证输出符合 JSON Schema。

> **`method="function_calling"` 必须显式指定**：langchain-openai 默认走 OpenAI 原生的 `json_schema` 模式（`response_format` 参数），智谱等 OpenAI 兼容接口并不支持它（模型会无视该参数返回自由文本，然后解析报错）；`function_calling` 模式把 schema 包装成一个工具强制调用，兼容性最好。

```python
from pydantic import BaseModel, Field

# 1. 定义数据结构：每个字段都用 description 说明抽取要求
class MovieReview(BaseModel):
    movie_name: str = Field(description="评论中提到的电影名")
    sentiment: str = Field(description="情感倾向，只能是：正面、负面 或 中性")
    rating: int = Field(description="从评论推断的评分（1-10），如果没有明确提到则为 -1")
    key_points: list[str] = Field(description="评论者的主要观点，2-4 条")


# 2. 创建结构化抽取器
extractor = llm.with_structured_output(MovieReview, method="function_calling")

# 3. 抽取 —— 返回的就是 MovieReview 实例，不再是字符串
review = extractor.invoke(
    "看完《流浪地球3》了，特效震撼剧情流畅，就是结尾有点仓促。整体我打 8 分。"
)
print(type(review))
print(review)
```

```python
# 抽取结果就是普通 Python 对象，可以直接点出字段、参与逻辑
print("电影:", review.movie_name)
print("情感:", review.sentiment)

if review.rating >= 7:
    print("✅ 可以放进推荐列表")
else:
    print("❌ 不推荐")
```

## 2. 枚举、可选字段与嵌套模型

真实业务里的 schema 往往更复杂：
- **枚举**：用 `Literal[...]` 限定取值，比"在 description 里写只能取哪些值"更可靠
- **可选字段**：`Optional[...]` + `default=None`，模型没抽取到时返回 None 而不是硬编一个值
- **嵌套模型**：模型里套模型，表达层级结构

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Address(BaseModel):
    city: str = Field(description="城市名")
    detail: Optional[str] = Field(default=None, description="详细地址，未提到则为 None")


class Customer(BaseModel):
    name: str = Field(description="客户姓名")
    vip_level: Literal["普通", "黄金", "钻石"] = Field(description="VIP 等级")
    age: Optional[int] = Field(default=None, description="年龄，未提到则为 None")
    address: Optional[Address] = Field(default=None, description="地址，未提到则为 None")
    tags: list[str] = Field(default_factory=list, description="给客户打的标签，如：活跃、高消费")


extractor = llm.with_structured_output(Customer, method="function_calling")

c = extractor.invoke("王芳是我们的钻石会员，住在上海浦东新区张江路100号，平时购物很频繁")
print(c)
print(c.vip_level, type(c.vip_level))  # Literal 枚举抽取后就是普通 str
```

## 3. 列表抽取：从一段文字抽取多个实体

把字段类型定义为 `list[Model]`，就能一次抽出多个实体。

```python
class Book(BaseModel):
    title: str = Field(description="书名")
    author: Optional[str] = Field(default=None, description="作者")
    year: Optional[int] = Field(default=None, description="出版年份")


class BookList(BaseModel):
    books: list[Book] = Field(description="文本中提到的所有书籍")


book_extractor = llm.with_structured_output(BookList, method="function_calling")

result = book_extractor.invoke(
    "书架上摆着《三体》，刘慈欣2008年写的；旁边是《人类简史》，作者赫拉利；"
    "还有一本掉了封皮的《红楼梦》，谁写的来着？"
)
for b in result.books:
    print(f"- 《{b.title}》 {b.author} {b.year}")
```

## 4. include_raw：同时拿到原始响应

`with_structured_output(Schema, include_raw=True)` 会返回一个字典：
- `raw`：原始 `AIMessage`（含 token 用量等）
- `parsed`：解析后的 Pydantic 对象
- `parsing_error`：解析失败时的错误信息

调试 schema 和排查抽取质量问题时非常有用。

```python
res = llm.with_structured_output(MovieReview, include_raw=True, method="function_calling").invoke(
    "《星际穿越》是神作，9 分不解释。"
)
print("parsed:", res["parsed"])
print("raw 消息类型:", type(res["raw"]))
print("解析错误:", res["parsing_error"])
```

## 5. 备选方案：JsonOutputParser

`with_structured_output` 依赖模型的工具调用能力。如果换用的模型/接口不支持，可以退回到"在提示词里描述 JSON 格式 + 解析器容错解析"的经典路线：

1. `JsonOutputParser` 自动生成格式说明（`format_instructions`）
2. 把说明塞进 system 提示
3. 输出交给解析器，它能容忍 ```json 代码围栏等常见噪音

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser(pydantic_object=Customer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "从用户输入中抽取客户信息，严格按格式输出 JSON。\n{format_instructions}"),
    ("human", "{text}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser   # 链的最后一个环节是解析器

info = chain.invoke({"text": "客户李雷是黄金会员，北京人，26岁，是个游戏爱好者"})
print(type(info))
print(info)
```

## 6. 实战 Pipeline：抽取 → 代码处理 → 再生成

结构化输出的典型用法：**用代码做确定性逻辑，把"判断"留给代码、把"生成"留给模型**。

流程：客服工单 → 抽取结构化字段 → 代码按规则定优先级 → 模型根据优先级起草回复。

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Ticket(BaseModel):
    category: Literal["账务问题", "技术故障", "功能咨询", "投诉"] = Field(description="工单类型")
    urgency: int = Field(description="紧急程度 1-5，5 最紧急")
    summary: str = Field(description="一句话概括问题")


ticket_extractor = llm.with_structured_output(Ticket)

raw_ticket = "我上周被重复扣了两次会员费，都三天了还没退！再不处理我要投诉到消协了！"
ticket = ticket_extractor.invoke(raw_ticket)
print("抽取结果:", ticket)

# 代码规则：投诉 + 高紧急 = P0
priority = "P0" if (ticket.category == "投诉" or ticket.urgency >= 4) else "P1"
print("定级:", priority)

reply_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是客服专员。当前工单优先级为 {priority}。"
     "P0 需要在回复中明确承诺 24 小时内解决并致歉；其他级别语气可平和一些。"),
    ("human", "工单内容：{content}\n问题概要：{summary}"),
])
reply = reply_prompt | llm | StrOutputParser()

print(reply.invoke({"priority": priority, "content": raw_ticket, "summary": ticket.summary}))
```

---

> 📝 本笔记由 [chapter03-structured-output.ipynb](../chapter03-structured-output.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
