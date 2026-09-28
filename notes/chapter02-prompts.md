# 第 2 章 提示词与消息（Prompts & Messages）

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
- 理解 LangChain 中 4 种核心消息类型及其在对话中的作用
- 熟练使用 `ChatPromptTemplate` 编写带变量的提示词模板
- 掌握 `MessagesPlaceholder` 插入对话历史 / 少样本示例
- 会用 few-shot（少样本）提示显著改变模型行为
- 理解 `prompt | llm | parser` 这种"链"的雏形（第 4 章展开）

> 前置条件：已完成 chapter01，确保 `.env` 中配置了 `API_KEY_ZHIPU` 和 `API_URL_ZHIPU`。

## 1. 消息类型（Messages）

LangChain 中一次对话就是 `BaseMessage` 列表，最常用的 4 种：

| 类型 | 角色 | 用途 |
|---|---|---|
| `SystemMessage` | system | 设定身份、规则、风格，通常放最前面且只有一条 |
| `HumanMessage` | user | 用户输入 |
| `AIMessage` | assistant | 模型的历史回复（多轮对话时回传给模型） |
| `ToolMessage` | tool | 工具执行结果（第 5 章详解） |

> **关键点**：模型本身是**无状态**的。所谓"记住上下文"，就是每次请求时把历史消息一起发过去。

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 多轮对话：把历史消息原样带上，模型就知道上下文
messages = [
    SystemMessage(content="你是一位面试官，语气专业但友善"),
    HumanMessage(content="你好，我想应聘 Python 后端岗位"),
    AIMessage(content="你好！很高兴认识你。请先介绍一下你做过的最有挑战的项目吧。"),
    HumanMessage(content="我做过一个日均百万请求的推荐服务"),
]

response = llm.invoke(messages)
print(response.content)
```

`response` 是一个 `AIMessage` 对象，除了 `content` 还有几个常用属性：
- `response_metadata`：模型名、finish_reason、token 用量等
- `usage_metadata`：input_tokens / output_tokens（计费参考）

下面观察一下：

```python
print("content 类型:", type(response.content))
print("回复长度:", len(response.content))
print("usage:", response.usage_metadata)
print("finish_reason:", response.response_metadata.get("finish_reason"))
```

## 2. ChatPromptTemplate：提示词模板

把固定的话术和可变的"变量"分离，是工程化使用 LLM 的第一步。

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),   # ("角色", "内容模板")
    ("human", "..."),
])
```

模板中用 `{变量名}` 占位，调用时用字典传值。

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{domain}领域的资深专家，用{style}的风格回答问题。"),
    ("human", "{question}"),
])

# 两种常用用法：

# 用法 1：format_messages() —— 只做格式化，得到消息列表，再手动传给模型
msgs = prompt.format_messages(
    domain="数据库", style="深入浅出", question="索引为什么能加快查询？"
)
print(msgs)

# 用法 2：invoke() —— 返回 ChatPromptValue，可直接接在链后面
prompt_value = prompt.invoke({
    "domain": "数据库",
    "style": "深入浅出",
    "question": "索引为什么能加快查询？",
})
print(type(prompt_value))
response = llm.invoke(prompt_value.to_messages())
print(response.content[:100])
```

```python
# 更常见的写法：模板直接接上模型和输出解析器，用 | 串成一条链
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{domain}领域的资深专家，用{style}的风格回答问题。"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()   # StrOutputParser 把 AIMessage 简化为纯字符串

result = chain.invoke({
    "domain": "数据库",
    "style": "深入浅出",
    "question": "索引为什么能加快查询？",
})
print(result)
```

> **`{}` 的坑**：模板里如果真的需要输出花括号本身（比如 JSON 示例），要写成双花括号 `{{` `}}`，否则会被当成变量。

## 3. MessagesPlaceholder：插入消息列表

当变量是一个**消息列表**（历史对话、示例）而不是字符串时，用 `MessagesPlaceholder`。它是实现"多轮对话"和"few-shot"的标配。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手 {name}，请保持{style}的语气。"),
    MessagesPlaceholder("history"),      # 这里会被展开成任意条历史消息
    ("human", "{question}"),
])

history = [
    HumanMessage(content="我叫小明"),
    AIMessage(content="你好，小明！很高兴见到你。"),
]

response = llm.invoke(prompt.invoke({
    "name": "小智",
    "style": "幽默",
    "history": history,
    "question": "你还记得我叫什么吗？",
}))
print(response.content)
```

`MessagesPlaceholder` 支持设置 `optional=True`：当调用时不传 `history` 也不会报错，非常适合"首轮对话没有历史"的场景。

```python
MessagesPlaceholder("history", optional=True)
```

## 4. Few-shot 少样本提示

在提示里给模型几个"输入→输出"示例，能显著提升分类、格式转换等任务的准确率。用 `FewShotChatMessagePromptTemplate` 可以把示例统一管理。

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 准备示例
examples = [
    {"input": "今天天气真差，心情都坏了", "output": "负面"},
    {"input": "服务态度特别好，下次还来", "output": "正面"},
    {"input": "东西一般般，凑合能用", "output": "中性"},
]

# 2. 单条示例长什么样
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 3. 组装 few-shot 模板：自动把 examples 逐条套进 example_prompt
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 最终提示 = 系统设定 + few-shot 示例 + 用户输入
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是评论情感分类器。只输出：正面、负面 或 中性 之一，不要输出其他内容。"),
    few_shot,
    ("human", "{input}"),
])

classifier = final_prompt | llm | StrOutputParser()

for text in ["物流速度快但是包装破损了", "性价比无敌，闭眼入", "跟描述基本一致，无惊喜也无雷"]:
    print(text, "->", classifier.invoke({"input": text}))
```

## 5. partial：模板的部分变量先固定

`partial()` 可以提前固定一部分变量（如当前日期、格式要求），剩下的变量留到调用时再传。

```python
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages([
    ("system", "今天是 {date}。你是 {role}。回答时请结合当前日期考虑时效性。"),
    ("human", "{question}"),
]).partial(date=today)   # date 变量已固定，之后只需传 role 和 question

chain = prompt | llm | StrOutputParser()
print(chain.invoke({"role": "旅行顾问", "question": "下周去三亚穿什么合适？"}))
```

---

> 📝 本笔记由 [chapter02-prompts.ipynb](../chapter02-prompts.ipynb) 提炼，完整练习与参考答案请运行对应 notebook。
