# LangChain 初始化大模型的几种方式

> 环境基准：langchain 1.3.17 / langchain-openai 1.6.0 / langchain-community 0.4.2（Python 3.13）
> 示例统一用智谱 GLM（`glm-4.5-air`）的 OpenAI 兼容接口，`.env` 提供 `API_KEY_ZHIPU` 与 `API_URL_ZHIPU`。

---

## 总览对比

| 方式 | 导入来源 | 适用场景 | 特点 |
|---|---|---|---|
| ① ChatOpenAI + 兼容接口 | `langchain_openai` | 任何提供 OpenAI 兼容端点的模型 | 最通用，一套参数走天下 |
| ② 厂商专属 ChatModel 类 | `langchain_community` / partner 包 | 想用厂商特有参数 | 每个 provider 一个类，生态分散 |
| ③ init_chat_model 工厂 | `langchain.chat_models` | 模型名来自配置、需运行时切换 | 字符串驱动，代码与 provider 解耦 |
| ④ 环境变量默认配置 | 同上 | 本地快速试验 | 不传参数，靠约定（`OPENAI_API_KEY` 等） |

---

## 方式一：ChatOpenAI + OpenAI 兼容接口（最通用，本项目主力）

国内外主流厂商（智谱、DeepSeek、月之暗面、通义）以及本地部署（vLLM / Ollama）几乎都提供 OpenAI 兼容端点，因此 `ChatOpenAI` 改个 `base_url` 就能连任意家：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.6,
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url=os.getenv("API_URL_ZHIPU"),   # 指向 https://open.bigmodel.cn/api/paas/v4/
)

print(llm.invoke("你好").content)
```

要点：

- `api_key` / `base_url` 是标准参数名；`openai_api_key` / `openai_api_base` 是等价别名（老教程常见），二者选一即可。
- 其他常用参数：`max_tokens`、`timeout`、`retries`、`default_headers`（部分厂商要求自定义 header）。
- 优点：只学一个类；换模型只改 `model` + `base_url`。
- 缺点：厂商独有能力（如智谱特有的 `thinking` 参数）没有暴露，需要时得靠 `extra_body` 或改用方式②。

## 方式二：厂商专属 ChatModel 类

每个厂商可以有专属集成类，能用到厂商特有参数：

```python
import os
from dotenv import load_dotenv
from langchain_community.chat_models.zhipuai import ChatZhipuAI

load_dotenv()

llm = ChatZhipuAI(
    model="glm-4.5-air",
    api_key=os.getenv("API_KEY_ZHIPU"),
)
print(llm.invoke("你好").content)
```

常见厂商类的来源（LangChain 1.x）：

| 模型 | 类 | 来源包 |
|---|---|---|
| OpenAI / 兼容端点 | `ChatOpenAI` | `langchain-openai`（partner 包） |
| Anthropic | `ChatAnthropic` | `langchain-anthropic`（partner 包） |
| DeepSeek | `ChatDeepSeek` | `langchain-deepseek`（partner 包） |
| 智谱 | `ChatZhipuAI` | `langchain-community`（社区包） |
| 通义千问 | `ChatTongyi` | `langchain-community`（社区包） |

要点：

- **partner 包**（`langchain-openai` 等）由官方重点维护，质量高于社区包；智谱这类在 `langchain-community` 里，更新滞后。
- 缺点：import 路径、参数名各家不同，切换模型要改代码；本项目实测社区版 `ChatZhipuAI` 对 LangChain 1.x 新特性支持不全，所以练习册只作了解。

## 方式三：init_chat_model 工厂函数（推荐做可切换应用）

`init_chat_model` 按 `"provider:model_name"` 字符串自动选择并实例化对应的 ChatModel 类，返回的就是标准实例（如 `ChatOpenAI`）：

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    "openai:glm-4.5-air",                     # provider 前缀:模型名
    api_key=os.getenv("API_KEY_ZHIPU"),
    base_url=os.getenv("API_URL_ZHIPU"),  # 必传！否则默认连 api.openai.com 会 ConnectTimeout
    temperature=0.6,
)
print(type(llm))   # <class 'langchain_openai.chat_models.base.ChatOpenAI'>
```

三种用法形态：

```python
# ① 前缀形式（最常用）
llm = init_chat_model("openai:glm-4.5-air", api_key=..., base_url=...)

# ② 分开传参（等价）
llm = init_chat_model(model="glm-4.5-air", model_provider="openai", api_key=..., base_url=...)

# ③ 部分实例化：先声明、后补全（工程化核心卖点，适合配置化/写库）
placeholder = init_chat_model(temperature=0.6)
llm = placeholder.bind(model="openai:glm-4.5-air", api_key=..., base_url=...)
```

要点：

- 内置 28 个 provider（`openai`、`anthropic`、`deepseek`、`google_genai`、`ollama`、`groq`、`xai` 等），写错会报 `ValueError` 并列出全部可选值。
- 需要安装对应集成包（`openai:` 要求 `langchain-openai`），缺失报 `ImportError`。
- **`glm-*` 不在自动推断列表**，智谱没有专属 provider，必须显式写 `openai:` 前缀走兼容路线（moonshot、qwen 等同理）。
- 参数（`temperature`、`base_url`、`max_tokens` 等）原样透传给底层类，返回对象无阉割，`bind_tools`、`with_structured_output` 等全部可用。
- 适用：模型名放配置文件/命令行，同一套代码今天跑 GLM、明天跑 DeepSeek 做对比。

## 方式四：环境变量默认配置（约定优于配置）

各集成类定义了默认环境变量，不传参数也能初始化，适合本地快速试验：

```python
# export OPENAI_API_KEY=sk-...  且可选 export OPENAI_BASE_URL=...
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="glm-4.5-air", base_url=os.getenv("API_URL_ZHIPU"))
```

常见默认变量：`ChatOpenAI` → `OPENAI_API_KEY` / `OPENAI_BASE_URL`；`ChatZhipuAI` → `ZHIPUAI_API_KEY`。生产代码建议仍显式传参或统一从配置读，避免隐式依赖。

## 附：调试用假模型（不花钱）

写测试/离线验证链路时用 langchain 自带的假模型，不发起真实 API 调用：

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

llm = GenericFakeChatModel(messages=iter(["这是固定回复"]))
```

---

## 已废弃写法（0.x 遗留，不要再用）

- `from langchain_openai import OpenAI`（补全模型类）与 `OpenAIChat`：已被 `ChatOpenAI` 取代。
- `LLMChain`、`ConversationChain`、旧版 `create_react_agent` 等：LangChain 1.x 已移入 `langchain_classic` 包，新代码用 LCEL 管道和 `langgraph` 的 `create_agent`。
- 网上大量 0.x 旧教程写法与 1.x 不兼容，甄别时先看 import 路径。

## 选型建议（结合本项目）

- **练习 / 确定只用智谱**：方式① `ChatOpenAI(base_url=智谱)`，直接可控，各章默认写法。
- **需要配置化切换模型 / 写可复用代码**：方式③ `init_chat_model`，把 provider 选择权交给配置。
- **需要厂商特有参数**：方式② 专属类（注意智谱在 community 包，优先级放低）。
- 本地跑通、临时验证：方式④ 环境变量；写测试用假模型。

## 常见坑（本项目实测）

1. **`init_chat_model` 漏传 `base_url`**：默认连 `api.openai.com`，智谱 key 必然 `ConnectTimeout`。`openai:` 前缀只决定实例化哪个类，不决定连哪家。
2. **智谱 `with_structured_output` 必须显式 `method="function_calling"`**：默认的 `json_schema` 模式智谱兼容层不支持，会返回自由文本导致 Pydantic 解析报错。
3. **`glm-*` 不会自动推断 provider**：`init_chat_model("glm-4.5-air")` 会失败，要写 `"openai:glm-4.5-air"`。
4. **不要把 `.env` / 含 key 的 notebook 输出提交进 git**（`.gitignore` 已包含 `.env`）。
