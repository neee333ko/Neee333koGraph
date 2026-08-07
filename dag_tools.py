"""Tool Node — 工具注册与调用。

LLM 根据用户问题选择工具并生成参数，框架自动执行工具并将结果写回状态。

用法:
    from dag_tools import Tool, tool_node

    tools = [
        Tool(
            name="get_weather",
            description="获取指定城市的天气",
            fn=lambda city: f"{city} 天气：晴，25°C",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"],
            },
        ),
    ]

    graph.add_node("agent", tool_node(tools))
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from dag_llm import _resolve_api_key
import json
import urllib.error
import urllib.request


@dataclass
class Tool:
    """工具定义。

    Attributes:
        name: 工具名称，唯一标识。
        description: 工具描述，LLM 据此决定是否调用。
        fn: 工具函数，接收参数并返回结果字符串。
        parameters: 参数 JSON Schema 定义。
    """
    name: str
    description: str
    fn: Callable[..., str]
    parameters: dict = field(default_factory=dict)


def tool_node(
    tools: list[Tool],
    *,
    llm_model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    input_key: str = "input",
    response_key: str = "tool_result",
    system_prompt: str = "你是一个助手，根据用户问题选择合适的工具并调用。",
    temperature: float = 0.1,
) -> Callable[[dict], dict]:
    """创建一个 Agent 节点：LLM 选择工具 → 执行工具 → 结果回写状态。

    Args:
        tools: 可用的工具列表。
        llm_model: 模型名称。
        api_key: API key。默认读取 OPENAI_API_KEY 环境变量。
        base_url: API 基础地址。
        input_key: 从 state 中取用户输入的字段名。
        response_key: 工具结果写入 state 的字段名。
        system_prompt: 系统提示词。
        temperature: 温度参数，工具调用建议用较低值。

    Returns:
        节点函数，签名 (state: dict) -> dict。
    """
    if not tools:
        raise ValueError("至少需要提供一个工具。")

    resolved_base_url = base_url or "https://api.openai.com/v1"
    tool_defs = _format_tools(tools)

    def _node(state: dict) -> dict:
        resolved_key = _resolve_api_key(api_key)
        user_message = str(state.get(input_key, ""))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # 调用 LLM 获取工具选择
        tool_call = _call_llm_with_tools(
            messages=messages,
            model=llm_model,
            api_key=resolved_key,
            base_url=resolved_base_url,
            tools=tool_defs,
            temperature=temperature,
        )

        if tool_call is None:
            return {response_key: "LLM 未选择任何工具"}

        # 查找并执行工具
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        matched = next((t for t in tools if t.name == tool_name), None)
        if matched is None:
            raise ValueError(f"未知工具: {tool_name}")

        result = matched.fn(**tool_args)
        return {response_key: result}

    return _node


def _format_tools(tools: list[Tool]) -> list[dict]:
    """将工具列表格式化为 OpenAI function calling 格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _call_llm_with_tools(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    tools: list[dict],
    temperature: float,
) -> dict | None:
    """调用 LLM 并解析工具调用结果。

    Args:
        messages: 消息列表。
        model: 模型名称。
        api_key: API key。
        base_url: API 基础地址。
        tools: 工具定义列表（OpenAI 格式）。
        temperature: 温度参数。

    Returns:
        {name, arguments} 或 None（未选择工具）。

    Raises:
        RuntimeError: API 调用失败或响应格式异常时抛出。
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM API 请求失败 (HTTP {e.code}): {body}"
        ) from e
    except OSError as e:
        raise RuntimeError(f"LLM API 网络请求失败: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM API 响应 JSON 解析失败: {e}") from e

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"LLM API 响应格式异常: {e}, 响应: {json.dumps(data, ensure_ascii=False)}"
        ) from e

    # 解析 tool_calls
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None

    tc = tool_calls[0]
    return {
        "name": tc["function"]["name"],
        "arguments": json.loads(tc["function"]["arguments"]),
    }