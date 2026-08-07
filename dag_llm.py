"""LLM Node — 封装 LLM 调用，支持 OpenAI 兼容 API。

用法:
    from dag_llm import llm_call
    from dag import StateGraph

    graph.add_node("agent", llm_call(
        model="gpt-4o",
        system_prompt="请回答用户问题：{input}",
    ))
"""

from collections.abc import Callable
import json
import os
import urllib.error
import urllib.request


def llm_call(
    model: str,
    system_prompt: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    input_key: str = "input",
    response_key: str = "response",
    temperature: float = 0.7,
) -> Callable[[dict], dict]:
    """创建一个调用 LLM 的节点函数。

    system_prompt 支持 {field_name} 模板变量，会自动从 state 中填充。

    Args:
        model: 模型名称，如 "gpt-4o", "deepseek-chat", "claude-3-opus"。
        system_prompt: 系统提示词，支持 {field} 模板变量。
        api_key: API key。默认读取 OPENAI_API_KEY 环境变量。
        base_url: API 基础地址。默认 https://api.openai.com/v1。
        input_key: 从 state 中取用户输入的字段名，默认 "input"。
        response_key: LLM 响应写入 state 的字段名，默认 "response"。
        temperature: 温度参数，默认 0.7。

    Returns:
        节点函数，签名 (state: dict) -> dict。
    """
    resolved_base_url = base_url or "https://api.openai.com/v1"

    def _node(state: dict) -> dict:
        resolved_key = _resolve_api_key(api_key)
        formatted_prompt = system_prompt.format(**state)
        user_message = str(state.get(input_key, ""))

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_message},
        ]

        response = _call_llm(
            messages=messages,
            model=model,
            api_key=resolved_key,
            base_url=resolved_base_url,
            temperature=temperature,
        )
        return {response_key: response}

    return _node


def _resolve_api_key(api_key: str | None) -> str:
    """解析 API key，优先使用参数，其次环境变量。"""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "未提供 API key。请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数。"
        )
    return key


def _call_llm(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> str:
    """调用 OpenAI 兼容 API。

    使用标准库 urllib，零外部依赖。

    Args:
        messages: 消息列表。
        model: 模型名称。
        api_key: API key。
        base_url: API 基础地址。
        temperature: 温度参数。

    Returns:
        LLM 返回的文本内容。

    Raises:
        RuntimeError: API 调用失败或响应格式异常时抛出。
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"LLM API 响应格式异常: {e}, 响应: {json.dumps(data, ensure_ascii=False)}"
        ) from e