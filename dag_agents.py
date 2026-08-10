"""通用 Agent 图工厂。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any
import asyncio
import inspect

from dag import CompiledGraph, State, StateGraph
from dag_tools import Tool, _aexecute_tool_call


def append_items(
    current: list[Any] | None,
    updates: list[Any],
) -> list[Any]:
    """安全追加列表状态，兼容首次更新时字段不存在。"""
    return list(current or []) + list(updates)


class ReActState(State, total=False):
    """ReAct Agent 的通用状态。"""

    input: str
    messages: Annotated[list[dict], append_items]
    tool_calls: list[dict]
    tool_results: Annotated[list[dict], append_items]
    output: Any
    done: bool
    iterations: int


ModelCallable = Callable[
    [dict],
    dict | Awaitable[dict],
]


def create_react_agent(
    model: ModelCallable,
    tools: list[Tool],
    *,
    max_iterations: int = 10,
) -> CompiledGraph:
    """创建异步 ReAct Agent 图。

    model 每轮接收完整状态，并返回局部更新。常用字段：
    - tool_calls: [{"name": str, "arguments": dict, "id": str | None}]
    - output: 最终答案
    - done: 是否结束

    Args:
        model: 同步或异步模型决策函数。
        tools: Agent 可调用的工具。
        max_iterations: 最大模型决策轮数。

    Returns:
        允许循环的 CompiledGraph，应使用 ainvoke()/astream() 执行。
    """
    if max_iterations < 1:
        raise ValueError("max_iterations 必须大于等于 1。")
    if not tools:
        raise ValueError("至少需要提供一个工具。")

    async def call_model(state: dict) -> dict:
        result = model(state)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError("ReAct model 必须返回 dict 状态更新。")
        updates = dict(result)
        updates["iterations"] = int(state.get("iterations", 0)) + 1
        updates.setdefault("tool_calls", [])
        return updates

    async def execute_tools(state: dict) -> dict:
        calls = state.get("tool_calls", [])
        results = await asyncio.gather(
            *(_aexecute_tool_call(tools, call) for call in calls)
        )
        records = [
            {
                "tool_call_id": call.get("id"),
                "name": call["name"],
                "result": result,
            }
            for call, result in zip(calls, results)
        ]
        messages = [
            {
                "role": "tool",
                "tool_call_id": record["tool_call_id"],
                "name": record["name"],
                "content": str(record["result"]),
            }
            for record in records
        ]
        return {
            "tool_calls": [],
            "tool_results": records,
            "messages": messages,
        }

    def route_after_model(state: dict) -> str:
        if state.get("done"):
            return "end"
        if int(state.get("iterations", 0)) >= max_iterations:
            return "end"
        return "tools" if state.get("tool_calls") else "end"

    graph = StateGraph(ReActState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("end", lambda state: None)
    graph.add_conditional_edges(
        "agent",
        route_after_model,
        {"tools": "tools", "end": "end"},
    )
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")
    graph.set_finish_point("end")
    return graph.compile(allow_cycles=True)
