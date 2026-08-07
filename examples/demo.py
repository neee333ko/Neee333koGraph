"""轻量 DAG 框架完整功能演示。

涵盖：核心引擎、持久化、人机协同、断点、LLM/Tool Node、Subgraph、Streaming、并行、超时重试。
"""

import os
import shutil
import time
from dag import (
    State, StateGraph, FilePersistence, Command,
    interrupt, parallel, with_timeout, with_retry, SubgraphNode,
)
from dag_llm import llm_call
from dag_tools import Tool, tool_node


# ===============================
# 1. 核心引擎：条件路由
# ===============================
print("=" * 60)
print("1. 核心引擎：条件路由")
print("=" * 60)

class TextState(State):
    text: str
    output: str
    category: str

def validate(state: dict) -> dict:
    if not state.get("text", ""):
        raise ValueError("输入文本不能为空")
    return {"output": "校验通过"}

def process(state: dict) -> dict:
    return {"output": state["text"].upper()}

def route(state: dict) -> str:
    return "short" if len(state["text"]) < 10 else "long"

def short_fn(state: dict) -> dict:
    return {"output": state["output"] + " [SHORT]", "category": "short"}

def long_fn(state: dict) -> dict:
    return {"output": state["output"] + " [LONG]", "category": "long"}

graph = StateGraph(TextState)
for name, fn in [("validate", validate), ("process", process),
                  ("short", short_fn), ("long", long_fn), ("end", lambda s: None)]:
    graph.add_node(name, fn)
graph.add_edge("validate", "process")
graph.add_conditional_edges("process", route, {"short": "short", "long": "long"})
graph.add_edge("short", "end")
graph.add_edge("long", "end")
graph.set_entry_point("validate")
graph.set_finish_point("end")

app = graph.compile()
result = app.invoke({"text": "hello"})
print(f"  短文本: {result['output']} (分类: {result['category']})")
result = app.invoke({"text": "hello world dag"})
print(f"  长文本: {result['output']} (分类: {result['category']})")


# ===============================
# 2. 持久化与可恢复执行
# ===============================
print("\n" + "=" * 60)
print("2. 持久化与可恢复执行")
print("=" * 60)

ckpt_dir = "./.demo_checkpoints"
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)

persistence = FilePersistence(base_dir=ckpt_dir)
result = app.invoke({"text": "hello"}, persistence=persistence, thread_id="demo1")
print(f"  执行结果: {result['output']}")
cp = persistence.get("demo1")
assert cp is not None and cp.next_node is None
print(f"  最终 checkpoint: step={cp.step}, 已完成")

# 模拟中断恢复
class SimError(Exception):
    pass

def step_a(state: dict) -> dict:
    return {"output": state.get("output", "") + "A->"}

graph2 = StateGraph(TextState)
graph2.add_node("a", step_a)
graph2.add_node("b", lambda s: {"output": s.get("output", "") + "B->"})
graph2.add_node("end", lambda s: None)
graph2.add_edge("a", "b")
graph2.add_edge("b", "end")
graph2.set_entry_point("a")
graph2.set_finish_point("end")
app2 = graph2.compile()

app2.invoke({"text": "x", "output": "", "category": ""}, persistence=persistence, thread_id="crash")
# 模拟崩溃后恢复
result2 = app2.resume("crash", persistence)
print(f"  崩溃恢复: {result2['output']}")


# ===============================
# 3. 人机协同 (Interrupt)
# ===============================
print("\n" + "=" * 60)
print("3. 人机协同 (Interrupt)")
print("=" * 60)

class ApprovalState(State):
    request: str
    approved: bool
    result: str

def review_node(state: dict) -> dict:
    print(f"  [等待审批] 请求: {state['request']}")
    interrupt()  # 暂停等待人工审批
    # 以下代码不会执行（由 Command 提供状态变更）
    return {"result": "自动审批通过"}

g3 = StateGraph(ApprovalState)
g3.add_node("review", review_node)
g3.add_node("process", lambda s: {"result": f"已处理: {s['request']}"})
g3.add_node("end", lambda s: None)
g3.add_edge("review", "process")
g3.add_edge("process", "end")
g3.set_entry_point("review")
g3.set_finish_point("end")
app3 = g3.compile()

partial = app3.invoke({"request": "请假申请", "approved": False, "result": ""},
                       persistence=persistence, thread_id="approve1")
print(f"  中断时状态: {partial}")

# 人工审批后恢复
final = app3.resume("approve1", persistence, command=Command(
    update={"approved": True, "result": "人工审批通过"},
))
print(f"  审批后结果: {final['result']} (approved: {final['approved']})")


# ===============================
# 4. 动态断点
# ===============================
print("\n" + "=" * 60)
print("4. 动态断点 (Breakpoint)")
print("=" * 60)

g4 = StateGraph(TextState)
g4.add_node("a", lambda s: {"output": "step_a"})
g4.add_node("b", lambda s: {"output": "step_b"})
g4.add_node("end", lambda s: None)
g4.add_edge("a", "b")
g4.add_edge("b", "end")
g4.set_entry_point("a")
g4.set_finish_point("end")
app4 = g4.compile()

app4.set_breakpoint("b")  # 在 b 上设断点
partial4 = app4.invoke({"text": "x", "output": "", "category": ""},
                        persistence=persistence, thread_id="bp1")
print(f"  断点暂停于: node=b, state={partial4}")

# 恢复执行
final4 = app4.resume("bp1", persistence, command=Command())
print(f"  断点恢复后: {final4['output']}")
app4.remove_breakpoint("b")


# ===============================
# 5. LLM Node (模拟)
# ===============================
print("\n" + "=" * 60)
print("5. LLM Node (不调用实际 API，仅演示创建)")
print("=" * 60)

# 此处仅演示创建，不实际调用
llm_node = llm_call(
    model="gpt-4o",
    system_prompt="请回答用户问题：{input}",
    response_key="llm_response",
)
# 实际调用需要设置 OPENAI_API_KEY 环境变量
print(f"  LLM Node 已创建，类型: {type(llm_node).__name__}")
print(f"  调用前请设置 OPENAI_API_KEY 环境变量")


# ===============================
# 6. Tool Node (模拟)
# ===============================
print("\n" + "=" * 60)
print("6. Tool Node (模拟工具调用)")
print("=" * 60)

def get_weather(city: str) -> str:
    return f"{city}：晴，22-28°C，湿度 60%"

def search_web(query: str) -> str:
    return f"关于 '{query}' 的搜索结果：共找到 42 条相关结果"

tools = [
    Tool(name="get_weather", description="查询城市天气",
         fn=get_weather,
         parameters={"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}),
    Tool(name="search_web", description="搜索互联网",
         fn=search_web,
         parameters={"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"]}),
]

# 验证工具函数直接调用
print(f"  工具1: get_weather('北京') → {get_weather('北京')}")
print(f"  工具2: search_web('DAG') → {search_web('DAG')}")

# 创建 Tool Node（实际调用需要 API key）
tool_agent = tool_node(tools, llm_model="gpt-4o")
print(f"  Tool Node 已创建，注册工具数: {len(tools)}")


# ===============================
# 7. Subgraph 子图嵌套
# ===============================
print("\n" + "=" * 60)
print("7. Subgraph 子图嵌套")
print("=" * 60)

class SubState(State):
    count: int

sub = StateGraph(SubState)
sub.add_node("inc", lambda s: {"count": s["count"] + 1})
sub.add_node("end", lambda s: None)
sub.add_edge("inc", "end")
sub.set_entry_point("inc")
sub.set_finish_point("end")
sub_app = sub.compile()

class ParentState(State):
    value: int
    result: int

parent = StateGraph(ParentState)
parent.add_node("pre", lambda s: {"value": s["value"] * 10})
parent.add_subgraph("sub", sub_app, {"value": "count"})
parent.add_node("post", lambda s: {"result": s["value"] + 100})
parent.add_node("end", lambda s: None)
parent.add_edge("pre", "sub")
parent.add_edge("sub", "post")
parent.add_edge("post", "end")
parent.set_entry_point("pre")
parent.set_finish_point("end")

parent_app = parent.compile()
result7 = parent_app.invoke({"value": 1, "result": 0})
print(f"  子图结果: value={result7['value']}, result={result7['result']}")
# 流程: pre(1*10=10) → sub(count=10+1=11→value=11) → post(11+100=111)


# ===============================
# 8. Streaming 逐节点输出
# ===============================
print("\n" + "=" * 60)
print("8. Streaming 逐节点输出")
print("=" * 60)

print("  执行步骤:")
for node_name, state in parent_app.stream({"value": 2, "result": 0}):
    print(f"    → {node_name}: {dict(state)}")


# ===============================
# 9. 并行执行
# ===============================
print("\n" + "=" * 60)
print("9. 并行执行")
print("=" * 60)

class MultiState(State):
    count: int
    msg: str
    flag: bool

pn = parallel(
    lambda s: {"count": s["count"] + 10},
    lambda s: {"msg": "hello from parallel"},
    lambda s: {"flag": True},
)
result9 = pn({"count": 0, "msg": "", "flag": False})
print(f"  并行结果: count={result9['count']}, msg={result9['msg']}, flag={result9['flag']}")


# ===============================
# 10. 超时与重试
# ===============================
print("\n" + "=" * 60)
print("10. 超时与重试")
print("=" * 60)

# 重试：模拟一个最终会成功的函数
attempt = {"n": 0}
def flaky(state: dict) -> dict:
    attempt["n"] += 1
    if attempt["n"] < 3:
        raise ValueError(f"第{attempt['n']}次失败")
    return {"count": state["count"] + 1}

safe_fn = with_retry(flaky, max_retries=3, delay=0.01)
result10 = safe_fn({"count": 0, "msg": "", "flag": False})
assert result10 is not None
print(f"  重试成功: 尝试{attempt['n']}次, 结果 count={result10['count']}")

# 超时：快速执行演示
def fast(state: dict) -> dict:
    return {"count": state["count"] * 2}

timed = with_timeout(5, fast)
result10b = timed({"count": 5, "msg": "", "flag": False})
assert result10b is not None
print(f"  超时控制(正常): count={result10b['count']}")

# 超时：超时异常演示
def slow(state: dict) -> dict:
    time.sleep(10)
    return state

timed_slow = with_timeout(0.01, slow)
try:
    timed_slow({"count": 0, "msg": "", "flag": False})
except TimeoutError:
    print(f"  超时控制(超时): 正确抛出 TimeoutError")


# ===============================
# 清理
# ===============================
shutil.rmtree(ckpt_dir, ignore_errors=True)
print("\n" + "=" * 60)
print("所有功能演示完成！")
print("=" * 60)