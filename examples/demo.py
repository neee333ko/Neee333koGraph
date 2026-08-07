"""轻量DAG框架使用示例——文本处理流水线。

执行流程：
  validate -> process -> route -> (short | long) -> end
"""

from dag import State, StateGraph


# 1. 定义状态
class TextState(State):
    text: str
    length_category: str
    output: str


# 2. 定义节点函数
def validate(state: dict) -> dict:
    """校验输入文本不为空。"""
    text = state.get("text", "")
    if not text:
        raise ValueError("输入文本不能为空")
    return {"output": "校验通过"}


def process(state: dict) -> dict:
    """将文本转为大写。"""
    text = state["text"].upper()
    return {"output": text}


def route(state: dict) -> str:
    """根据文本长度决定路径。"""
    text = state["text"]
    return "short" if len(text) < 10 else "long"


def short_process(state: dict) -> dict:
    """短文本处理：添加后缀。"""
    return {"output": state["output"] + " [SHORT]", "length_category": "short"}


def long_process(state: dict) -> dict:
    """长文本处理：添加后缀。"""
    return {"output": state["output"] + " [LONG]", "length_category": "long"}


def end(state: dict) -> dict:
    """结束节点，汇总结果。"""
    return {"output": state["output"] + " [DONE]"}


# 3. 构建图
graph = StateGraph(TextState)
graph.add_node("validate", validate)
graph.add_node("process", process)
graph.add_node("short", short_process)
graph.add_node("long", long_process)
graph.add_node("end", end)

graph.add_edge("validate", "process")
graph.add_conditional_edges("process", route, {"short": "short", "long": "long"})
graph.add_edge("short", "end")
graph.add_edge("long", "end")

graph.set_entry_point("validate")
graph.set_finish_point("end")

# 4. 编译
app = graph.compile()

# 5. 执行
if __name__ == "__main__":
    # 测试短文本
    print("=== 短文本测试 ===")
    result1 = app.invoke({"text": "hello"})
    print(f"输入: 'hello'")
    print(f"输出: {result1['output']}")
    print(f"分类: {result1['length_category']}")
    print()

    # 测试长文本
    print("=== 长文本测试 ===")
    result2 = app.invoke({"text": "hello world dag"})
    print(f"输入: 'hello world dag'")
    print(f"输出: {result2['output']}")
    print(f"分类: {result2['length_category']}")