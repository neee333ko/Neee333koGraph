"""轻量 DAG 框架单元测试。"""

import unittest
from dag import (
    State, Node, Edge, ConditionalEdge,
    StateGraph, CompiledGraph, CompileError,
    _merge_state,
)


class TestState(unittest.TestCase):
    """测试 State 基类。"""

    def test_inherit(self):
        """验证 State 可被继承并定义字段。"""
        class MyState(State):
            count: int
            name: str

        s = MyState(count=1, name="test")
        self.assertEqual(s["count"], 1)
        self.assertEqual(s["name"], "test")

    def test_empty_state(self):
        """验证 State 可以没有字段。"""
        class EmptyState(State):
            pass

        s = EmptyState()
        self.assertEqual(dict(s), {})


class TestMergeState(unittest.TestCase):
    """测试 _merge_state 辅助函数。"""

    def test_merge_overwrite(self):
        """验证 updates 覆盖 current 同名字段。"""
        result = _merge_state({"count": 1, "name": "a"}, {"count": 2})
        self.assertEqual(result, {"count": 2, "name": "a"})

    def test_merge_add_new(self):
        """验证 updates 可以添加新字段。"""
        result = _merge_state({"count": 1}, {"name": "b"})
        self.assertEqual(result, {"count": 1, "name": "b"})

    def test_merge_none(self):
        """验证 updates 为 None 时返回原字典。"""
        original = {"count": 1}
        result = _merge_state(original, None)
        self.assertIs(result, original)

    def test_merge_empty(self):
        """验证 updates 为空 dict 时状态不变。"""
        original = {"count": 1}
        result = _merge_state(original, {})
        self.assertEqual(result, original)

    def test_merge_returns_new_dict(self):
        """验证返回新字典，不修改原对象。"""
        original = {"count": 1}
        result = _merge_state(original, {"count": 2})
        self.assertIsNot(result, original)
        self.assertEqual(original["count"], 1)


class TestNode(unittest.TestCase):
    """测试 Node 类。"""

    def test_create_and_run(self):
        """验证节点创建和执行。"""
        node = Node("add", lambda s: {"count": s["count"] + 1})
        self.assertEqual(node.name, "add")
        result = node.run({"count": 1})
        self.assertEqual(result, {"count": 2})

    def test_run_return_none(self):
        """验证节点返回 None。"""
        node = Node("noop", lambda s: None)
        result = node.run({"count": 1})
        self.assertIsNone(result)


class TestEdge(unittest.TestCase):
    """测试 Edge 和 ConditionalEdge 数据类。"""

    def test_normal_edge(self):
        """验证普通边创建。"""
        e = Edge(source="a", target="b")
        self.assertEqual(e.source, "a")
        self.assertEqual(e.target, "b")

    def test_conditional_edge(self):
        """验证条件边创建和路由。"""
        def router(s: dict) -> str:
            return "high" if s["count"] > 5 else "low"

        ce = ConditionalEdge(
            source="a",
            router=router,
            path_map={"high": "b", "low": "c"},
        )
        self.assertEqual(ce.source, "a")
        self.assertEqual(ce.router({"count": 10}), "high")
        self.assertEqual(ce.path_map["high"], "b")


class TestStateGraphBuild(unittest.TestCase):
    """测试 StateGraph 构建方法。"""

    def setUp(self):
        class S(State):
            pass
        self.S = S

    def test_add_node(self):
        """验证 add_node 注册节点。"""
        g = StateGraph(self.S)
        g.add_node("a", lambda s: None)
        self.assertIn("a", g._nodes)
        self.assertEqual(g._nodes["a"].name, "a")

    def test_add_edge(self):
        """验证 add_edge 添加边。"""
        g = StateGraph(self.S)
        g.add_edge("a", "b")
        self.assertEqual(len(g._edges), 1)
        self.assertEqual(g._edges[0].source, "a")
        self.assertEqual(g._edges[0].target, "b")

    def test_add_conditional_edges(self):
        """验证 add_conditional_edges 添加条件边。"""
        g = StateGraph(self.S)
        g.add_conditional_edges("a", lambda s: "x", {"x": "b"})
        self.assertEqual(len(g._conditional_edges), 1)
        self.assertEqual(g._conditional_edges[0].source, "a")

    def test_set_entry_and_finish(self):
        """验证设置入口和出口。"""
        g = StateGraph(self.S)
        g.set_entry_point("start")
        g.set_finish_point("end")
        self.assertEqual(g._entry_point, "start")
        self.assertEqual(g._finish_point, "end")


class TestCompile(unittest.TestCase):
    """测试编译与校验。"""

    def setUp(self):
        class S(State):
            count: int
        self.S = S
        self.fn = lambda s: {"count": s["count"] + 1}

    def _make_graph(self):
        g = StateGraph(self.S)
        g.add_node("a", self.fn)
        g.add_node("b", self.fn)
        g.add_node("end", lambda s: None)
        return g

    def test_compile_success(self):
        """验证正常编译通过。"""
        g = self._make_graph()
        g.add_edge("a", "b")
        g.add_edge("b", "end")
        g.set_entry_point("a")
        g.set_finish_point("end")
        app = g.compile()
        self.assertIsInstance(app, CompiledGraph)
        self.assertEqual(app._entry_point, "a")
        self.assertEqual(app._execution_order, ["a", "b", "end"])

    def test_compile_no_entry(self):
        """验证未设置入口时抛出异常。"""
        g = self._make_graph()
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_no_finish(self):
        """验证未设置出口时抛出异常。"""
        g = self._make_graph()
        g.set_entry_point("a")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_entry_not_registered(self):
        """验证入口节点未注册时抛出异常。"""
        g = self._make_graph()
        g.set_entry_point("nonexistent")
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_finish_not_registered(self):
        """验证出口节点未注册时抛出异常。"""
        g = self._make_graph()
        g.add_node("a", self.fn)
        g.set_entry_point("a")
        g.set_finish_point("nonexistent")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_edge_source_not_registered(self):
        """验证边的源节点未注册时抛出异常。"""
        g = self._make_graph()
        g.add_edge("nonexistent", "b")
        g.set_entry_point("a")
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_edge_target_not_registered(self):
        """验证边的目标节点未注册时抛出异常。"""
        g = self._make_graph()
        g.add_edge("a", "nonexistent")
        g.set_entry_point("a")
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_cycle(self):
        """验证环检测。"""
        g = self._make_graph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")  # 环
        g.set_entry_point("a")
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()

    def test_compile_conditional_edge_bad_target(self):
        """验证条件边 path_map 指向未注册节点时抛出异常。"""
        g = self._make_graph()
        g.add_conditional_edges("a", lambda s: "x", {"x": "nonexistent"})
        g.set_entry_point("a")
        g.set_finish_point("end")
        with self.assertRaises(CompileError):
            g.compile()


class TestInvoke(unittest.TestCase):
    """测试图执行。"""

    def setUp(self):
        class S(State):
            count: int
        self.S = S

    def test_linear_chain(self):
        """验证线性链执行。"""
        g = StateGraph(self.S)
        g.add_node("a", lambda s: {"count": s["count"] + 1})
        g.add_node("b", lambda s: {"count": s["count"] * 2})
        g.add_node("end", lambda s: None)
        g.add_edge("a", "b")
        g.add_edge("b", "end")
        g.set_entry_point("a")
        g.set_finish_point("end")
        app = g.compile()
        result = app.invoke({"count": 1})
        self.assertEqual(result["count"], 4)  # (1+1)*2 = 4

    def test_conditional_route_low(self):
        """验证条件路由 low 路径。"""
        g = StateGraph(self.S)
        g.add_node("a", lambda s: {"count": s["count"] + 1})
        g.add_node("end", lambda s: None)
        g.add_conditional_edges(
            "a", lambda s: "low" if s["count"] < 5 else "high",
            {"low": "end", "high": "end"},
        )
        g.set_entry_point("a")
        g.set_finish_point("end")
        app = g.compile()
        result = app.invoke({"count": 1})
        self.assertEqual(result["count"], 2)

    def test_conditional_route_high(self):
        """验证条件路由 high 路径。"""
        g = StateGraph(self.S)
        g.add_node("a", lambda s: {"count": s["count"] + 1})
        g.add_node("b", lambda s: {"count": s["count"] * 2})
        g.add_node("end", lambda s: None)
        g.add_conditional_edges(
            "a", lambda s: "high" if s["count"] > 5 else "low",
            {"high": "b", "low": "end"},
        )
        g.add_edge("b", "end")
        g.set_entry_point("a")
        g.set_finish_point("end")
        app = g.compile()
        result = app.invoke({"count": 10})
        self.assertEqual(result["count"], 22)  # (10+1)*2 = 22

    def test_node_returns_none(self):
        """验证节点返回 None 时状态不变。"""
        g = StateGraph(self.S)
        g.add_node("noop", lambda s: None)
        g.add_node("end", lambda s: None)
        g.add_edge("noop", "end")
        g.set_entry_point("noop")
        g.set_finish_point("end")
        app = g.compile()
        result = app.invoke({"count": 42})
        self.assertEqual(result["count"], 42)

    def test_multiple_fields(self):
        """验证多字段状态正确流转。"""
        class MultiState(State):
            count: int
            name: str

        g = StateGraph(MultiState)
        g.add_node("add", lambda s: {"count": s["count"] + 1})
        g.add_node("end", lambda s: None)
        g.add_edge("add", "end")
        g.set_entry_point("add")
        g.set_finish_point("end")
        app = g.compile()
        result = app.invoke({"count": 0, "name": "test"})
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["name"], "test")


if __name__ == "__main__":
    unittest.main()