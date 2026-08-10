"""Nekograph 扩展能力测试。"""

import asyncio
import tempfile
import unittest
from pathlib import Path

import nekograph
from dag import (
    Checkpoint,
    RunConfig,
    SQLitePersistence,
    State,
    StateGraph,
    aparallel,
    awith_retry,
    awith_timeout,
)
from dag_agents import create_react_agent
from dag_llm import _parse_json_content
from dag_store import InMemoryStore, SQLiteStore
from dag_tools import Tool, _aexecute_tool_call, _execute_tool_call


class TestSQLitePersistence(unittest.TestCase):
    """测试 SQLite checkpoint 历史。"""

    def test_history_and_parent_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            persistence = SQLitePersistence(
                Path(directory) / "checkpoints.db"
            )
            try:
                for step in range(1, 4):
                    persistence.put(
                        "room",
                        Checkpoint(
                            "room",
                            step,
                            {"step": step},
                            "work",
                            metadata={"source": "test"},
                        ),
                    )
                latest = persistence.get("room")
                self.assertIsNotNone(latest)
                assert latest is not None
                self.assertEqual(latest.step, 3)
                history = persistence.history("room")
                self.assertEqual(
                    [checkpoint.step for checkpoint in history],
                    [3, 2, 1],
                )
                self.assertEqual(
                    history[0].parent_id,
                    history[1].checkpoint_id,
                )
                self.assertEqual(len(persistence.history("room", limit=2)), 2)
            finally:
                persistence.close()


class TestStores(unittest.TestCase):
    """测试长期记忆后端。"""

    def test_memory_and_sqlite_store(self):
        stores = [InMemoryStore(), SQLiteStore(":memory:")]
        try:
            for store in stores:
                namespace = ("streamer", "42")
                store.put(
                    namespace,
                    "profile",
                    {"game": "demo", "risk": "low"},
                )
                store.put(
                    namespace,
                    "baseline",
                    {"game": "demo", "score": 90},
                )
                item = store.get(namespace, "profile")
                self.assertIsNotNone(item)
                assert item is not None
                self.assertEqual(item.value["risk"], "low")
                self.assertEqual(len(store.search(namespace, query="demo")), 2)
                self.assertEqual(
                    len(store.search(
                        namespace,
                        filters={"game": "demo"},
                    )),
                    2,
                )
                self.assertTrue(store.delete(namespace, "profile"))
                self.assertIsNone(store.get(namespace, "profile"))
        finally:
            stores[1].close()


class TestAsyncReliability(unittest.IsolatedAsyncioTestCase):
    """测试异步并行、重试和超时。"""

    async def test_async_wrappers(self):
        attempt = 0

        async def flaky(state):
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ValueError("retry")
            return {"count": state["count"] + 1}

        retried = awith_retry(flaky, max_retries=2, delay=0)
        self.assertEqual(await retried({"count": 0}), {"count": 1})

        parallel_node = aparallel(
            lambda state: {"left": state["value"]},
            lambda state: {"right": state["value"] * 2},
        )
        self.assertEqual(
            await parallel_node({"value": 2}),
            {"left": 2, "right": 4},
        )

        async def slow(state):
            await asyncio.sleep(0.05)
            return state

        timed = awith_timeout(0.001, slow)
        with self.assertRaises(TimeoutError):
            await timed({})

    async def test_async_subgraph(self):
        class ChildState(State):
            value: int

        async def increment(state):
            await asyncio.sleep(0)
            return {"value": state["value"] + 1}

        child = StateGraph(ChildState)
        child.add_node("inc", increment)
        child.add_node("end", lambda state: None)
        child.add_edge("inc", "end")
        child.set_entry_point("inc")
        child.set_finish_point("end")

        parent = StateGraph(ChildState)
        parent.add_subgraph("child", child.compile())
        parent.add_node("end", lambda state: None)
        parent.add_edge("child", "end")
        parent.set_entry_point("child")
        parent.set_finish_point("end")

        result = await parent.compile().ainvoke({"value": 1})
        self.assertEqual(result["value"], 2)

    async def test_async_checkpoint_persistence(self):
        class CountState(State):
            count: int

        graph = StateGraph(CountState)
        graph.add_node(
            "inc",
            lambda state: {"count": state["count"] + 1},
        )
        graph.add_node("end", lambda state: None)
        graph.add_edge("inc", "end")
        graph.set_entry_point("inc")
        graph.set_finish_point("end")

        persistence = SQLitePersistence(":memory:")
        try:
            result = await graph.compile().ainvoke(
                {"count": 0},
                persistence=persistence,
                config=RunConfig(
                    thread_id="async-room",
                    metadata={"room_id": 42},
                ),
            )
            self.assertEqual(result["count"], 1)
            latest = await persistence.aget("async-room")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertIsNone(latest.next_node)
            self.assertEqual(latest.metadata, {"room_id": 42})
            self.assertEqual(
                len(await persistence.ahistory("async-room")),
                3,
            )
        finally:
            persistence.close()

    async def test_sync_invoke_rejects_async_node(self):
        class CountState(State):
            count: int

        async def increment(state):
            return {"count": state["count"] + 1}

        graph = StateGraph(CountState)
        graph.add_node("inc", increment)
        graph.add_node("end", lambda state: None)
        graph.add_edge("inc", "end")
        graph.set_entry_point("inc")
        graph.set_finish_point("end")

        with self.assertRaisesRegex(TypeError, "请使用 ainvoke"):
            graph.compile().invoke({"count": 0})


class TestAgentExtensions(unittest.IsolatedAsyncioTestCase):
    """测试工具与 ReAct Agent。"""

    async def test_tool_execution_and_react_loop(self):
        async def metric(room: str):
            await asyncio.sleep(0)
            return {"room": room, "score": 88}

        tools = [Tool("metric", "查询直播指标", metric)]
        call = {
            "id": "call-1",
            "name": "metric",
            "arguments": {"room": "42"},
        }
        with self.assertRaises(TypeError):
            _execute_tool_call(tools, call)
        self.assertEqual(
            await _aexecute_tool_call(tools, call),
            {"room": "42", "score": 88},
        )

        async def model(state):
            if not state.get("tool_results"):
                return {"tool_calls": [call]}
            return {
                "output": state["tool_results"][-1]["result"],
                "done": True,
            }

        app = create_react_agent(model, tools, max_iterations=3)
        result = await app.ainvoke({
            "input": "诊断直播间",
            "messages": [],
            "tool_results": [],
            "iterations": 0,
        })
        self.assertEqual(result["output"]["score"], 88)
        self.assertEqual(result["iterations"], 2)

    def test_structured_json_and_package_exports(self):
        self.assertEqual(
            _parse_json_content('```json\n{"risk": "low"}\n```'),
            {"risk": "low"},
        )
        self.assertIs(nekograph.StateGraph, StateGraph)
        self.assertEqual(nekograph.__version__, "0.2.0")


if __name__ == "__main__":
    unittest.main()
