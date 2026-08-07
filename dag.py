"""轻量级 DAG 框架，参考 LangGraph 设计理念。

核心概念：
  - State: 节点间传递的共享状态
  - Node: 处理状态的函数节点
  - Edge: 节点间的连接（支持条件路由）
  - StateGraph: 管理状态和节点的有向图
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict
import abc
import json
import uuid


class CompileError(ValueError):
    """图编译失败时抛出的异常。"""
    pass


class NodeInterrupt(Exception):
    """节点执行中断信号。

    节点函数中调用 interrupt() 时抛出，由 CompiledGraph 捕获后暂停执行。
    """
    pass


def interrupt() -> None:
    """在节点函数中调用，暂停执行等待人工介入。

    执行引擎会保存当前 checkpoint 并返回部分状态。
    调用 resume() 并传入 Command 后继续执行。

    用法:
        def my_node(state: dict) -> dict:
            result = do_something(state)
            interrupt()  # 暂停，等待人工审批
            return result
    """
    raise NodeInterrupt()


class State(TypedDict):
    """状态基类，用户通过继承定义结构化字段。

    用法:
        class MyState(State):
            count: int
            messages: list[str]
    """
    pass


def _merge_state(current: dict, updates: dict | None) -> dict:
    """将节点返回的局部更新合并到当前状态。

    合并策略：updates 中的字段直接覆盖 current 中同名字段（浅合并）。
    当 updates 为 None 时，保持当前状态不变。

    Args:
        current: 当前完整状态。
        updates: 节点返回的局部更新，或 None。

    Returns:
        合并后的新状态字典。
    """
    if updates is None:
        return current
    merged = dict(current)
    merged.update(updates)
    return merged


class Node:
    """图中的节点，封装一个可调用函数。

    节点接收当前状态，返回局部更新（dict 或 None）。

    Args:
        name: 节点名称，在图中唯一标识。
        fn: 可调用对象，签名 (state: dict) -> dict | None。
    """

    def __init__(self, name: str, fn: Callable[[dict], dict | None]) -> None:
        self.name = name
        self.fn = fn

    def run(self, state: dict) -> dict | None:
        """执行节点函数。

        Args:
            state: 当前完整状态。

        Returns:
            节点返回的局部更新，或 None。
        """
        return self.fn(state)


@dataclass
class Edge:
    """普通边：从一个节点无条件指向另一个节点。

    Attributes:
        source: 源节点名称。
        target: 目标节点名称。
    """
    source: str
    target: str


@dataclass
class ConditionalEdge:
    """条件边：根据路由函数的结果动态选择下一个节点。

    Attributes:
        source: 源节点名称。
        router: 路由函数，接收状态返回一个字符串键。
        path_map: 路由返回值到目标节点名称的映射字典。
    """
    source: str
    router: Callable[[dict], str]
    path_map: dict[str, str]


def _topological_sort(
    node_names: set[str],
    edges: list[Edge],
    conditional_edges: list[ConditionalEdge],
) -> list[str]:
    """对图进行拓扑排序，返回节点执行顺序。

    将条件边的所有可能路径并入普通边进行保守校验。
    使用 Kahn 算法。

    Args:
        node_names: 所有节点名称集合。
        edges: 普通边列表。
        conditional_edges: 条件边列表。

    Returns:
        拓扑排序后的节点名称列表。

    Raises:
        CompileError: 图中存在环时抛出。
    """
    # 收集所有边（普通边 + 条件边的所有可能路径）
    all_edges: list[tuple[str, str]] = [(e.source, e.target) for e in edges]
    for ce in conditional_edges:
        for target in ce.path_map.values():
            all_edges.append((ce.source, target))

    # 构建邻接表和入度表
    in_degree: dict[str, int] = {name: 0 for name in node_names}
    adj: dict[str, list[str]] = {name: [] for name in node_names}

    for source, target in all_edges:
        if source in adj and target in adj:
            adj[source].append(target)
            in_degree[target] = in_degree.get(target, 0) + 1

    # Kahn 算法
    queue = [name for name, deg in in_degree.items() if deg == 0]
    sorted_nodes: list[str] = []

    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_nodes) != len(node_names):
        raise CompileError(
            f"图中检测到环，无法编译。"
            f"已排序 {len(sorted_nodes)}/{len(node_names)} 个节点。"
        )

    return sorted_nodes


class StateGraph:
    """管理状态和节点的有向图。

    用户通过此类构建图结构，然后调用 compile() 得到可执行的 CompiledGraph。

    Args:
        state_class: 状态类型（继承自 State 的 TypedDict 子类）。
    """

    def __init__(self, state_class: type[State]) -> None:
        self._state_class = state_class
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._entry_point: str | None = None
        self._finish_point: str | None = None

    def add_node(self, name: str, fn: Callable[[dict], dict | None]) -> None:
        """注册一个节点。

        Args:
            name: 节点名称，在图中唯一标识。
            fn: 可调用对象，签名 (state: dict) -> dict | None。
        """
        self._nodes[name] = Node(name=name, fn=fn)

    def add_edge(self, source: str, target: str) -> None:
        """添加一条普通边：从 source 节点无条件指向 target 节点。

        Args:
            source: 源节点名称。
            target: 目标节点名称。
        """
        self._edges.append(Edge(source=source, target=target))

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[dict], str],
        path_map: dict[str, str],
    ) -> None:
        """添加条件边：根据路由函数的结果动态选择下一个节点。

        Args:
            source: 源节点名称。
            router: 路由函数，接收状态返回一个字符串键。
            path_map: 路由返回值到目标节点名称的映射字典。
        """
        self._conditional_edges.append(
            ConditionalEdge(source=source, router=router, path_map=path_map),
        )

    def set_entry_point(self, name: str) -> None:
        """设置入口节点。

        Args:
            name: 作为入口的节点名称。
        """
        self._entry_point = name

    def set_finish_point(self, name: str) -> None:
        """设置出口节点。

        Args:
            name: 作为出口的节点名称。
        """
        self._finish_point = name

    def _validate(self) -> None:
        """编译前校验图的合法性。"""
        if self._entry_point is None:
            raise CompileError("未设置入口节点（set_entry_point）。")
        if self._finish_point is None:
            raise CompileError("未设置出口节点（set_finish_point）。")
        if self._entry_point not in self._nodes:
            raise CompileError(
                f"入口节点 '{self._entry_point}' 未注册。"
            )
        if self._finish_point not in self._nodes:
            raise CompileError(
                f"出口节点 '{self._finish_point}' 未注册。"
            )

        for edge in self._edges:
            if edge.source not in self._nodes:
                raise CompileError(
                    f"边 '{edge.source} -> {edge.target}' 的源节点 '{edge.source}' 未注册。"
                )
            if edge.target not in self._nodes:
                raise CompileError(
                    f"边 '{edge.source} -> {edge.target}' 的目标节点 '{edge.target}' 未注册。"
                )

        for ce in self._conditional_edges:
            if ce.source not in self._nodes:
                raise CompileError(
                    f"条件边源节点 '{ce.source}' 未注册。"
                )
            for key, target in ce.path_map.items():
                if target not in self._nodes:
                    raise CompileError(
                        f"条件边 '{ce.source}' 的路径 '{key} -> {target}' 目标节点未注册。"
                    )

    def compile(self) -> "CompiledGraph":
        """编译图：校验合法性，生成拓扑排序，返回 CompiledGraph。

        Returns:
            可执行的 CompiledGraph 实例。

        Raises:
            CompileError: 图不合法或存在环时抛出。
        """
        self._validate()

        # 校验通过后，入口/出口一定不为 None
        assert self._entry_point is not None
        assert self._finish_point is not None

        node_names = set(self._nodes.keys())
        execution_order = _topological_sort(
            node_names, self._edges, self._conditional_edges,
        )

        return CompiledGraph(
            state_class=self._state_class,
            nodes=self._nodes,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            entry_point=self._entry_point,
            finish_point=self._finish_point,
            execution_order=execution_order,
        )


class CompiledGraph:
    """编译后的可执行图。

    由 StateGraph.compile() 返回，持有编译后的所有数据和执行顺序。
    """

    def __init__(
        self,
        state_class: type[State],
        nodes: dict[str, Node],
        edges: list[Edge],
        conditional_edges: list[ConditionalEdge],
        entry_point: str,
        finish_point: str,
        execution_order: list[str],
    ) -> None:
        self._state_class = state_class
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._execution_order = execution_order

        # 构建查找索引：source -> target（普通边）
        self._outgoing: dict[str, str] = {}
        for e in edges:
            self._outgoing[e.source] = e.target

        # 构建查找索引：source -> ConditionalEdge（条件边）
        self._conditional_map: dict[str, ConditionalEdge] = {}
        for ce in conditional_edges:
            self._conditional_map[ce.source] = ce

        # 运行时动态断点（非侵入式，无需修改节点代码）
        self._breakpoints: set[str] = set()

    def set_breakpoint(self, node_name: str) -> None:
        """设置运行时断点。执行到指定节点时自动暂停。

        Args:
            node_name: 要设置断点的节点名称。
        """
        self._breakpoints.add(node_name)

    def remove_breakpoint(self, node_name: str) -> None:
        """移除指定节点的运行时断点。

        Args:
            node_name: 要移除断点的节点名称。
        """
        self._breakpoints.discard(node_name)

    def invoke(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
    ) -> dict:
        """执行图，从入口节点开始，按拓扑顺序执行，到达出口节点后停止。

        Args:
            input: 初始状态字典。
            persistence: 可选的持久化存储实例。启用后每步执行后自动保存 checkpoint。
            thread_id: 线程 ID，用于 checkpoint 关联。不传时自动生成。

        Returns:
            执行完成后的最终状态字典。
        """
        state = dict(input)
        current: str | None = self._entry_point

        persistence_enabled = persistence is not None
        if persistence_enabled:
            tid = thread_id or uuid.uuid4().hex
            step = 0

        while current is not None:
            # 保存 checkpoint（在执行节点前保存，崩溃后可恢复到此节点）
            if persistence_enabled:
                step += 1
                cp = Checkpoint(
                    thread_id=tid,
                    step=step,
                    state=state,
                    next_node=current,
                )
                persistence.put(tid, cp)

            # 检查运行时断点
            if current in self._breakpoints:
                if persistence_enabled:
                    final_cp = Checkpoint(
                        thread_id=tid,
                        step=step,
                        state=state,
                        next_node=current,
                        reason="breakpoint",
                    )
                    persistence.put(tid, final_cp)
                return state

            # 执行当前节点
            node = self._nodes[current]
            try:
                updates = node.run(state)
            except NodeInterrupt:
                # 节点调用了 interrupt()，保存 checkpoint 后返回当前状态
                if persistence_enabled:
                    final_cp = Checkpoint(
                        thread_id=tid,
                        step=step,
                        state=state,
                        next_node=current,
                        reason="interrupt",
                    )
                    persistence.put(tid, final_cp)
                return state

            state = _merge_state(state, updates)

            # 到达出口节点则停止
            if current == self._finish_point:
                # 保存最终 checkpoint，标记执行完成
                if persistence_enabled:
                    cp = Checkpoint(
                        thread_id=tid,
                        step=step,
                        state=state,
                        next_node=None,
                        reason="checkpoint",
                    )
                    persistence.put(tid, cp)
                break

            # 查找下一个节点
            current = self._get_next_node(current, state)

        return state

    def _get_next_node(self, current: str, state: dict) -> str | None:
        """根据当前节点和状态，查找下一个应执行的节点。

        Args:
            current: 当前节点名称。
            state: 当前状态。

        Returns:
            下一个节点名称，如果没有则返回 None。
        """
        # 先查普通边
        if current in self._outgoing:
            return self._outgoing[current]
        # 再查条件边
        if current in self._conditional_map:
            ce = self._conditional_map[current]
            route_key = ce.router(state)
            return ce.path_map.get(route_key)
        return None

    def resume(
        self,
        thread_id: str,
        persistence: "BasePersistence",
        *,
        command: "Command | None" = None,
    ) -> dict:
        """从指定线程的 checkpoint 恢复执行。

        加载最近一次 checkpoint，从中断处继续执行。

        Args:
            thread_id: 线程 ID。
            persistence: 持久化存储实例。
            command: 可选的人类指令，可更新状态、重定向节点或终止执行。

        Returns:
            执行完成后的最终状态字典。

        Raises:
            ValueError: 指定线程的 checkpoint 不存在时抛出。
        """
        cp = persistence.get(thread_id)
        if cp is None:
            raise ValueError(f"线程 '{thread_id}' 的 checkpoint 不存在。")

        state = dict(cp.state)
        current: str | None = cp.next_node

        # 如果 next_node 为 None，说明已经执行完成
        if current is None:
            return state

        # 应用人类指令
        if command is not None:
            if command.update is not None:
                state = _merge_state(state, command.update)
            if command.goto is not None:
                current = command.goto
            elif cp.reason == "interrupt":
                # 中断恢复：跳过已中断的节点，找下一个
                current = self._get_next_node(current, state)
            # 断点/崩溃恢复：重新执行当前节点（current 保持不变）
            if not command.resume:
                return state
        elif current is not None:
            # 没有 command，正常恢复：跳过已执行过的 checkpoint 节点
            pass

        step = cp.step

        while current is not None:
            # 保存 checkpoint（在执行节点前保存）
            step += 1
            cp = Checkpoint(
                thread_id=thread_id,
                step=step,
                state=state,
                next_node=current,
            )
            persistence.put(thread_id, cp)

            # 执行当前节点
            node = self._nodes[current]
            updates = node.run(state)
            state = _merge_state(state, updates)

            # 到达出口节点则停止
            if current == self._finish_point:
                cp = Checkpoint(
                    thread_id=thread_id,
                    step=step,
                    state=state,
                    next_node=None,
                )
                persistence.put(thread_id, cp)
                break

            # 查找下一个节点
            current = self._get_next_node(current, state)

        return state


@dataclass
class Checkpoint:
    """执行快照，记录某个时刻的状态和下一步待执行的节点。

    Attributes:
        thread_id: 线程 ID，关联同一执行流。
        step: 步骤编号，从 1 开始递增。
        state: 当前状态快照。
        next_node: 下一步待执行的节点名称。None 表示执行已完成。
        reason: 暂停原因。"checkpoint"（正常保存）、"interrupt"（中断）、"breakpoint"（断点）。
    """
    thread_id: str
    step: int
    state: dict
    next_node: str | None
    reason: str = "checkpoint"


@dataclass
class Command:
    """人类指令，用于恢复执行时提供输入。

    Attributes:
        update: 可选的状态更新，恢复前合并到当前状态。
        goto: 可选的重定向到指定节点。不传则按原路径继续。
        resume: 是否继续执行。设为 False 则终止执行。
    """
    update: dict | None = None
    goto: str | None = None
    resume: bool = True


class BasePersistence(abc.ABC):
    """持久化存储抽象基类。"""

    @abc.abstractmethod
    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        """保存 checkpoint。"""
        ...

    @abc.abstractmethod
    def get(self, thread_id: str) -> Checkpoint | None:
        """加载指定线程的最新 checkpoint。"""
        ...


class FilePersistence(BasePersistence):
    """基于文件系统的持久化存储。

    每个线程的 checkpoint 序列化为 JSON 文件，存储在指定目录下。
    文件名格式: {thread_id}.json
    """

    def __init__(self, base_dir: str = "./.checkpoints") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        file_path = self._base_dir / f"{thread_id}.json"
        data = {
            "thread_id": checkpoint.thread_id,
            "step": checkpoint.step,
            "state": checkpoint.state,
            "next_node": checkpoint.next_node,
            "reason": checkpoint.reason,
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

    def get(self, thread_id: str) -> Checkpoint | None:
        file_path = self._base_dir / f"{thread_id}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data = json.load(f)
        return Checkpoint(
            thread_id=data["thread_id"],
            step=data["step"],
            state=data["state"],
            next_node=data["next_node"],
            reason=data.get("reason", "checkpoint"),
        )