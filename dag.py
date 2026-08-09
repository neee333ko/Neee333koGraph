"""轻量级 DAG 框架，参考 LangGraph 设计理念。

核心概念：
  - State: 节点间传递的共享状态
  - Node: 处理状态的函数节点
  - Edge: 节点间的连接（支持条件路由）
  - StateGraph: 管理状态和节点的有向图
"""

from collections.abc import AsyncGenerator, Callable, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Protocol,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)
import abc
import inspect
import json
import time
import uuid


@runtime_checkable
class _NodeProtocol(Protocol):
    """节点协议：任何具有 run 方法的对象都可用作节点。"""
    name: str
    def run(self, state: dict) -> Any: ...


class CompileError(ValueError):
    """图编译失败时抛出的异常。"""
    pass


class GraphRecursionError(RuntimeError):
    """图执行超过最大允许节点步数时抛出的异常。"""
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


@dataclass
class RunConfig:
    """单次图执行的运行配置。

    Attributes:
        thread_id: 执行线程 ID，用于关联 checkpoint。
        recursion_limit: 单次执行允许的最大节点步数。
        max_concurrency: 动态并行任务的最大并发数。
        metadata: 运行级可观测元数据。
        context: 不写入 State 的运行时上下文。
    """

    thread_id: str | None = None
    recursion_limit: int = 100
    max_concurrency: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


def _resolve_run_config(
    config: RunConfig | None,
    thread_id: str | None = None,
) -> RunConfig:
    """解析并校验单次执行使用的运行配置。

    Args:
        config: 可选的运行配置。
        thread_id: 兼容旧 API 的线程 ID；传入时优先于 config.thread_id。

    Returns:
        与调用方配置隔离的 RunConfig 实例。

    Raises:
        ValueError: recursion_limit 或 max_concurrency 小于 1 时抛出。
    """
    source = config or RunConfig()
    if source.recursion_limit < 1:
        raise ValueError("recursion_limit 必须大于等于 1。")
    if source.max_concurrency < 1:
        raise ValueError("max_concurrency 必须大于等于 1。")
    return RunConfig(
        thread_id=thread_id if thread_id is not None else source.thread_id,
        recursion_limit=source.recursion_limit,
        max_concurrency=source.max_concurrency,
        metadata=dict(source.metadata),
        context=dict(source.context),
    )


def _merge_state(
    current: dict,
    updates: dict | None,
    reducers: dict[str, Callable[[Any, Any], Any]] | None = None,
) -> dict:
    """将节点返回的局部更新合并到当前状态。

    合并策略：
      - 默认行为：updates 中的字段直接覆盖 current 中同名字段（浅合并）。
      - 归约器：如果某字段在 reducers 中有注册，则调用归约器 (current_value, update_value) 计算新值。
    当 updates 为 None 时，保持当前状态不变。

    Args:
        current: 当前完整状态。
        updates: 节点返回的局部更新，或 None。
        reducers: 可选的自定义归约器字典 {字段名: 归约函数}。
                  归约函数签名 (current_value, update_value) -> new_value。

    Returns:
        合并后的新状态字典。
    """
    if updates is None:
        return current
    merged = dict(current)
    for key, value in updates.items():
        if reducers and key in reducers:
            merged[key] = reducers[key](merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _parse_reducers(state_class: type) -> dict[str, Callable[[Any, Any], Any]]:
    """从 State 类的 Annotated 类型注解中提取归约器。

    用法:
        class MyState(State):
            messages: Annotated[list[str], operator.add]

    解析后返回 {"messages": operator.add}。

    Args:
        state_class: 继承自 State 的 TypedDict 子类。

    Returns:
        {字段名: 归约函数} 字典，没有归约器则返回空字典。
    """
    reducers: dict[str, Callable[[Any, Any], Any]] = {}
    type_hints = get_type_hints(state_class, include_extras=True)
    for field_name, field_type in type_hints.items():
        if get_origin(field_type) is not Annotated:
            continue
        for metadata in get_args(field_type)[1:]:
            if callable(metadata):
                reducers[field_name] = metadata
                break
    return reducers


class Node:
    """图中的节点，封装一个可调用函数。

    节点接收当前状态，返回局部更新（dict 或 None）。
    支持同步和 async 函数。

    Args:
        name: 节点名称，在图中唯一标识。
        fn: 可调用对象，签名 (state: dict) -> dict | None。
    """

    def __init__(self, name: str, fn: Callable[..., Any]) -> None:
        self.name = name
        self.fn = fn

    def run(self, state: dict) -> Any:
        """执行节点函数。

        如果是 async 函数，返回 coroutine 需要 await。

        Args:
            state: 当前完整状态。

        Returns:
            节点返回的局部更新，或 None。
        """
        return self.fn(state)


class SubgraphNode:
    """子图节点，包装一个编译后的子图。

    执行时从父状态提取子状态 → 执行子图 → 合并回父状态。

    Args:
        name: 节点名称。
        subgraph: 编译后的子图。
        state_mapping: 可选的状态字段映射 {父字段: 子字段}。
                      不传则整个父状态传入子图，子图结果全部合并回父状态。
    """

    def __init__(
        self,
        name: str,
        subgraph: "CompiledGraph",
        state_mapping: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.subgraph = subgraph
        self.state_mapping = state_mapping

    def run(self, state: dict) -> dict | None:
        """执行子图。

        Args:
            state: 父状态。

        Returns:
            子图执行后的状态更新。
        """
        if self.state_mapping:
            # 选择性映射：只提取映射的字段
            child_state = {}
            for parent_field, child_field in self.state_mapping.items():
                if parent_field in state:
                    child_state[child_field] = state[parent_field]
        else:
            # 完整映射：整个父状态传入子图
            child_state = dict(state)

        result = self.subgraph.invoke(child_state)

        if self.state_mapping:
            # 反向映射：子图结果写回父状态字段
            updates = {}
            for parent_field, child_field in self.state_mapping.items():
                if child_field in result:
                    updates[parent_field] = result[child_field]
            return updates
        return result


def parallel(*fns: Callable[[dict], dict | None]) -> Callable[[dict], dict]:
    """创建一个并行执行节点。

    多个函数并发执行，结果合并后返回。适用于扇出场景。

    Args:
        fns: 需要并发执行的函数列表，签名 (state) -> dict | None。

    Returns:
        节点函数，签名 (state: dict) -> dict。
    """
    if not fns:
        raise ValueError("至少需要提供一个函数。")

    def _node(state: dict) -> dict:
        with ThreadPoolExecutor(max_workers=len(fns)) as executor:
            futures = [executor.submit(fn, dict(state)) for fn in fns]
            results: list[dict | None] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    raise RuntimeError(f"并行执行失败: {e}") from e

        merged: dict = {}
        for r in results:
            if r is not None:
                merged.update(r)
        return merged

    return _node


def with_timeout(
    seconds: float,
    fn: Callable[[dict], dict | None],
) -> Callable[[dict], dict | None]:
    """为节点函数添加超时控制。

    如果函数执行超过指定秒数，抛出 TimeoutError。

    Args:
        seconds: 超时秒数。
        fn: 节点函数，签名 (state: dict) -> dict | None。

    Returns:
        包装后的节点函数。
    """
    def _wrapped(state: dict) -> dict | None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, state)
            try:
                return future.result(timeout=seconds)
            except TimeoutError:
                raise TimeoutError(
                    f"节点执行超时（{seconds}秒）"
                ) from None

    return _wrapped


def with_retry(
    fn: Callable[[dict], dict | None],
    *,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[dict], dict | None]:
    """为节点函数添加自动重试。

    函数执行失败时自动重试，支持指数退避。

    Args:
        fn: 节点函数，签名 (state: dict) -> dict | None。
        max_retries: 最大重试次数（不包括首次执行），默认 3。
        delay: 首次重试等待秒数，默认 1.0。
        backoff: 每次重试延迟倍数，默认 2.0（即 1s, 2s, 4s...）。

    Returns:
        包装后的节点函数。
    """
    def _wrapped(state: dict) -> dict | None:
        last_exception: Exception | None = None
        current_delay = delay
        for attempt in range(max_retries + 1):
            try:
                return fn(state)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(current_delay)
                    current_delay *= backoff
        raise RuntimeError(
            f"节点执行失败（已重试 {max_retries} 次）: {last_exception}"
        ) from last_exception

    return _wrapped


@dataclass
class Edge:
    """普通边：从一个节点无条件指向另一个节点。

    Attributes:
        source: 源节点名称。
        target: 目标节点名称。
    """
    source: str
    target: str


@dataclass(frozen=True)
class Send:
    """动态并行分支任务。

    Attributes:
        node: 分支需要执行的目标节点名称。
        arg: 合并到当前状态副本后传给目标节点的局部输入。
    """

    node: str
    arg: dict[str, Any]


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
    *,
    allow_cycles: bool = False,
) -> list[str]:
    """对图进行拓扑排序，返回节点执行顺序。

    将条件边的所有可能路径并入普通边进行保守校验。
    使用 Kahn 算法。允许环时，将无法拓扑排序的节点按名称追加到结果末尾。

    Args:
        node_names: 所有节点名称集合。
        edges: 普通边列表。
        conditional_edges: 条件边列表。
        allow_cycles: 是否允许图中存在环，默认 False。

    Returns:
        拓扑排序后的节点名称列表。

    Raises:
        CompileError: 图中存在环且 allow_cycles=False 时抛出。
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
    queue = sorted(name for name, deg in in_degree.items() if deg == 0)
    sorted_nodes: list[str] = []

    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in sorted(adj[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()

    if len(sorted_nodes) != len(node_names):
        if not allow_cycles:
            raise CompileError(
                f"图中检测到环，无法编译。"
                f"已排序 {len(sorted_nodes)}/{len(node_names)} 个节点。"
            )
        sorted_set = set(sorted_nodes)
        sorted_nodes.extend(sorted(node_names - sorted_set))

    return sorted_nodes


class StateGraph:
    """管理状态和节点的有向图。

    用户通过此类构建图结构，然后调用 compile() 得到可执行的 CompiledGraph。

    Args:
        state_class: 状态类型（继承自 State 的 TypedDict 子类）。
    """

    def __init__(self, state_class: type[State]) -> None:
        self._state_class = state_class
        self._nodes: dict[str, _NodeProtocol] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge] = []
        self._entry_point: str | None = None
        self._finish_point: str | None = None
        self._reducers: dict[str, Callable[[Any, Any], Any]] = _parse_reducers(state_class)

    def add_node(self, name: str, fn: Callable[..., Any]) -> None:
        """注册一个节点。

        Args:
            name: 节点名称，在图中唯一标识。
            fn: 可调用对象，签名 (state: dict) -> dict | None。
                支持同步和 async 函数。
        """
        self._nodes[name] = Node(name=name, fn=fn)

    def add_subgraph(
        self,
        name: str,
        subgraph: "CompiledGraph",
        state_mapping: dict[str, str] | None = None,
    ) -> None:
        """注册一个子图节点。

        Args:
            name: 节点名称，在图中唯一标识。
            subgraph: 编译后的子图。
            state_mapping: 可选的状态字段映射 {父字段: 子字段}。
        """
        self._nodes[name] = SubgraphNode(
            name=name,
            subgraph=subgraph,
            state_mapping=state_mapping,
        )

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

    def compile(self, *, allow_cycles: bool = False) -> "CompiledGraph":
        """编译图：校验合法性，生成拓扑排序，返回 CompiledGraph。

        Args:
            allow_cycles: 是否允许图中存在环。Agent 循环场景需设为 True。

        Returns:
            可执行的 CompiledGraph 实例。

        Raises:
            CompileError: 图不合法，或存在环且 allow_cycles=False 时抛出。
        """
        self._validate()

        # 校验通过后，入口/出口一定不为 None
        assert self._entry_point is not None
        assert self._finish_point is not None

        node_names = set(self._nodes.keys())
        execution_order = _topological_sort(
            node_names,
            self._edges,
            self._conditional_edges,
            allow_cycles=allow_cycles,
        )

        return CompiledGraph(
            state_class=self._state_class,
            nodes=self._nodes,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            entry_point=self._entry_point,
            finish_point=self._finish_point,
            execution_order=execution_order,
            reducers=self._reducers,
        )


class CompiledGraph:
    """编译后的可执行图。

    由 StateGraph.compile() 返回，持有编译后的所有数据和执行顺序。
    """

    def __init__(
        self,
        state_class: type[State],
        nodes: dict[str, _NodeProtocol],
        edges: list[Edge],
        conditional_edges: list[ConditionalEdge],
        entry_point: str,
        finish_point: str,
        execution_order: list[str],
        reducers: dict[str, Callable[[Any, Any], Any]] | None = None,
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

        # 自定义归约器
        self._reducers: dict[str, Callable[[Any, Any], Any]] = reducers or {}

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
        config: RunConfig | None = None,
        callback: Callable[[str, dict], None] | None = None,
    ) -> dict:
        """执行图，从入口节点开始，按拓扑顺序执行，到达出口节点后停止。

        Args:
            input: 初始状态字典。
            persistence: 可选的持久化存储实例。启用后每步执行后自动保存 checkpoint。
            thread_id: 线程 ID，用于 checkpoint 关联。不传时自动生成。
            config: 可选的运行配置。thread_id 参数优先于 config.thread_id。
            callback: 每步执行后的回调，签名 (node_name, state) -> None。

        Returns:
            执行完成后的最终状态字典。
        """
        for _node_name, _state in self._run_steps(
            input,
            persistence=persistence,
            thread_id=thread_id,
            config=config,
        ):
            if callback is not None:
                callback(_node_name, _state)
        return _state

    def stream(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
        config: RunConfig | None = None,
    ) -> Generator[tuple[str, dict], None, dict]:
        """流式执行图，逐节点输出 (node_name, state) 状态变化。

        Args:
            input: 初始状态字典。
            persistence: 可选的持久化存储实例。
            thread_id: 线程 ID，用于 checkpoint 关联。
            config: 可选的运行配置。thread_id 参数优先于 config.thread_id。

        Yields:
            (node_name, state) 元组，每次节点执行后输出。

        Returns:
            执行完成后的最终状态字典。
        """
        final_state: dict = None  # type: ignore[assignment]
        for name, state in self._run_steps(
            input,
            persistence=persistence,
            thread_id=thread_id,
            config=config,
        ):
            final_state = state
            yield name, state
        assert final_state is not None
        return final_state

    def _run_steps(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
        config: RunConfig | None = None,
    ) -> Generator[tuple[str, dict], None, dict]:
        """内部生成器：逐步骤执行，产出 (node_name, state) 元组。"""
        state = dict(input)
        current: str | None = self._entry_point
        run_config = _resolve_run_config(config, thread_id)
        tid = run_config.thread_id or uuid.uuid4().hex
        step = 0
        persistence_enabled = persistence is not None

        while current is not None:
            if step >= run_config.recursion_limit:
                raise GraphRecursionError(
                    f"图执行超过最大步数 {run_config.recursion_limit}，"
                    f"当前节点为 '{current}'。"
                )
            step += 1

            # 保存 checkpoint（在执行节点前保存，崩溃后可恢复到此节点）
            if persistence_enabled:
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
                yield current, state
                return state

            # 执行当前节点
            node = self._nodes[current]
            try:
                updates = node.run(state)
            except NodeInterrupt:
                if persistence_enabled:
                    final_cp = Checkpoint(
                        thread_id=tid,
                        step=step,
                        state=state,
                        next_node=current,
                        reason="interrupt",
                    )
                    persistence.put(tid, final_cp)
                yield current, state
                return state

            state = _merge_state(state, updates, self._reducers)

            # 产出此步骤的结果
            yield current, state

            # 到达出口节点则停止
            if current == self._finish_point:
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
        config: RunConfig | None = None,
    ) -> dict:
        """从指定线程的 checkpoint 恢复执行。

        加载最近一次 checkpoint，从中断处继续执行。

        Args:
            thread_id: 线程 ID。
            persistence: 持久化存储实例。
            command: 可选的人类指令，可更新状态、重定向节点或终止执行。
            config: 可选的运行配置。recursion_limit 限制本次恢复执行的节点步数。

        Returns:
            执行完成后的最终状态字典。

        Raises:
            ValueError: 指定线程的 checkpoint 不存在时抛出。
            GraphRecursionError: 本次恢复执行超过最大节点步数时抛出。
        """
        run_config = _resolve_run_config(config, thread_id)
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
                state = _merge_state(state, command.update, self._reducers)
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
        executed_steps = 0

        while current is not None:
            if executed_steps >= run_config.recursion_limit:
                raise GraphRecursionError(
                    f"图恢复执行超过最大步数 {run_config.recursion_limit}，"
                    f"当前节点为 '{current}'。"
                )
            executed_steps += 1

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
            state = _merge_state(state, updates, self._reducers)

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

    async def ainvoke(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
        config: RunConfig | None = None,
        callback: Callable[[str, dict], None] | None = None,
    ) -> dict:
        """异步执行图，从入口节点开始，按拓扑顺序执行，到达出口节点后停止。

        Args:
            input: 初始状态字典。
            persistence: 可选的持久化存储实例。启用后每步执行后自动保存 checkpoint。
            thread_id: 线程 ID，用于 checkpoint 关联。不传时自动生成。
            config: 可选的运行配置。thread_id 参数优先于 config.thread_id。
            callback: 每步执行后的回调，签名 (node_name, state) -> None。

        Returns:
            执行完成后的最终状态字典。
        """
        async for _node_name, _state in self._arun_steps(
            input,
            persistence=persistence,
            thread_id=thread_id,
            config=config,
        ):
            if callback is not None:
                callback(_node_name, _state)
        return _state

    async def astream(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
        config: RunConfig | None = None,
    ) -> AsyncGenerator[tuple[str, dict], None]:
        """异步流式执行图，逐节点输出 (node_name, state) 状态变化。

        Args:
            input: 初始状态字典。
            persistence: 可选的持久化存储实例。
            thread_id: 线程 ID，用于 checkpoint 关联。
            config: 可选的运行配置。thread_id 参数优先于 config.thread_id。

        Yields:
            (node_name, state) 元组，每次节点执行后输出。

        Note:
            异步生成器不返回最终值；最后一次产出的 state 即最终状态。
        """
        final_state: dict = None  # type: ignore[assignment]
        async for name, state in self._arun_steps(
            input,
            persistence=persistence,
            thread_id=thread_id,
            config=config,
        ):
            final_state = state
            yield name, state
        assert final_state is not None
        # async generator cannot return a value, but we don't need it anyway
        return

    async def _arun_steps(
        self,
        input: dict,
        *,
        persistence: "BasePersistence | None" = None,
        thread_id: str | None = None,
        config: RunConfig | None = None,
    ) -> AsyncGenerator[tuple[str, dict], None]:
        """内部异步生成器：逐步骤执行，产出 (node_name, state) 元组。"""
        state = dict(input)
        current: str | None = self._entry_point
        run_config = _resolve_run_config(config, thread_id)
        tid = run_config.thread_id or uuid.uuid4().hex
        step = 0
        persistence_enabled = persistence is not None

        while current is not None:
            if step >= run_config.recursion_limit:
                raise GraphRecursionError(
                    f"图执行超过最大步数 {run_config.recursion_limit}，"
                    f"当前节点为 '{current}'。"
                )
            step += 1

            # 保存 checkpoint（在执行节点前保存，崩溃后可恢复到此节点）
            if persistence_enabled:
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
                yield current, state
                return

            # 执行当前节点
            node = self._nodes[current]
            try:
                result = node.run(state)
                if inspect.iscoroutine(result):
                    updates = await result
                else:
                    updates = result
            except NodeInterrupt:
                if persistence_enabled:
                    final_cp = Checkpoint(
                        thread_id=tid,
                        step=step,
                        state=state,
                        next_node=current,
                        reason="interrupt",
                    )
                    persistence.put(tid, final_cp)
                yield current, state
                return

            state = _merge_state(state, updates, self._reducers)

            # 产出此步骤的结果
            yield current, state

            # 到达出口节点则停止
            if current == self._finish_point:
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

        return

    async def aresume(
        self,
        thread_id: str,
        persistence: "BasePersistence",
        *,
        command: "Command | None" = None,
        config: RunConfig | None = None,
    ) -> dict:
        """异步从指定线程的 checkpoint 恢复执行。

        加载最近一次 checkpoint，从中断处继续执行。

        Args:
            thread_id: 线程 ID。
            persistence: 持久化存储实例。
            command: 可选的人类指令，可更新状态、重定向节点或终止执行。
            config: 可选的运行配置。recursion_limit 限制本次恢复执行的节点步数。

        Returns:
            执行完成后的最终状态字典。

        Raises:
            ValueError: 指定线程的 checkpoint 不存在时抛出。
            GraphRecursionError: 本次恢复执行超过最大节点步数时抛出。
        """
        run_config = _resolve_run_config(config, thread_id)
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
                state = _merge_state(state, command.update, self._reducers)
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
        executed_steps = 0

        while current is not None:
            if executed_steps >= run_config.recursion_limit:
                raise GraphRecursionError(
                    f"图恢复执行超过最大步数 {run_config.recursion_limit}，"
                    f"当前节点为 '{current}'。"
                )
            executed_steps += 1

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
            result = node.run(state)
            if inspect.iscoroutine(result):
                updates = await result
            else:
                updates = result
            state = _merge_state(state, updates, self._reducers)

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
