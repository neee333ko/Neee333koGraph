"""跨会话长期记忆存储。

Store 与 Checkpoint 的职责不同：
- Checkpoint 保存单次图执行过程，支持暂停与恢复。
- Store 保存跨执行、跨线程共享的业务记忆。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy
import json
import sqlite3
import threading
import time


Namespace = tuple[str, ...]


@dataclass(frozen=True)
class StoreItem:
    """长期记忆条目。"""

    namespace: Namespace
    key: str
    value: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class BaseStore(ABC):
    """长期记忆存储抽象接口。"""

    @abstractmethod
    def put(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
    ) -> StoreItem:
        """新增或覆盖一条记忆。"""
        ...

    @abstractmethod
    def get(self, namespace: Namespace, key: str) -> StoreItem | None:
        """读取指定记忆。"""
        ...

    @abstractmethod
    def delete(self, namespace: Namespace, key: str) -> bool:
        """删除指定记忆并返回是否存在。"""
        ...

    @abstractmethod
    def search(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        """按命名空间、文本和字段过滤检索记忆。"""
        ...


def _matches(
    item: StoreItem,
    query: str | None,
    filters: dict[str, Any] | None,
) -> bool:
    """判断记忆条目是否满足文本和字段过滤条件。"""
    if filters and any(item.value.get(key) != value for key, value in filters.items()):
        return False
    if query:
        haystack = f"{item.key} {json.dumps(item.value, ensure_ascii=False)}".lower()
        if query.lower() not in haystack:
            return False
    return True


class InMemoryStore(BaseStore):
    """线程安全的内存长期记忆存储。"""

    def __init__(self) -> None:
        self._items: dict[tuple[Namespace, str], StoreItem] = {}
        self._lock = threading.RLock()

    def put(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
    ) -> StoreItem:
        if not namespace:
            raise ValueError("namespace 不能为空。")
        if not key:
            raise ValueError("key 不能为空。")
        with self._lock:
            storage_key = (tuple(namespace), key)
            previous = self._items.get(storage_key)
            now = time.time()
            item = StoreItem(
                namespace=tuple(namespace),
                key=key,
                value=copy.deepcopy(value),
                created_at=previous.created_at if previous else now,
                updated_at=now,
            )
            self._items[storage_key] = item
            return copy.deepcopy(item)

    def get(self, namespace: Namespace, key: str) -> StoreItem | None:
        with self._lock:
            item = self._items.get((tuple(namespace), key))
            return copy.deepcopy(item) if item is not None else None

    def delete(self, namespace: Namespace, key: str) -> bool:
        with self._lock:
            return self._items.pop((tuple(namespace), key), None) is not None

    def search(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        if limit <= 0:
            return []
        with self._lock:
            items = [
                item
                for (item_namespace, _), item in self._items.items()
                if item_namespace == tuple(namespace)
                and _matches(item, query, filters)
            ]
            items.sort(key=lambda item: item.updated_at, reverse=True)
            return copy.deepcopy(items[:limit])


class SQLiteStore(BaseStore):
    """基于 SQLite 的长期记忆存储。"""

    def __init__(self, database: str | Path = "./memory.db") -> None:
        self._database = str(database)
        if self._database != ":memory:":
            Path(self._database).expanduser().parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._database,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_namespace_updated
                ON memories (namespace, updated_at DESC)
                """
            )

    def put(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
    ) -> StoreItem:
        if not namespace:
            raise ValueError("namespace 不能为空。")
        if not key:
            raise ValueError("key 不能为空。")
        namespace_json = json.dumps(tuple(namespace), ensure_ascii=False)
        value_json = json.dumps(value, ensure_ascii=False)
        now = time.time()
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT created_at
                FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_json, key),
            ).fetchone()
            created_at = row["created_at"] if row else now
            self._connection.execute(
                """
                INSERT INTO memories (
                    namespace, key, value_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (namespace_json, key, value_json, created_at, now),
            )
        return StoreItem(tuple(namespace), key, copy.deepcopy(value), created_at, now)

    def get(self, namespace: Namespace, key: str) -> StoreItem | None:
        namespace_json = json.dumps(tuple(namespace), ensure_ascii=False)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_json, key),
            ).fetchone()
        return self._row_to_item(row) if row is not None else None

    def delete(self, namespace: Namespace, key: str) -> bool:
        namespace_json = json.dumps(tuple(namespace), ensure_ascii=False)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM memories
                WHERE namespace = ? AND key = ?
                """,
                (namespace_json, key),
            )
            return cursor.rowcount > 0

    def search(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        if limit <= 0:
            return []
        namespace_json = json.dumps(tuple(namespace), ensure_ascii=False)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE namespace = ?
                ORDER BY updated_at DESC
                """,
                (namespace_json,),
            ).fetchall()
        items = [self._row_to_item(row) for row in rows]
        return [
            item
            for item in items
            if _matches(item, query, filters)
        ][:limit]

    def _row_to_item(self, row: sqlite3.Row) -> StoreItem:
        """将 SQLite 行转换为 StoreItem。"""
        return StoreItem(
            namespace=tuple(json.loads(row["namespace"])),
            key=row["key"],
            value=json.loads(row["value_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        with self._lock:
            self._connection.close()
