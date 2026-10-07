"""Operation behavior shared by dataset plans and their cursors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from . import cursor as cursors
from .cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from .cursor import Cursor

if TYPE_CHECKING:
    from .dataset import Dataset, IndexedDataset, RangeDataset


@runtime_checkable
class _Operation(Protocol):
    @property
    def description(self) -> str: ...


@runtime_checkable
class _StreamOperation[T, U](_Operation, Protocol):
    def cardinality(self, parent: Cardinality) -> Cardinality: ...

    def open(self, dataset: Dataset[U], parent: Cursor[T]) -> Cursor[U]: ...


@runtime_checkable
class _ExactOperation[T, U](_Operation, Protocol):
    def exact_cardinality(self, parent: Exact) -> Exact: ...

    def get(self, parent: IndexedDataset[T], position: int) -> U: ...

    def open_range(
        self,
        dataset: RangeDataset[U],
        parent: RangeDataset[T],
        start: int,
        stop: int,
    ) -> Cursor[U]: ...


@dataclass(frozen=True, slots=True)
class _Map[T, U]:
    fn: Callable[[T], U]
    name: str

    @property
    def description(self) -> str:
        return f"Map(name={self.name!r})"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        return parent

    def exact_cardinality(self, parent: Exact) -> Exact:
        return parent

    def get(self, parent: IndexedDataset[T], position: int) -> U:
        return self.fn(parent._get(position))

    def open(self, dataset: Dataset[U], parent: Cursor[T]) -> Cursor[U]:
        return cursors.TransformCursor(
            dataset,
            parent,
            map(self.fn, parent),
        )

    def open_range(
        self,
        dataset: RangeDataset[U],
        parent: RangeDataset[T],
        start: int,
        stop: int,
    ) -> Cursor[U]:
        parent_cursor = parent.open_range(start, stop)
        return cursors.TransformCursor(
            dataset,
            parent_cursor,
            map(self.fn, parent_cursor),
        )


@dataclass(frozen=True, slots=True)
class _ParallelMap[T, U]:
    fn: Callable[[T], U]
    name: str
    workers: int
    buffer_size: int
    backend: str

    @property
    def description(self) -> str:
        return (
            f"ParallelMap(name={self.name!r}, workers={self.workers}, "
            f"buffer_size={self.buffer_size}, backend={self.backend!r})"
        )

    def cardinality(self, parent: Cardinality) -> Cardinality:
        return parent

    def open(self, dataset: Dataset[U], parent: Cursor[T]) -> Cursor[U]:
        return cursors.ParallelMapCursor(
            dataset,
            parent,
            self.fn,
            self.workers,
            self.buffer_size,
            self.backend,
        )


@dataclass(frozen=True, slots=True)
class _Filter[T]:
    predicate: Callable[[T], bool]
    name: str

    @property
    def description(self) -> str:
        return f"Filter(name={self.name!r})"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        if isinstance(parent, Exact):
            return Bounds(0, parent)
        if isinstance(parent, Bounds):
            return Bounds(0, parent.upper)
        return Unknown()

    def open(self, dataset: Dataset[T], parent: Cursor[T]) -> Cursor[T]:
        return cursors.TransformCursor(
            dataset,
            parent,
            filter(self.predicate, parent),
        )


@dataclass(frozen=True, slots=True)
class _FlatMap[T, U]:
    fn: Callable[[T], Iterable[U]]
    name: str

    @property
    def description(self) -> str:
        return f"FlatMap(name={self.name!r})"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        return Unknown()

    def open(self, dataset: Dataset[U], parent: Cursor[T]) -> Cursor[U]:
        return cursors.FlatMapCursor(dataset, parent, self.fn)


@dataclass(frozen=True, slots=True)
class _Take[T]:
    count: int

    @property
    def description(self) -> str:
        return f"Take(count={self.count})"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        if isinstance(parent, Exact):
            return Exact(min(parent, self.count))
        if isinstance(parent, Bounds):
            return Bounds(min(parent.lower, self.count), min(parent.upper, self.count))
        if isinstance(parent, Infinite):
            return Exact(self.count)
        return Bounds(0, self.count)

    def open(self, dataset: Dataset[T], parent: Cursor[T]) -> Cursor[T]:
        return cursors.TakeCursor(dataset, parent, self.count)


@dataclass(frozen=True, slots=True)
class _Skip[T]:
    count: int

    @property
    def description(self) -> str:
        return f"Skip(count={self.count})"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        if isinstance(parent, Exact):
            return Exact(max(0, parent - self.count))
        if isinstance(parent, Bounds):
            return Bounds(
                max(0, parent.lower - self.count),
                max(0, parent.upper - self.count),
            )
        return parent

    def open(self, dataset: Dataset[T], parent: Cursor[T]) -> Cursor[T]:
        return cursors.SkipCursor(dataset, parent, self.count)


@dataclass(frozen=True, slots=True)
class _Batch[T]:
    size: int
    drop_last: bool

    @property
    def description(self) -> str:
        return f"Batch(size={self.size}, drop_last={self.drop_last})"

    def _count(self, count: int) -> int:
        return count // self.size if self.drop_last else (count + self.size - 1) // self.size

    def cardinality(self, parent: Cardinality) -> Cardinality:
        if isinstance(parent, Exact):
            return Exact(self._count(parent))
        if isinstance(parent, Bounds):
            return Bounds(self._count(parent.lower), self._count(parent.upper))
        return parent

    def exact_cardinality(self, parent: Exact) -> Exact:
        return Exact(self._count(parent))

    def get(self, parent: IndexedDataset[T], position: int) -> tuple[T, ...]:
        start = position * self.size
        stop = min(start + self.size, len(parent))
        return tuple(parent._get(index) for index in range(start, stop))

    def open(
        self,
        dataset: Dataset[tuple[T, ...]],
        parent: Cursor[T],
    ) -> Cursor[tuple[T, ...]]:
        return cursors.BatchCursor(dataset, parent, self.size, self.drop_last)

    def open_range(
        self,
        dataset: RangeDataset[tuple[T, ...]],
        parent: RangeDataset[T],
        start: int,
        stop: int,
    ) -> Cursor[tuple[T, ...]]:
        parent_start = start * self.size
        parent_stop = min(stop * self.size, len(parent))
        parent_cursor = parent.open_range(parent_start, parent_stop)
        return cursors.BatchCursor(dataset, parent_cursor, self.size, self.drop_last)
