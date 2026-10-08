from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from operator import index as to_index
from typing import Any, Protocol, runtime_checkable

from ..cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from ..cursor import Cursor, State, TransformCursor, _CloseSlot, _ParentCursor
from ..dataset import Dataset, IndexedDataset, RangeDataset


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


class _UnaryNode[T]:
    parent: Dataset[T]
    operation: _Operation

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def description(self) -> str:
        return self.operation.description


@dataclass(frozen=True, slots=True)
class _UnaryRange[T, U](_UnaryNode[T], RangeDataset[U]):
    parent: RangeDataset[T]
    operation: _ExactOperation[T, U]

    @property
    def cardinality(self) -> Exact:
        return self.operation.exact_cardinality(self.parent.cardinality)

    def _open_range(self, start: int, stop: int) -> Cursor[U]:
        return self.operation.open_range(self, self.parent, start, stop)


@dataclass(frozen=True, slots=True)
class _UnaryIndexed[T, U](_UnaryNode[T], IndexedDataset[U]):
    parent: IndexedDataset[T]
    operation: _ExactOperation[T, U]

    @property
    def cardinality(self) -> Exact:
        return self.operation.exact_cardinality(self.parent.cardinality)

    def _get(self, position: int) -> U:
        return self.operation.get(self.parent, position)


@dataclass(frozen=True, slots=True)
class _UnaryDataset[T, U](_UnaryNode[T], Dataset[U]):
    parent: Dataset[T]
    operation: _StreamOperation[T, U]

    @property
    def cardinality(self) -> Cardinality:
        return self.operation.cardinality(self.parent.cardinality)

    def cursor(self) -> Cursor[U]:
        return self.operation.open(self, self.parent.cursor())


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
        return TransformCursor(dataset, parent, map(self.fn, parent))

    def open_range(
        self,
        dataset: RangeDataset[U],
        parent: RangeDataset[T],
        start: int,
        stop: int,
    ) -> Cursor[U]:
        parent_cursor = parent.open_range(start, stop)
        return TransformCursor(dataset, parent_cursor, map(self.fn, parent_cursor))


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
        return TransformCursor(dataset, parent, filter(self.predicate, parent))


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
        return _FlatMapCursor(dataset, parent, self.fn)


@dataclass(frozen=True, slots=True)
class _Unbatch[T]:
    @property
    def description(self) -> str:
        return "Unbatch()"

    def cardinality(self, parent: Cardinality) -> Cardinality:
        return Unknown()

    def open(self, dataset: Dataset[T], parent: Cursor[Iterable[T]]) -> Cursor[T]:
        return _FlatMapCursor(dataset, parent, iter)


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
        return _TakeCursor(dataset, parent, self.count)


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
        return _SkipCursor(dataset, parent, self.count)


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
        return _BatchCursor(dataset, parent, self.size, self.drop_last)

    def open_range(
        self,
        dataset: RangeDataset[tuple[T, ...]],
        parent: RangeDataset[T],
        start: int,
        stop: int,
    ) -> Cursor[tuple[T, ...]]:
        parent_start = start * self.size
        parent_stop = min(stop * self.size, len(parent))
        return _BatchCursor(
            dataset,
            parent.open_range(parent_start, parent_stop),
            self.size,
            self.drop_last,
        )


class _FlatMapCursor[T, U](_ParentCursor[T, U]):
    def __init__(
        self,
        dataset: Dataset[U],
        parent: Cursor[T],
        fn: Callable[[T], Iterable[U]],
    ):
        super().__init__(dataset, parent)
        self._fn = fn
        self._current: T | None = None
        self._has_current = False
        self._child: Iterator[U] | None = None
        self._child_offset = 0
        self._child_resource = _CloseSlot()
        self.callback(self._child_resource.close)

    def _load_local_state(self, state: State) -> None:
        if state["has_current"]:
            self._current = state["current"]
            self._has_current = True
            child = self._start_child(self._current)
            offset = to_index(state["child_offset"])
            if offset < 0:
                raise ValueError("flat_map child offset must be nonnegative")
            for _ in range(offset):
                try:
                    next(child)
                except StopIteration as error:
                    raise ValueError("flat_map checkpoint no longer matches output") from error
            self._child_offset = offset

    def _start_child(self, value: T) -> Iterator[U]:
        self._child_resource.close()
        self._child = iter(self._fn(value))
        self._child_resource.replace(self._child)
        self._child_offset = 0
        return self._child

    def _next(self) -> U:
        while True:
            if self._child is not None:
                try:
                    value = next(self._child)
                except StopIteration:
                    self._child_resource.close()
                    self._child = None
                    self._current = None
                    self._has_current = False
                else:
                    self._child_offset += 1
                    return value

            self._current = next(self._parent)
            self._has_current = True
            self._start_child(self._current)

    def _local_state(self) -> State:
        return {
            "has_current": self._has_current,
            "current": self._current,
            "child_offset": self._child_offset,
        }


class _TakeCursor[T](_ParentCursor[T, T]):
    def __init__(self, dataset: Dataset[T], parent: Cursor[T], remaining: int):
        super().__init__(dataset, parent)
        self._limit = remaining
        self._remaining = remaining

    def _next(self) -> T:
        if self._remaining == 0:
            raise StopIteration
        value = next(self._parent)
        self._remaining -= 1
        return value

    def _local_state(self) -> State:
        return {"remaining": self._remaining}

    def _load_local_state(self, state: State) -> None:
        remaining = to_index(state["remaining"])
        if not 0 <= remaining <= self._limit:
            raise ValueError("take checkpoint has an invalid remaining count")
        self._remaining = remaining


class _SkipCursor[T](_ParentCursor[T, T]):
    def __init__(self, dataset: Dataset[T], parent: Cursor[T], remaining: int):
        super().__init__(dataset, parent)
        self._limit = remaining
        self._remaining = remaining

    def _next(self) -> T:
        while self._remaining:
            next(self._parent)
            self._remaining -= 1
        return next(self._parent)

    def _local_state(self) -> State:
        return {"remaining": self._remaining}

    def _load_local_state(self, state: State) -> None:
        remaining = to_index(state["remaining"])
        if not 0 <= remaining <= self._limit:
            raise ValueError("skip checkpoint has an invalid remaining count")
        self._remaining = remaining


class _BatchCursor[T](_ParentCursor[T, tuple[T, ...]]):
    def __init__(
        self,
        dataset: Dataset[tuple[T, ...]],
        parent: Cursor[T],
        size: int,
        drop_last: bool,
    ):
        super().__init__(dataset, parent)
        self._size = size
        self._drop_last = drop_last
        self._buffer: list[T] = []

    def _next(self) -> tuple[T, ...]:
        while len(self._buffer) < self._size:
            try:
                self._buffer.append(next(self._parent))
            except StopIteration:
                if not self._buffer or self._drop_last:
                    self._buffer.clear()
                    raise
                batch = tuple(self._buffer)
                self._buffer.clear()
                self.close()
                return batch
        batch = tuple(self._buffer)
        self._buffer.clear()
        return batch

    def _local_state(self) -> State:
        return {"buffer": list(self._buffer)}

    def _load_local_state(self, state: State) -> None:
        buffer = list(state["buffer"])
        if len(buffer) >= self._size:
            raise ValueError("batch checkpoint has an invalid buffer")
        self._buffer = buffer
