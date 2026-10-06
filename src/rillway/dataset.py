"""Immutable dataset plans.

Transformations build private plan nodes; calling ``cursor()`` turns those nodes
into the stateful executors in ``cursor.py``. Unary nodes share graph metadata
and delegate mode-specific execution to their operation object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import accumulate
from operator import index as to_index
from typing import Any, ClassVar, Literal, Protocol, cast, overload, runtime_checkable

from . import cursor as cursors
from ._operation import (
    _Batch,
    _ExactOperation,
    _Filter,
    _FlatMap,
    _Map,
    _Operation,
    _ParallelMap,
    _Skip,
    _StreamOperation,
    _Take,
)
from .cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from .cursor import Cursor


class Dataset[T](ABC):
    """Functional callbacks are treated as stateless and deterministic. Put
    stateful processing in a custom Cursor so its state can be checkpointed.
    """

    supports_checkpointing: ClassVar[bool] = False

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return ()

    @property
    @abstractmethod
    def cardinality(self) -> Cardinality: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def checkpointable(self) -> bool:
        return self.supports_checkpointing and all(parent.checkpointable for parent in self.parents)

    @abstractmethod
    def cursor(self) -> Cursor[T]: ...

    def __iter__(self) -> Cursor[T]:
        return self.cursor()

    def explain(self) -> str:
        return "\n".join(_explain(self))

    def map[U](self, fn: Callable[[T], U], *, name: str | None = None) -> Dataset[U]:
        return _UnaryDataset(self, _Map(fn, _callable_name(fn, name)))

    def parallel_map[U](
        self,
        fn: Callable[[T], U],
        *,
        workers: int,
        buffer_size: int | None = None,
        backend: Literal["thread", "process"] = "thread",
        name: str | None = None,
    ) -> Dataset[U]:
        """Apply ``fn`` concurrently and yield results in input order.

        ``buffer_size`` controls how many calls may wait beyond the active
        workers. It defaults to the worker count. Process workers require the
        function, inputs, and outputs to be picklable.
        """
        workers = _positive("workers", workers)
        if backend not in ("thread", "process"):
            raise ValueError(f"unsupported parallel backend: {backend!r}")
        buffer_size = (
            workers
            if buffer_size is None
            else _nonnegative("buffer_size", buffer_size)
        )
        return _ParallelMapDataset(
            self,
            _ParallelMap(
                fn,
                _callable_name(fn, name),
                workers,
                buffer_size,
                backend,
            ),
        )

    def prefetch(self, buffer_size: int) -> Dataset[T]:
        return _PrefetchDataset(self, _positive("buffer_size", buffer_size))

    def filter(self, predicate: Callable[[T], bool], *, name: str | None = None) -> Dataset[T]:
        return _UnaryDataset(self, _Filter(predicate, _callable_name(predicate, name)))

    def flat_map[U](self, fn: Callable[[T], Iterable[U]], *, name: str | None = None) -> Dataset[U]:
        return _UnaryDataset(self, _FlatMap(fn, _callable_name(fn, name)))

    def take(self, count: int) -> Dataset[T]:
        return _UnaryDataset(self, _Take(_nonnegative("count", count)))

    def skip(self, count: int) -> Dataset[T]:
        return _UnaryDataset(self, _Skip(_nonnegative("count", count)))

    def batch(self, size: int, *, drop_last: bool = False) -> Dataset[tuple[T, ...]]:
        return _UnaryDataset(self, _Batch(_positive("size", size), drop_last))

    def repeat(self, count: int | None) -> Dataset[T]:
        return _repeat_dataset(self, count)

    def concat(self, *others: Dataset[T]) -> Dataset[T]:
        return _concat_datasets(self, others)

    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]:
        return _zip_datasets(self, other, strict)

    @classmethod
    def from_factory[U](
        cls,
        factory: Callable[[], Iterable[U]],
        *,
        cardinality: Cardinality | None = None,
        name: str | None = None,
    ) -> Dataset[U]:
        """Build a replayable dataset by creating a fresh iterable per traversal."""
        if not callable(factory):
            raise TypeError("factory must be callable")
        return _FactoryDataset(
            factory,
            Unknown() if cardinality is None else cardinality,
            _callable_name(factory, name),
        )

    @classmethod
    def from_cursor_factory[U](
        cls,
        factory: Callable[[], Cursor[U]],
        *,
        cardinality: Cardinality | None = None,
        name: str | None = None,
    ) -> Dataset[U]:
        """Build a checkpointable dataset from fresh stateful cursors.

        The cursor implements ``_state_dict`` and ``_load_state_dict`` for its
        local payload; the public methods add and validate checkpoint metadata.
        """
        if not callable(factory):
            raise TypeError("factory must be callable")
        return _CursorFactoryDataset(
            factory,
            Unknown() if cardinality is None else cardinality,
            _callable_name(factory, name),
        )


class RangeDataset[T](Dataset[T], ABC):
    @property
    @abstractmethod
    def cardinality(self) -> Exact: ...

    @abstractmethod
    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[T]: ...

    def __len__(self) -> int:
        return self.cardinality

    def open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[T]:
        start = to_index(start)
        stop = to_index(stop)
        if not 0 <= start <= stop <= len(self):
            raise ValueError("range must satisfy 0 <= start <= stop <= length")
        cursor = self._open_range(start, stop)
        if not isinstance(cursor, Cursor):
            raise TypeError("range reader must return a Cursor")
        return cursors.RangeCursor(cursor, start, stop)

    def cursor(self) -> Cursor[T]:
        return self.open_range(0, len(self))

    def shard(self, index: int, count: int) -> RangeDataset[T]:
        start, stop = _shard_bounds(len(self), index, count)
        return _RangeSlice(self, start, stop)

    def map[U](self, fn: Callable[[T], U], *, name: str | None = None) -> RangeDataset[U]:
        return _UnaryRange(self, _Map(fn, _callable_name(fn, name)))

    def take(self, count: int) -> RangeDataset[T]:
        return _RangeSlice(self, 0, min(_nonnegative("count", count), len(self)))

    def skip(self, count: int) -> RangeDataset[T]:
        start = min(_nonnegative("count", count), len(self))
        return _RangeSlice(self, start, len(self))

    def batch(self, size: int, *, drop_last: bool = False) -> RangeDataset[tuple[T, ...]]:
        return _UnaryRange(self, _Batch(_positive("size", size), drop_last))

    @overload  # type: ignore[override]
    def repeat(
        self,
        count: int,
    ) -> RangeDataset[T]: ...

    @overload
    def repeat(
        self,
        count: None,
    ) -> Dataset[T]: ...

    def repeat(self, count: int | None) -> Dataset[T]:
        return _repeat_dataset(self, count)

    @overload
    def concat(self, *others: RangeDataset[T]) -> RangeDataset[T]: ...

    @overload
    def concat(self, *others: Dataset[T]) -> Dataset[T]: ...

    def concat(self, *others: Dataset[T]) -> Dataset[T]:
        return _concat_datasets(self, others)

    @overload
    def zip[U](self, other: RangeDataset[U], *, strict: bool = False) -> RangeDataset[tuple[T, U]]: ...

    @overload
    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]: ...

    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]:
        return _zip_datasets(self, other, strict)


class IndexedDataset[T](RangeDataset[T], ABC):
    """IndexedDataset transformations must be deterministic functions of their input and
    fixed configuration. Stateful processing belongs in a custom Cursor.
    """

    supports_checkpointing = True

    @abstractmethod
    def _get(self, position: int) -> T: ...

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[T]:
        return cursors.IndexedCursor(self, start, stop)

    @classmethod
    def from_source[U](cls, source: IndexedSource[U]) -> IndexedDataset[U]:
        if not isinstance(source, IndexedSource):
            raise TypeError("indexed source must provide __len__ and __getitem__")
        return _SourceIndexed(source, len(source))

    @overload
    def __getitem__(self, position: int) -> T: ...

    @overload
    def __getitem__(self, position: slice) -> IndexedDataset[T]: ...

    def __getitem__(self, position: int | slice) -> T | IndexedDataset[T]:
        if isinstance(position, slice):
            return _SliceIndexed(self, range(*position.indices(len(self))))
        position = to_index(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(position)
        return self._get(position)

    def cursor(self) -> Cursor[T]:
        return cursors.IndexedCursor(self)

    def shard(self, index: int, count: int) -> IndexedDataset[T]:
        start, stop = _shard_bounds(len(self), index, count)
        return self[start:stop]

    def map[U](self, fn: Callable[[T], U], *, name: str | None = None) -> IndexedDataset[U]:
        return _UnaryIndexed(self, _Map(fn, _callable_name(fn, name)))

    def take(self, count: int) -> IndexedDataset[T]:
        return self[: _nonnegative("count", count)]

    def skip(self, count: int) -> IndexedDataset[T]:
        return self[_nonnegative("count", count) :]

    def batch(self, size: int, *, drop_last: bool = False) -> IndexedDataset[tuple[T, ...]]:
        return _UnaryIndexed(self, _Batch(_positive("size", size), drop_last))

    @overload  # type: ignore[override]
    def repeat(
        self,
        count: int,
        *,
        shuffle: bool = False,
        seed: int = 42,
    ) -> IndexedDataset[T]: ...

    @overload
    def repeat(
        self,
        count: None,
        *,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Dataset[T]: ...

    def repeat(
        self,
        count: int | None,
        *,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Dataset[T]:
        return _repeat_dataset(self, count, shuffle=shuffle, seed=seed)

    @overload  # type: ignore[override]
    def concat(
        self,
        *others: IndexedDataset[T],
    ) -> IndexedDataset[T]: ...

    @overload
    def concat(self, *others: RangeDataset[T]) -> RangeDataset[T]: ...

    @overload
    def concat(self, *others: Dataset[T]) -> Dataset[T]: ...

    def concat(self, *others: Dataset[T]) -> Dataset[T]:
        return _concat_datasets(self, others)

    @overload  # type: ignore[override]
    def zip[U](
        self,
        other: IndexedDataset[U],
        *,
        strict: bool = False,
    ) -> IndexedDataset[tuple[T, U]]: ...

    @overload
    def zip[U](self, other: RangeDataset[U], *, strict: bool = False) -> RangeDataset[tuple[T, U]]: ...

    @overload
    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]: ...

    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]:
        return _zip_datasets(self, other, strict)


class _UnaryNode[T]:
    parent: Dataset[T]
    operation: _Operation

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def description(self) -> str:
        return self.operation.description


@runtime_checkable
class IndexedSource[T](Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, position: int, /) -> T: ...


@dataclass(frozen=True, slots=True)
class _SourceIndexed[T](IndexedDataset[T]):
    source: IndexedSource[T]
    length: int

    @property
    def cardinality(self) -> Exact:
        return Exact(self.length)

    @property
    def description(self) -> str:
        return f"Source(type={type(self.source).__name__})"

    def _get(self, position: int) -> T:
        current_length = len(self.source)
        if current_length != self.length:
            raise RuntimeError(f"source size changed from {self.length} to {current_length}")
        return self.source[position]


@dataclass(frozen=True, slots=True)
class _SliceIndexed[T](IndexedDataset[T]):
    parent: IndexedDataset[T]
    positions: range

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return Exact(len(self.positions))

    @property
    def description(self) -> str:
        return f"Slice(start={self.positions.start}, stop={self.positions.stop}, step={self.positions.step})"

    def _get(self, position: int) -> T:
        return self.parent._get(self.positions[position])


@dataclass(frozen=True, slots=True)
class _RangeSlice[T](RangeDataset[T]):
    supports_checkpointing = True

    parent: RangeDataset[T]
    start: int
    stop: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return Exact(self.stop - self.start)

    @property
    def description(self) -> str:
        return f"Range(start={self.start}, stop={self.stop})"

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[T]:
        return self.parent.open_range(self.start + start, self.start + stop)


@dataclass(frozen=True, slots=True)
class _RepeatRange[T](RangeDataset[T]):
    supports_checkpointing = True

    parent: RangeDataset[T]
    count: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return Exact(len(self.parent) * self.count)

    @property
    def description(self) -> str:
        return _repeat_description(self.count, False, 42)

    def _open_range(self, start: int, stop: int) -> Cursor[T]:
        length = len(self.parent)
        if length == 0:
            return cursors.ConcatCursor(())
        components = []
        while start < stop:
            offset = start % length
            component_stop = min(length, offset + stop - start)
            components.append(_RangeSlice(self.parent, offset, component_stop))
            start += component_stop - offset
        return cursors.ConcatCursor(tuple(components))


@dataclass(frozen=True, slots=True)
class _RepeatIndexed[T](IndexedDataset[T]):
    parent: IndexedDataset[T]
    count: int
    shuffle: bool
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return Exact(len(self.parent) * self.count)

    @property
    def description(self) -> str:
        return _repeat_description(self.count, self.shuffle, self.seed)

    def _get(self, position: int) -> T:
        length = len(self.parent)
        epoch, position = divmod(position, length)
        if self.shuffle:
            position = _shuffle_position(position, length, self.seed, epoch)
        return self.parent._get(position)


@dataclass(frozen=True, slots=True)
class _ShuffledEpochIndexed[T](IndexedDataset[T]):
    parent: IndexedDataset[T]
    seed: int
    epoch: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return self.parent.cardinality

    @property
    def description(self) -> str:
        return f"Shuffle(seed={self.seed}, epoch={self.epoch})"

    def _get(self, position: int) -> T:
        position = _shuffle_position(position, len(self.parent), self.seed, self.epoch)
        return self.parent._get(position)


@dataclass(frozen=True, slots=True)
class _UnaryRange[T, U](_UnaryNode[T], RangeDataset[U]):
    supports_checkpointing = True

    parent: RangeDataset[T]
    operation: _ExactOperation[T, U]

    @property
    def cardinality(self) -> Exact:
        return self.operation.exact_cardinality(self.parent.cardinality)

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[U]:
        return self.operation.open_range(self.parent, start, stop)


@dataclass(frozen=True, slots=True)
class _ConcatRange[T](RangeDataset[T]):
    supports_checkpointing = True

    components: tuple[RangeDataset[T], ...]
    _ends: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ends", tuple(accumulate(map(len, self.components))))

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Exact:
        return Exact(self._ends[-1] if self._ends else 0)

    @property
    def description(self) -> str:
        return f"Concat(count={len(self.components)})"

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[T]:
        selected = []
        position = start
        while position < stop:
            component_index = bisect_right(self._ends, position)
            component_start = self._ends[component_index - 1] if component_index else 0
            component_stop = self._ends[component_index]
            local_stop = min(stop, component_stop) - component_start
            selected.append(
                _RangeSlice(
                    self.components[component_index],
                    position - component_start,
                    local_stop,
                )
            )
            position = component_stop
        return cursors.ConcatCursor(tuple(selected))


@dataclass(frozen=True, slots=True)
class _ZipRange[T, U](RangeDataset[tuple[T, U]]):
    supports_checkpointing = True

    left: RangeDataset[T]
    right: RangeDataset[U]
    strict: bool

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.left, self.right

    @property
    def cardinality(self) -> Exact:
        return Exact(min(len(self.left), len(self.right)))

    @property
    def description(self) -> str:
        return f"Zip(strict={self.strict})"

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[tuple[T, U]]:
        return cursors.ZipCursor(
            self.left.open_range(start, stop),
            self.right.open_range(start, stop),
            self.strict,
        )


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
    supports_checkpointing = True

    parent: Dataset[T]
    operation: _StreamOperation[T, U]

    @property
    def cardinality(self) -> Cardinality:
        return self.operation.cardinality(self.parent.cardinality)

    def cursor(self) -> Cursor[U]:
        return self.operation.open(self.parent.cursor())


@dataclass(frozen=True, slots=True)
class _ParallelMapDataset[T, U](_UnaryNode[T], Dataset[U]):
    parent: Dataset[T]
    operation: _ParallelMap[T, U]

    @property
    def cardinality(self) -> Cardinality:
        return self.operation.cardinality(self.parent.cardinality)

    def cursor(self) -> Cursor[U]:
        return self.operation.open(self.parent.cursor())


@dataclass(frozen=True, slots=True)
class _PrefetchDataset[T](Dataset[T]):
    parent: Dataset[T]
    buffer_size: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        return self.parent.cardinality

    @property
    def description(self) -> str:
        return f"Prefetch(buffer_size={self.buffer_size})"

    def cursor(self) -> Cursor[T]:
        return cursors.PrefetchCursor(self.parent.cursor(), self.buffer_size)


@dataclass(frozen=True, slots=True)
class _RepeatDataset[T](Dataset[T]):
    supports_checkpointing = True

    parent: Dataset[T]
    count: int | None
    shuffle: bool
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        return _repeat_cardinality(self.parent.cardinality, self.count)

    @property
    def description(self) -> str:
        return _repeat_description(self.count, self.shuffle, self.seed)

    def cursor(self) -> Cursor[T]:
        return cursors.RepeatCursor(
            self._open_epoch,
            self.count,
            self.explain(),
        )

    def _open_epoch(self, epoch: int) -> Cursor[T]:
        if self.shuffle:
            assert isinstance(self.parent, IndexedDataset)
            return _ShuffledEpochIndexed(self.parent, self.seed, epoch).cursor()
        return self.parent.cursor()


@dataclass(frozen=True, slots=True)
class _FactoryDataset[T](Dataset[T]):
    factory: Callable[[], Iterable[T]]
    _cardinality: Cardinality
    name: str

    @property
    def cardinality(self) -> Cardinality:
        return self._cardinality

    @property
    def description(self) -> str:
        return f"Factory(name={self.name!r})"

    def cursor(self) -> Cursor[T]:
        return Cursor.from_iterator(iter(self.factory()))


@dataclass(frozen=True, slots=True)
class _CursorFactoryDataset[T](Dataset[T]):
    supports_checkpointing = True

    factory: Callable[[], Cursor[T]]
    _cardinality: Cardinality
    name: str

    @property
    def cardinality(self) -> Cardinality:
        return self._cardinality

    @property
    def description(self) -> str:
        return f"CursorFactory(name={self.name!r})"

    def cursor(self) -> Cursor[T]:
        cursor = self.factory()
        if not isinstance(cursor, Cursor):
            raise TypeError("cursor factory must return a Cursor")
        return cursor


@dataclass(frozen=True, slots=True)
class _ConcatIndexed[T](IndexedDataset[T]):
    components: tuple[IndexedDataset[T], ...]
    _ends: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ends", tuple(accumulate(map(len, self.components))))

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Exact:
        return Exact(self._ends[-1] if self._ends else 0)

    @property
    def description(self) -> str:
        return f"Concat(count={len(self.components)})"

    def _get(self, position: int) -> T:
        component = bisect_right(self._ends, position)
        start = self._ends[component - 1] if component else 0
        return self.components[component]._get(position - start)


@dataclass(frozen=True, slots=True)
class _ConcatDataset[T](Dataset[T]):
    supports_checkpointing = True

    components: tuple[Dataset[T], ...]

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Cardinality:
        lower = upper = 0
        unknown = False
        for component in self.components:
            cardinality = component.cardinality
            if isinstance(cardinality, Infinite):
                return Infinite()
            if isinstance(cardinality, Unknown):
                unknown = True
            elif isinstance(cardinality, Exact):
                lower += cardinality
                upper += cardinality
            else:
                lower += cardinality.lower
                upper += cardinality.upper
        if unknown:
            return Unknown()
        return Exact(lower) if lower == upper else Bounds(lower, upper)

    @property
    def description(self) -> str:
        return f"Concat(count={len(self.components)})"

    def cursor(self) -> Cursor[T]:
        return cursors.ConcatCursor(self.components)


@dataclass(frozen=True, slots=True)
class _ZipIndexed[T, U](IndexedDataset[tuple[T, U]]):
    left: IndexedDataset[T]
    right: IndexedDataset[U]
    strict: bool

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.left, self.right

    @property
    def cardinality(self) -> Exact:
        return Exact(min(len(self.left), len(self.right)))

    @property
    def description(self) -> str:
        return f"Zip(strict={self.strict})"

    def _get(self, position: int) -> tuple[T, U]:
        return self.left._get(position), self.right._get(position)


@dataclass(frozen=True, slots=True)
class _ZipDataset[T, U](Dataset[tuple[T, U]]):
    supports_checkpointing = True

    left: Dataset[T]
    right: Dataset[U]
    strict: bool

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.left, self.right

    @property
    def cardinality(self) -> Cardinality:
        left = self.left.cardinality
        right = self.right.cardinality
        if isinstance(left, Infinite) and isinstance(right, Infinite):
            return Infinite()
        if isinstance(left, Infinite):
            return right
        if isinstance(right, Infinite):
            return left

        def bounds(cardinality: Cardinality) -> tuple[int, int] | None:
            if isinstance(cardinality, Exact):
                return cardinality, cardinality
            if isinstance(cardinality, Bounds):
                return cardinality.lower, cardinality.upper
            return None

        left_bounds, right_bounds = bounds(left), bounds(right)
        if left_bounds is None and right_bounds is None:
            return Unknown()
        if left_bounds is None:
            assert right_bounds is not None
            return Exact(0) if right_bounds[1] == 0 else Bounds(0, right_bounds[1])
        if right_bounds is None:
            return Exact(0) if left_bounds[1] == 0 else Bounds(0, left_bounds[1])
        lower = min(left_bounds[0], right_bounds[0])
        upper = min(left_bounds[1], right_bounds[1])
        return Exact(lower) if lower == upper else Bounds(lower, upper)

    @property
    def description(self) -> str:
        return f"Zip(strict={self.strict})"

    def cursor(self) -> Cursor[tuple[T, U]]:
        return cursors.ZipCursor(
            self.left.cursor(),
            self.right.cursor(),
            self.strict,
        )


def _concat_datasets[T](first: Dataset[T], others: tuple[Dataset[T], ...]) -> Dataset[T]:
    if not others:
        return first
    components = (first, *others)
    if all(isinstance(component, IndexedDataset) for component in components):
        return _ConcatIndexed(cast(tuple[IndexedDataset[T], ...], components))
    if all(isinstance(component, RangeDataset) for component in components):
        return _ConcatRange(cast(tuple[RangeDataset[T], ...], components))
    return _ConcatDataset(components)


def _zip_datasets[T, U](
    left: Dataset[T],
    right: Dataset[U],
    strict: bool,
) -> Dataset[tuple[T, U]]:
    left_cardinality = left.cardinality
    right_cardinality = right.cardinality
    if (
        strict
        and isinstance(left_cardinality, Exact)
        and isinstance(right_cardinality, Exact)
        and left_cardinality != right_cardinality
    ):
        raise ValueError("zip inputs have different lengths")
    if isinstance(left, IndexedDataset) and isinstance(right, IndexedDataset):
        return _ZipIndexed(left, right, strict)
    if isinstance(left, RangeDataset) and isinstance(right, RangeDataset):
        return _ZipRange(left, right, strict)
    return _ZipDataset(left, right, strict)


def _repeat_dataset[T](
    parent: Dataset[T],
    count: int | None,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> Dataset[T]:
    count = None if count is None else _nonnegative("count", count)
    seed = to_index(seed)
    if shuffle and not isinstance(parent, IndexedDataset):
        raise TypeError("shuffled repetition requires an IndexedDataset")
    if count is not None:
        if isinstance(parent, IndexedDataset):
            return _RepeatIndexed(parent, count, shuffle, seed)
        if isinstance(parent, RangeDataset):
            return _RepeatRange(parent, count)
    return _RepeatDataset(parent, count, shuffle, seed)


def _repeat_cardinality(parent: Cardinality, count: int | None) -> Cardinality:
    if (
        count == 0
        or isinstance(parent, Exact)
        and parent == 0
        or isinstance(parent, Bounds)
        and parent.upper == 0
    ):
        return Exact(0)
    if count is None:
        if isinstance(parent, Exact) or isinstance(parent, Infinite):
            return Infinite()
        if isinstance(parent, Bounds) and parent.lower > 0:
            return Infinite()
        return Unknown()
    if isinstance(parent, Exact):
        return Exact(parent * count)
    if isinstance(parent, Bounds):
        return Bounds(parent.lower * count, parent.upper * count)
    return parent


def _repeat_description(count: int | None, shuffle: bool, seed: int) -> str:
    if shuffle:
        return f"Repeat(count={count}, shuffle=True, seed={seed})"
    return f"Repeat(count={count}, shuffle=False)"


_UINT64_MASK = (1 << 64) - 1


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _shuffle_position(position: int, length: int, seed: int, epoch: int) -> int:
    if length < 2:
        return position
    bits = (length - 1).bit_length()
    bits += bits % 2
    half_bits = bits // 2
    half_mask = (1 << half_bits) - 1
    key = _mix64((seed & _UINT64_MASK) ^ _mix64(epoch))

    # Cycle walking restricts the Feistel permutation to exactly [0, length).
    while True:
        left, right = position >> half_bits, position & half_mask
        for round_index in range(6):
            round_key = key ^ (round_index * 0x9E3779B97F4A7C15)
            left, right = right, left ^ (_mix64(right ^ round_key) & half_mask)
        position = (left << half_bits) | right
        if position < length:
            return position


def _callable_name(fn: Callable[..., Any], name: str | None) -> str:
    if name is not None:
        if not name:
            raise ValueError("operation name must not be empty")
        return name
    return getattr(fn, "__qualname__", type(fn).__name__)


def _nonnegative(name: str, value: int) -> int:
    value = to_index(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive(name: str, value: int) -> int:
    value = to_index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _shard_bounds(length: int, index: int, count: int) -> tuple[int, int]:
    count = _positive("count", count)
    index = to_index(index)
    if not 0 <= index < count:
        raise ValueError("index must satisfy 0 <= index < count")
    return length * index // count, length * (index + 1) // count


def _explain(dataset: Dataset[Any], prefix: str = "") -> list[str]:
    match dataset.cardinality:
        case Bounds(lower, upper):
            cardinality = f"Bounds({lower}, {upper})"
        case Infinite():
            cardinality = "Infinite"
        case Unknown():
            cardinality = "Unknown"
        case exact:
            cardinality = f"Exact({exact})"
    lines = [f"{prefix}{dataset.description} [{cardinality}]"]
    for parent in dataset.parents:
        lines.extend(_explain(parent, prefix + "  "))
    return lines
