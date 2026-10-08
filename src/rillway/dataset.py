"""Immutable dataset plans.

Transformations build private plan nodes; calling ``cursor()`` turns those nodes
into the stateful executors in ``cursor.py``. Unary nodes share graph metadata
and delegate mode-specific execution to their operation object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from operator import index as to_index
from typing import Any, ClassVar, Literal, Protocol, overload, runtime_checkable

from . import cursor as cursors
from .cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from .cursor import Cursor


class Dataset[T](ABC):
    """Functional callbacks are treated as stateless and deterministic. Put
    stateful processing in a custom Cursor so its state can be checkpointed.
    """

    supports_checkpointing: ClassVar[bool] = True

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
        from ._ops.transforms import _Map, _UnaryDataset

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
        from ._ops.concurrency import _ParallelMapDataset

        return _ParallelMapDataset(
            self,
            fn,
            _callable_name(fn, name),
            workers,
            buffer_size,
            backend,
        )

    def prefetch(self, buffer_size: int) -> Dataset[T]:
        from ._ops.concurrency import _PrefetchDataset

        return _PrefetchDataset(self, _positive("buffer_size", buffer_size))

    def shuffle(self, buffer_size: int, *, seed: int = 42) -> Dataset[T]:
        from ._ops.ordering import _ShuffleDataset

        return _ShuffleDataset(
            self,
            _positive("buffer_size", buffer_size),
            to_index(seed),
        )

    def filter(self, predicate: Callable[[T], bool], *, name: str | None = None) -> Dataset[T]:
        from ._ops.transforms import _Filter, _UnaryDataset

        return _UnaryDataset(self, _Filter(predicate, _callable_name(predicate, name)))

    def flat_map[U](self, fn: Callable[[T], Iterable[U]], *, name: str | None = None) -> Dataset[U]:
        from ._ops.transforms import _FlatMap, _UnaryDataset

        return _UnaryDataset(self, _FlatMap(fn, _callable_name(fn, name)))

    def take(self, count: int) -> Dataset[T]:
        from ._ops.transforms import _Take, _UnaryDataset

        return _UnaryDataset(self, _Take(_nonnegative("count", count)))

    def skip(self, count: int) -> Dataset[T]:
        from ._ops.transforms import _Skip, _UnaryDataset

        return _UnaryDataset(self, _Skip(_nonnegative("count", count)))

    def batch(self, size: int, *, drop_last: bool = False) -> Dataset[tuple[T, ...]]:
        from ._ops.transforms import _Batch, _UnaryDataset

        return _UnaryDataset(self, _Batch(_positive("size", size), drop_last))

    def unbatch[U](self: Dataset[Iterable[U]]) -> Dataset[U]:
        from ._ops.transforms import _UnaryDataset, _Unbatch

        return _UnaryDataset(self, _Unbatch())

    def repeat(self, count: int | None) -> Dataset[T]:
        from ._ops.ordering import _repeat_dataset

        return _repeat_dataset(self, count)

    def concat(self, *others: Dataset[T]) -> Dataset[T]:
        from ._ops.composition import _concat_datasets

        return _concat_datasets(self, others)

    def interleave(self, *others: Dataset[T]) -> Dataset[T]:
        if not others:
            return self
        from ._ops.composition import _InterleaveDataset

        return _InterleaveDataset((self, *others))

    def mix(
        self,
        *others: Dataset[T],
        weights: Iterable[float],
        seed: int = 42,
    ) -> Dataset[T]:
        from ._ops.composition import _mix_datasets

        return _mix_datasets(self, others, weights, seed)

    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]:
        from ._ops.composition import _zip_datasets

        return _zip_datasets(self, other, strict)

    def shard(self, index: int, count: int) -> Dataset[T]:
        from ._ops.ordering import _ShardDataset

        index, count = _validate_shard(index, count)
        return _ShardDataset(self, index, count)

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
        return cursors.RangeCursor(self, cursor, start, stop)

    def cursor(self) -> Cursor[T]:
        return self.open_range(0, len(self))

    def shard(self, index: int, count: int) -> RangeDataset[T]:
        start, stop = _shard_bounds(len(self), index, count)
        return _RangeSlice(self, start, stop)

    def map[U](self, fn: Callable[[T], U], *, name: str | None = None) -> RangeDataset[U]:
        from ._ops.transforms import _Map, _UnaryRange

        return _UnaryRange(self, _Map(fn, _callable_name(fn, name)))

    def take(self, count: int) -> RangeDataset[T]:
        return _RangeSlice(self, 0, min(_nonnegative("count", count), len(self)))

    def skip(self, count: int) -> RangeDataset[T]:
        start = min(_nonnegative("count", count), len(self))
        return _RangeSlice(self, start, len(self))

    def batch(self, size: int, *, drop_last: bool = False) -> RangeDataset[tuple[T, ...]]:
        from ._ops.transforms import _Batch, _UnaryRange

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
        from ._ops.ordering import _repeat_dataset

        return _repeat_dataset(self, count)

    @overload
    def concat(self, *others: RangeDataset[T]) -> RangeDataset[T]: ...

    @overload
    def concat(self, *others: Dataset[T]) -> Dataset[T]: ...

    def concat(self, *others: Dataset[T]) -> Dataset[T]:
        from ._ops.composition import _concat_datasets

        return _concat_datasets(self, others)

    @overload
    def zip[U](self, other: RangeDataset[U], *, strict: bool = False) -> RangeDataset[tuple[T, U]]: ...

    @overload
    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]: ...

    def zip[U](self, other: Dataset[U], *, strict: bool = False) -> Dataset[tuple[T, U]]:
        from ._ops.composition import _zip_datasets

        return _zip_datasets(self, other, strict)


class IndexedDataset[T](RangeDataset[T], ABC):
    """IndexedDataset transformations must be deterministic functions of their input and
    fixed configuration. Stateful processing belongs in a custom Cursor.
    """

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
        from ._ops.transforms import _Map, _UnaryIndexed

        return _UnaryIndexed(self, _Map(fn, _callable_name(fn, name)))

    def take(self, count: int) -> IndexedDataset[T]:
        return self[: _nonnegative("count", count)]

    def skip(self, count: int) -> IndexedDataset[T]:
        return self[_nonnegative("count", count) :]

    def batch(self, size: int, *, drop_last: bool = False) -> IndexedDataset[tuple[T, ...]]:
        from ._ops.transforms import _Batch, _UnaryIndexed

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
        from ._ops.ordering import _repeat_dataset

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
        from ._ops.composition import _concat_datasets

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
        from ._ops.composition import _zip_datasets

        return _zip_datasets(self, other, strict)


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
class _FactoryDataset[T](Dataset[T]):
    supports_checkpointing: ClassVar[bool] = False

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
        return Cursor.from_iterator(iter(self.factory()), dataset=self)


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
    index, count = _validate_shard(index, count)
    return length * index // count, length * (index + 1) // count


def _validate_shard(index: int, count: int) -> tuple[int, int]:
    count = _positive("count", count)
    index = to_index(index)
    if not 0 <= index < count:
        raise ValueError("index must satisfy 0 <= index < count")
    return index, count


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
