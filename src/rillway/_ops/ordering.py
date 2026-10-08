from dataclasses import dataclass
from operator import index as to_index
from random import Random
from typing import Any

from ..cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from ..cursor import Cursor, State, _CloseSlot, _ParentCursor
from ..dataset import Dataset, IndexedDataset, RangeDataset, _nonnegative, _RangeSlice


@dataclass(frozen=True, slots=True)
class _RepeatRange[T](RangeDataset[T]):
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
        from .composition import _ConcatCursor

        length = len(self.parent)
        if length == 0:
            return _ConcatCursor(self, ())
        components = []
        while start < stop:
            offset = start % length
            component_stop = min(length, offset + stop - start)
            components.append(_RangeSlice(self.parent, offset, component_stop))
            start += component_stop - offset
        return _ConcatCursor(self, tuple(components))


@dataclass(frozen=True, slots=True)
class _RepeatIndexed[T](IndexedDataset[T]):
    parent: IndexedDataset[T]
    count: int
    shuffled: bool
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Exact:
        return Exact(len(self.parent) * self.count)

    @property
    def description(self) -> str:
        return _repeat_description(self.count, self.shuffled, self.seed)

    def _get(self, position: int) -> T:
        length = len(self.parent)
        epoch, position = divmod(position, length)
        if self.shuffled:
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
class _ShuffleDataset[T](Dataset[T]):
    parent: Dataset[T]
    buffer_size: int
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        return self.parent.cardinality

    @property
    def description(self) -> str:
        return f"Shuffle(buffer_size={self.buffer_size}, seed={self.seed})"

    def cursor(self) -> Cursor[T]:
        return _ShuffleCursor(
            self,
            self.parent.cursor(),
            self.buffer_size,
            self.seed,
        )


@dataclass(frozen=True, slots=True)
class _ShardDataset[T](Dataset[T]):
    parent: Dataset[T]
    index: int
    count: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        parent = self.parent.cardinality
        if isinstance(parent, Exact):
            return Exact(_strided_shard_size(parent, self.index, self.count))
        if isinstance(parent, Bounds):
            return Bounds(
                _strided_shard_size(parent.lower, self.index, self.count),
                _strided_shard_size(parent.upper, self.index, self.count),
            )
        return parent

    @property
    def description(self) -> str:
        return f"Shard(index={self.index}, count={self.count})"

    def cursor(self) -> Cursor[T]:
        return _ShardCursor(
            self,
            self.parent.cursor(),
            self.index,
            self.count,
        )


@dataclass(frozen=True, slots=True)
class _RepeatDataset[T](Dataset[T]):
    parent: Dataset[T]
    count: int | None
    shuffled: bool
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        return _repeat_cardinality(self.parent.cardinality, self.count)

    @property
    def description(self) -> str:
        return _repeat_description(self.count, self.shuffled, self.seed)

    def cursor(self) -> Cursor[T]:
        return _RepeatCursor(self)

    def _open_epoch(self, epoch: int) -> Cursor[T]:
        if self.shuffled:
            assert isinstance(self.parent, IndexedDataset)
            return _ShuffledEpochIndexed(self.parent, self.seed, epoch).cursor()
        return self.parent.cursor()


class _ShuffleCursor[T](_ParentCursor[T, T]):
    def __init__(
        self,
        dataset: Dataset[T],
        parent: Cursor[T],
        buffer_size: int,
        seed: int,
    ):
        super().__init__(dataset, parent)
        self._buffer_size = buffer_size
        self._seed = seed
        self._random = Random(seed)
        self._buffer: list[T] = []
        self._parent_done = False
        self._position = 0
        self._anchor = parent._snapshot() if parent.checkpointable else None

    def _next(self) -> T:
        while not self._parent_done and len(self._buffer) < self._buffer_size:
            try:
                self._buffer.append(next(self._parent))
            except StopIteration:
                self._parent_done = True
        if not self._buffer:
            raise StopIteration
        position = self._random.randrange(len(self._buffer))
        value = self._buffer[position]
        self._buffer[position] = self._buffer[-1]
        self._buffer.pop()
        self._position += 1
        return value

    def _snapshot(self) -> State:
        if self._anchor is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return {"parent": self._anchor, "replay": self._position}

    def _restore(self, state: State) -> None:
        try:
            anchor = state["parent"]
            replay = to_index(state["replay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid shuffle checkpoint") from error
        if replay < 0:
            raise ValueError("shuffle checkpoint has an invalid replay count")

        self._parent._restore(anchor)
        self._anchor = anchor
        self._random = Random(self._seed)
        self._buffer.clear()
        self._parent_done = False
        self._position = 0
        for _ in range(replay):
            try:
                self._next()
            except StopIteration as error:
                raise ValueError("shuffle checkpoint exceeds the parent") from error


class _ShardCursor[T](_ParentCursor[T, T]):
    def __init__(
        self,
        dataset: Dataset[T],
        parent: Cursor[T],
        index: int,
        count: int,
    ):
        super().__init__(dataset, parent)
        self._index = index
        self._count = count
        self._position = 0

    def _next(self) -> T:
        while True:
            value = next(self._parent)
            selected = self._position % self._count == self._index
            self._position += 1
            if selected:
                return value

    def _local_state(self) -> State:
        return {"position": self._position}

    def _load_local_state(self, state: State) -> None:
        position = to_index(state["position"])
        if position < 0:
            raise ValueError("shard checkpoint has an invalid position")
        self._position = position


class _RepeatCursor[T](Cursor[T]):
    def __init__(self, dataset: _RepeatDataset[T]):
        super().__init__(dataset)
        self._repeat_dataset = dataset
        self._epoch = 0
        self._active: Cursor[T] | None = None
        self._yielded = False
        self._active_resource = _CloseSlot()
        self.callback(self._active_resource.close)

    def _restore(self, state: State) -> None:
        epoch = to_index(state["epoch"])
        if (
            epoch < 0
            or self._repeat_dataset.count is not None
            and epoch > self._repeat_dataset.count
        ):
            raise ValueError("repeat checkpoint has an invalid epoch")
        yielded = state["yielded"]
        if type(yielded) is not bool:
            raise ValueError("repeat checkpoint has an invalid yielded flag")
        active_state = state["active"]
        if (active_state is None) == yielded:
            raise ValueError("repeat checkpoint has inconsistent active state")
        if (
            active_state is not None
            and self._repeat_dataset.count is not None
            and epoch == self._repeat_dataset.count
        ):
            raise ValueError("repeat checkpoint has an invalid active epoch")
        self._epoch = epoch
        self._yielded = yielded
        if active_state is not None:
            self._active = self._repeat_dataset._open_epoch(epoch)
            self._active_resource.replace(self._active)
            self._active._restore(active_state)

    def _next(self) -> T:
        while self._repeat_dataset.count is None or self._epoch < self._repeat_dataset.count:
            if self._active is None:
                self._active = self._repeat_dataset._open_epoch(self._epoch)
                self._active_resource.replace(self._active)
                self._yielded = False
            try:
                value = next(self._active)
            except StopIteration:
                self._active_resource.close()
                self._active = None
                if not self._yielded:
                    raise
                self._epoch += 1
            else:
                self._yielded = True
                return value
        raise StopIteration

    def _snapshot(self) -> State:
        return {
            "epoch": self._epoch,
            "yielded": self._yielded,
            "active": None if self._active is None else self._active._snapshot(),
        }


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
        if isinstance(parent, Exact | Infinite):
            return Infinite()
        if isinstance(parent, Bounds) and parent.lower > 0:
            return Infinite()
        return Unknown()
    if isinstance(parent, Exact):
        return Exact(parent * count)
    if isinstance(parent, Bounds):
        return Bounds(parent.lower * count, parent.upper * count)
    return parent


def _strided_shard_size(size: int, index: int, count: int) -> int:
    return max(0, (size + count - index - 1) // count)


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
