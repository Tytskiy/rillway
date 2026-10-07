from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from multiprocessing import get_context
from operator import index as to_index
from random import Random
from threading import Lock, Thread, current_thread
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self
from weakref import finalize

from ._queue import _QueueClosed, _ThreadingQueue

if TYPE_CHECKING:
    from .dataset import Dataset, IndexedDataset, _RepeatDataset

type State = dict[str, Any]
type _CheckpointKey = str | int | bool | tuple[_CheckpointKey, ...]

_STATE_VERSION = 1
_READ_AHEAD_SNAPSHOT_INTERVAL = 64


class Cursor[T](Iterator[T], ABC):
    def __init__(self, dataset: Dataset[Any] | None = None) -> None:
        self._dataset = dataset
        self._closed = False
        self._started = False
        self._scope = ExitStack()
        self._finalizer = finalize(self, self._scope.close)

    @classmethod
    def from_iterator[U](
        cls,
        iterator: Iterator[U],
        *,
        dataset: Dataset[Any] | None = None,
    ) -> Cursor[U]:
        return _IteratorCursor(iterator, dataset)

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        self._started = True
        try:
            return self._next()
        except StopIteration:
            self.close()
            raise
        except BaseException as error:
            self._exit(type(error), error, error.__traceback__)
            raise

    @abstractmethod
    def _next(self) -> T: ...

    def enter_context[U](self, resource: AbstractContextManager[U]) -> U:
        return self._scope.enter_context(resource)

    def callback(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._scope.callback(callback, *args, **kwargs)

    def state_dict(self) -> State:
        self._require_checkpointable()
        return {
            "version": _STATE_VERSION,
            "cursor": self._checkpoint_key(),
            "state": self._state_dict(),
        }

    def load_state_dict(self, state: State) -> None:
        if self._started:
            raise RuntimeError("cannot restore a cursor after iteration has started")
        if self._closed:
            raise RuntimeError("cannot restore a closed cursor")
        self._require_checkpointable()
        try:
            self._load_state_dict(self._checkpoint_payload(state))
        except BaseException as error:
            self._exit(type(error), error, error.__traceback__)
            raise

    def _state_dict(self) -> State:
        raise TypeError(f"{type(self).__name__} is not checkpointable")

    def _load_state_dict(self, state: State) -> None:
        raise TypeError(f"{type(self).__name__} is not checkpointable")

    def _checkpoint_key(self) -> _CheckpointKey:
        if self._dataset is not None:
            return "dataset", self._dataset.explain()
        cursor_type = type(self)
        return f"{cursor_type.__module__}.{cursor_type.__qualname__}"

    @property
    def checkpointable(self) -> bool:
        if self._dataset is not None:
            return self._dataset.checkpointable
        cursor_type = type(self)
        return not (
            cursor_type._state_dict is Cursor._state_dict
            or cursor_type._load_state_dict is Cursor._load_state_dict
        )

    def _require_checkpointable(self) -> None:
        if not self.checkpointable:
            raise TypeError(f"{type(self).__name__} is not checkpointable")

    def _checkpoint_payload(self, checkpoint: State) -> State:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"version", "cursor", "state"}:
            raise ValueError("invalid checkpoint")
        version = checkpoint["version"]
        if type(version) is not int or version != _STATE_VERSION:
            raise ValueError(f"unsupported checkpoint version: {version!r}")
        if checkpoint["cursor"] != self._checkpoint_key():
            raise ValueError("checkpoint does not match this cursor")
        state = checkpoint["state"]
        if not isinstance(state, dict):
            raise ValueError("invalid checkpoint state")
        return state

    def close(self) -> None:
        self._exit(None, None, None)

    def _exit(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._finalizer.detach()
        return bool(self._scope.__exit__(exc_type, exc, traceback))

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("cannot enter a closed cursor")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self._exit(exc_type, exc, traceback)


class _IteratorCursor[T](Cursor[T]):
    def __init__(self, iterator: Iterator[T], dataset: Dataset[Any] | None = None):
        super().__init__(dataset)
        self._iterator = iterator
        close = getattr(iterator, "close", None)
        if close is not None:
            self.callback(close)

    def _next(self) -> T:
        return next(self._iterator)


class IndexedCursor[T](Cursor[T]):
    def __init__(
        self,
        dataset: IndexedDataset[T],
        start: int = 0,
        stop: int | None = None,
    ):
        super().__init__(dataset)
        self._indexed_dataset = dataset
        self._start = start
        self._stop = len(dataset) if stop is None else stop
        self._position = start

    def _checkpoint_key(self) -> _CheckpointKey:
        return "indexed", self._indexed_dataset.explain(), self._start, self._stop

    def _load_state_dict(self, state: State) -> None:
        position = to_index(state["position"])
        if not self._start <= position <= self._stop:
            raise ValueError("cursor position is outside the requested range")
        self._position = position

    def _next(self) -> T:
        if self._position == self._stop:
            raise StopIteration
        value = self._indexed_dataset._get(self._position)
        self._position += 1
        return value

    def _state_dict(self) -> State:
        return {"position": self._position}


class _ParentCursor[T, U](Cursor[U]):
    def __init__(self, dataset: Dataset[Any] | None, parent: Cursor[T]):
        super().__init__(dataset)
        self._parent = self.enter_context(parent)

    @property
    def checkpointable(self) -> bool:
        return super().checkpointable and self._parent.checkpointable

    def _local_state(self) -> State:
        return {}

    def _load_local_state(self, state: State) -> None:
        pass

    def _load_state_dict(self, state: State) -> None:
        self._load_local_state(state)
        self._parent.load_state_dict(state["parent"])

    def _state_dict(self) -> State:
        return {"parent": self._parent.state_dict(), **self._local_state()}


class RangeCursor[T](_ParentCursor[T, T]):
    def __init__(
        self,
        dataset: Dataset[T],
        parent: Cursor[T],
        start: int,
        stop: int,
    ):
        super().__init__(dataset, parent)
        self._start = start
        self._stop = stop

    def _next(self) -> T:
        return next(self._parent)

    def _local_state(self) -> State:
        return {"range_start": self._start, "range_stop": self._stop}

    def _load_local_state(self, state: State) -> None:
        try:
            start = to_index(state["range_start"])
            stop = to_index(state["range_stop"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint does not match the requested range") from error
        if (start, stop) != (self._start, self._stop):
            raise ValueError("checkpoint does not match the requested range")


class TransformCursor[T, U](_ParentCursor[T, U]):
    def __init__(
        self,
        dataset: Dataset[U],
        parent: Cursor[T],
        iterator: Iterator[U],
    ):
        super().__init__(dataset, parent)
        self._iterator = iterator

    def _next(self) -> U:
        return next(self._iterator)


class _ReadAheadCheckpoint[T]:
    def __init__(self, parent: Cursor[T]):
        self._parent = parent
        self._lock = Lock()
        self._anchor = parent.state_dict()
        self._anchor_position = 0
        self._produced = 0
        self._consumed = 0
        self._snapshots: deque[tuple[int, State]] = deque()

    def produced(self) -> None:
        self._produced += 1
        if self._produced % _READ_AHEAD_SNAPSHOT_INTERVAL:
            return
        snapshot = self._parent.state_dict()
        with self._lock:
            self._snapshots.append((self._produced, snapshot))

    def consumed(self) -> None:
        with self._lock:
            self._consumed += 1
            while self._snapshots and self._snapshots[0][0] <= self._consumed:
                self._anchor_position, self._anchor = self._snapshots.popleft()

    def state_dict(self) -> State:
        with self._lock:
            return {
                "parent": self._anchor,
                "replay": self._consumed - self._anchor_position,
            }

    def load_state_dict(self, state: State) -> None:
        try:
            parent_state = state["parent"]
            replay = to_index(state["replay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid read-ahead checkpoint") from error
        if not 0 <= replay < _READ_AHEAD_SNAPSHOT_INTERVAL:
            raise ValueError("read-ahead checkpoint has an invalid replay count")

        self._parent.load_state_dict(parent_state)
        for _ in range(replay):
            try:
                next(self._parent)
            except StopIteration as error:
                raise ValueError("read-ahead checkpoint exceeds the parent") from error

        with self._lock:
            self._anchor = parent_state
            self._anchor_position = 0
            self._produced = replay
            self._consumed = replay
            self._snapshots.clear()


class ParallelMapCursor[T, U](Cursor[U]):
    def __init__(
        self,
        dataset: Dataset[U],
        parent: Cursor[T],
        fn: Callable[[T], U],
        workers: int,
        buffer_size: int,
        backend: str,
    ):
        super().__init__(dataset)
        self._parent = self.enter_context(parent)
        self._fn = fn
        self._capacity = workers + buffer_size
        self._pending: deque[Future[U]] = deque()
        self._parent_done = False
        self._checkpoint = _ReadAheadCheckpoint(parent) if parent.checkpointable else None
        self._executor: Executor
        if backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=workers)
        elif backend == "process":
            self._executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
            )
        else:
            raise ValueError(f"unsupported parallel backend: {backend!r}")
        self.callback(self._executor.shutdown, wait=True, cancel_futures=True)

    @property
    def checkpointable(self) -> bool:
        return self._checkpoint is not None

    def _fill(self) -> None:
        while not self._parent_done and len(self._pending) < self._capacity:
            try:
                value = next(self._parent)
            except StopIteration:
                self._parent_done = True
            except Exception as error:
                self._parent_done = True
                failure: Future[U] = Future()
                failure.set_exception(error)
                self._pending.append(failure)
            else:
                pending = self._executor.submit(self._fn, value)
                if self._checkpoint is not None:
                    self._checkpoint.produced()
                self._pending.append(pending)

    def _next(self) -> U:
        self._fill()
        if not self._pending:
            raise StopIteration
        pending = self._pending.popleft()
        value = pending.result()
        self._fill()
        if self._checkpoint is not None:
            self._checkpoint.consumed()
        return value

    def _state_dict(self) -> State:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return self._checkpoint.state_dict()

    def _load_state_dict(self, state: State) -> None:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        self._checkpoint.load_state_dict(state)


@dataclass(slots=True)
class _PrefetchFailure:
    error: BaseException


class _PrefetchDone:
    pass


@dataclass(slots=True)
class _PrefetchedValue[T]:
    value: T


type _PrefetchItem[T] = _PrefetchedValue[T] | _PrefetchFailure | _PrefetchDone


class _PrefetchState[T]:
    def __init__(
        self,
        parent: Cursor[T],
        buffer_size: int,
        checkpoint: _ReadAheadCheckpoint[T] | None,
    ):
        self._parent = parent
        self._checkpoint = checkpoint
        self._queue = _ThreadingQueue[_PrefetchItem[T]](buffer_size)
        self._thread: Thread | None = None

    def get(self) -> T:
        if self._thread is None:
            self._thread = Thread(
                target=self._produce,
                name="rillway-prefetch",
                daemon=True,
            )
            self._thread.start()
        try:
            item = self._queue.get()
        except _QueueClosed:
            raise StopIteration from None
        if isinstance(item, _PrefetchedValue):
            if self._checkpoint is not None:
                self._checkpoint.consumed()
            return item.value
        if isinstance(item, _PrefetchDone):
            raise StopIteration
        if isinstance(item, _PrefetchFailure):
            raise item.error
        raise AssertionError("invalid prefetch item")

    def close(self) -> None:
        self._queue.close()
        self._parent.close()
        if self._thread is not None and self._thread is not current_thread():
            self._thread.join()

    def _produce(self) -> None:
        try:
            while True:
                try:
                    value = next(self._parent)
                    if self._checkpoint is not None:
                        self._checkpoint.produced()
                    item: _PrefetchItem[T] = _PrefetchedValue(value)
                except StopIteration:
                    item = _PrefetchDone()
                except BaseException as error:
                    item = _PrefetchFailure(error)
                try:
                    self._queue.put(item)
                except _QueueClosed:
                    return
                if isinstance(item, _PrefetchDone | _PrefetchFailure):
                    return
        finally:
            self._parent.close()


class PrefetchCursor[T](Cursor[T]):
    def __init__(self, dataset: Dataset[T], parent: Cursor[T], buffer_size: int):
        super().__init__(dataset)
        self._checkpoint = _ReadAheadCheckpoint(parent) if parent.checkpointable else None
        self._state = _PrefetchState(parent, buffer_size, self._checkpoint)
        self.callback(self._state.close)

    @property
    def checkpointable(self) -> bool:
        return self._checkpoint is not None

    def _next(self) -> T:
        return self._state.get()

    def _state_dict(self) -> State:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return self._checkpoint.state_dict()

    def _load_state_dict(self, state: State) -> None:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        self._checkpoint.load_state_dict(state)


class _CloseSlot:
    def __init__(self) -> None:
        self._close: Callable[[], Any] | None = None

    def replace(self, resource: object) -> None:
        self.close()
        close = getattr(resource, "close", None)
        self._close = close if callable(close) else None

    def close(self) -> None:
        if self._close is not None:
            close, self._close = self._close, None
            close()


class FlatMapCursor[T, U](_ParentCursor[T, U]):
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
                    raise ValueError(
                        "flat_map checkpoint no longer matches output"
                    ) from error
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


class TakeCursor[T](_ParentCursor[T, T]):
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


class SkipCursor[T](_ParentCursor[T, T]):
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


class BatchCursor[T](_ParentCursor[T, tuple[T, ...]]):
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


class ShuffleCursor[T](_ParentCursor[T, T]):
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
        self._anchor = parent.state_dict() if parent.checkpointable else None

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

    def _state_dict(self) -> State:
        if self._anchor is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return {"parent": self._anchor, "replay": self._position}

    def _load_state_dict(self, state: State) -> None:
        try:
            anchor = state["parent"]
            replay = to_index(state["replay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid shuffle checkpoint") from error
        if replay < 0:
            raise ValueError("shuffle checkpoint has an invalid replay count")

        self._parent.load_state_dict(anchor)
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


class ShardCursor[T](_ParentCursor[T, T]):
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


class RepeatCursor[T](Cursor[T]):
    def __init__(self, dataset: _RepeatDataset[T]):
        super().__init__(dataset)
        self._repeat_dataset = dataset
        self._epoch = 0
        self._active: Cursor[T] | None = None
        self._yielded = False
        self._active_resource = _CloseSlot()
        self.callback(self._active_resource.close)

    def _load_state_dict(self, state: State) -> None:
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
            self._active.load_state_dict(active_state)

    def _next(self) -> T:
        while (
            self._repeat_dataset.count is None
            or self._epoch < self._repeat_dataset.count
        ):
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

    def _state_dict(self) -> State:
        return {
            "epoch": self._epoch,
            "yielded": self._yielded,
            "active": None if self._active is None else self._active.state_dict(),
        }


class ConcatCursor[T](Cursor[T]):
    def __init__(self, dataset: Dataset[T], components: tuple[Dataset[T], ...]):
        super().__init__(dataset)
        self._components = components
        self._component = 0
        self._active: Cursor[T] | None = None
        self._active_resource = _CloseSlot()
        self.callback(self._active_resource.close)

    @property
    def checkpointable(self) -> bool:
        return all(component.checkpointable for component in self._components)

    def _load_state_dict(self, state: State) -> None:
        component = to_index(state["component"])
        active_state = state["active"]
        if not 0 <= component <= len(self._components):
            raise ValueError("concat checkpoint has an invalid component")
        if component == len(self._components) and active_state is not None:
            raise ValueError("concat checkpoint has an invalid active component")
        self._component = component
        if active_state is not None:
            self._active = self._components[self._component].cursor()
            self._active_resource.replace(self._active)
            self._active.load_state_dict(active_state)

    def _next(self) -> T:
        while self._component < len(self._components):
            if self._active is None:
                self._active = self._components[self._component].cursor()
                self._active_resource.replace(self._active)
            try:
                return next(self._active)
            except StopIteration:
                self._active_resource.close()
                self._active = None
                self._component += 1
        raise StopIteration

    def _state_dict(self) -> State:
        return {
            "component": self._component,
            "active": None if self._active is None else self._active.state_dict(),
        }


class InterleaveCursor[T](Cursor[T]):
    def __init__(self, dataset: Dataset[T], components: tuple[Dataset[T], ...]):
        super().__init__(dataset)
        self._cursors = tuple(
            self.enter_context(component.cursor()) for component in components
        )
        self._active = [True] * len(self._cursors)
        self._remaining = len(self._cursors)
        self._component = 0

    def _next(self) -> T:
        while self._remaining:
            component = self._component
            self._component = (component + 1) % len(self._cursors)
            if not self._active[component]:
                continue
            try:
                return next(self._cursors[component])
            except StopIteration:
                self._active[component] = False
                self._remaining -= 1
        raise StopIteration

    def _state_dict(self) -> State:
        return {
            "component": self._component,
            "active": list(self._active),
            "parents": [cursor.state_dict() for cursor in self._cursors],
        }

    def _load_state_dict(self, state: State) -> None:
        component = to_index(state["component"])
        active = list(state["active"])
        parents = list(state["parents"])
        if (
            not 0 <= component < len(self._cursors)
            or len(active) != len(self._cursors)
            or any(type(value) is not bool for value in active)
            or len(parents) != len(self._cursors)
        ):
            raise ValueError("interleave checkpoint has invalid state")
        for cursor, parent_state in zip(self._cursors, parents, strict=True):
            cursor.load_state_dict(parent_state)
        self._component = component
        self._active = active
        self._remaining = sum(active)


class ZipCursor[T, U](Cursor[tuple[T, U]]):
    def __init__(
        self,
        dataset: Dataset[tuple[T, U]],
        left: Cursor[T],
        right: Cursor[U],
        strict: bool,
    ):
        super().__init__(dataset)
        self._left = self.enter_context(left)
        self._right = self.enter_context(right)
        self._iterator = zip(self._left, self._right, strict=strict)

    @property
    def checkpointable(self) -> bool:
        return self._left.checkpointable and self._right.checkpointable

    def _next(self) -> tuple[T, U]:
        try:
            return next(self._iterator)
        except ValueError:
            self.close()
            raise

    def _state_dict(self) -> State:
        return {
            "left": self._left.state_dict(),
            "right": self._right.state_dict(),
        }

    def _load_state_dict(self, state: State) -> None:
        self._left.load_state_dict(state["left"])
        self._right.load_state_dict(state["right"])
