from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from operator import index as to_index
from threading import Lock, Thread, current_thread
from typing import Any

from ..cardinality import Cardinality
from ..cursor import Cursor, State
from ..dataset import Dataset
from ..utils.queue import _QueueClosed, _ThreadingQueue

_READ_AHEAD_SNAPSHOT_INTERVAL = 64


@dataclass(frozen=True, slots=True)
class _ParallelMapDataset[T, U](Dataset[U]):
    parent: Dataset[T]
    fn: Callable[[T], U]
    name: str
    workers: int
    buffer_size: int
    backend: str

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return (self.parent,)

    @property
    def cardinality(self) -> Cardinality:
        return self.parent.cardinality

    @property
    def description(self) -> str:
        return (
            f"ParallelMap(name={self.name!r}, workers={self.workers}, "
            f"buffer_size={self.buffer_size}, backend={self.backend!r})"
        )

    def cursor(self) -> Cursor[U]:
        return _ParallelMapCursor(
            self,
            self.parent.cursor(),
            self.fn,
            self.workers,
            self.buffer_size,
            self.backend,
        )


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
        return _PrefetchCursor(self, self.parent.cursor(), self.buffer_size)


class _ReadAheadCheckpoint[T]:
    def __init__(self, parent: Cursor[T]):
        self._parent = parent
        self._lock = Lock()
        self._anchor = parent._snapshot()
        self._anchor_position = 0
        self._produced = 0
        self._consumed = 0
        self._snapshots: deque[tuple[int, State]] = deque()

    def produced(self) -> None:
        self._produced += 1
        if self._produced % _READ_AHEAD_SNAPSHOT_INTERVAL:
            return
        snapshot = self._parent._snapshot()
        with self._lock:
            self._snapshots.append((self._produced, snapshot))

    def consumed(self) -> None:
        with self._lock:
            self._consumed += 1
            while self._snapshots and self._snapshots[0][0] <= self._consumed:
                self._anchor_position, self._anchor = self._snapshots.popleft()

    def snapshot(self) -> State:
        with self._lock:
            return {
                "parent": self._anchor,
                "replay": self._consumed - self._anchor_position,
            }

    def restore(self, state: State) -> None:
        try:
            parent_state = state["parent"]
            replay = to_index(state["replay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid read-ahead checkpoint") from error
        if not 0 <= replay < _READ_AHEAD_SNAPSHOT_INTERVAL:
            raise ValueError("read-ahead checkpoint has an invalid replay count")

        self._parent._restore(parent_state)
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


class _ParallelMapCursor[T, U](Cursor[U]):
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

    def _snapshot(self) -> State:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return self._checkpoint.snapshot()

    def _restore(self, state: State) -> None:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        self._checkpoint.restore(state)


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


class _PrefetchCursor[T](Cursor[T]):
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

    def _snapshot(self) -> State:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        return self._checkpoint.snapshot()

    def _restore(self, state: State) -> None:
        if self._checkpoint is None:
            raise TypeError(f"{type(self).__name__} is not checkpointable")
        self._checkpoint.restore(state)
