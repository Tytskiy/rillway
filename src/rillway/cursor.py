from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack
from operator import index as to_index
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self
from weakref import finalize

if TYPE_CHECKING:
    from .dataset import Dataset, IndexedDataset

type State = dict[str, Any]
type _CheckpointKey = str | int | bool | tuple[_CheckpointKey, ...]

_STATE_VERSION = 1


class Cursor[T](Iterator[T], ABC):
    def __init__(self) -> None:
        self._closed = False
        self._started = False
        self._scope = ExitStack()
        self._finalizer = finalize(self, self._scope.close)

    @classmethod
    def from_iterator[U](cls, iterator: Iterator[U]) -> Cursor[U]:
        return _IteratorCursor(iterator)

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
        cursor_type = type(self)
        return f"{cursor_type.__module__}.{cursor_type.__qualname__}"

    def _require_checkpointable(self) -> None:
        cursor_type = type(self)
        if (
            cursor_type._state_dict is Cursor._state_dict
            or cursor_type._load_state_dict is Cursor._load_state_dict
        ):
            raise TypeError(f"{cursor_type.__name__} is not checkpointable")

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
    def __init__(self, iterator: Iterator[T]):
        super().__init__()
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
        super().__init__()
        self._dataset = dataset
        self._start = start
        self._stop = len(dataset) if stop is None else stop
        self._position = start

    def _checkpoint_key(self) -> _CheckpointKey:
        return "indexed", self._dataset.explain(), self._start, self._stop

    def _load_state_dict(self, state: State) -> None:
        position = to_index(state["position"])
        if not self._start <= position <= self._stop:
            raise ValueError("cursor position is outside the requested range")
        self._position = position

    def _next(self) -> T:
        if self._position == self._stop:
            raise StopIteration
        value = self._dataset._get(self._position)
        self._position += 1
        return value

    def _state_dict(self) -> State:
        return {"position": self._position}


class _ParentCursor[T, U](Cursor[U]):
    def __init__(self, parent: Cursor[T]):
        super().__init__()
        self._parent = self.enter_context(parent)

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
    def __init__(self, parent: Cursor[T], start: int, stop: int):
        super().__init__(parent)
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
    def __init__(self, parent: Cursor[T], iterator: Iterator[U], operation: _CheckpointKey):
        super().__init__(parent)
        self._iterator = iterator
        self._operation = operation

    def _checkpoint_key(self) -> _CheckpointKey:
        return "transform", self._operation

    def _next(self) -> U:
        return next(self._iterator)


class ParallelMapCursor[T, U](Cursor[U]):
    def __init__(
        self,
        parent: Cursor[T],
        fn: Callable[[T], U],
        workers: int,
        buffer_size: int,
    ):
        super().__init__()
        self._parent = self.enter_context(parent)
        self._fn = fn
        self._capacity = workers + buffer_size
        self._pending: deque[Future[U]] = deque()
        self._parent_done = False
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self.callback(self._executor.shutdown, wait=True, cancel_futures=True)

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
                self._pending.append(self._executor.submit(self._fn, value))

    def _next(self) -> U:
        self._fill()
        if not self._pending:
            raise StopIteration
        pending = self._pending.popleft()
        value = pending.result()
        self._fill()
        return value


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
        parent: Cursor[T],
        fn: Callable[[T], Iterable[U]],
        operation: _CheckpointKey,
    ):
        super().__init__(parent)
        self._fn = fn
        self._operation = operation
        self._current: T | None = None
        self._has_current = False
        self._child: Iterator[U] | None = None
        self._child_offset = 0
        self._child_resource = _CloseSlot()
        self.callback(self._child_resource.close)

    def _checkpoint_key(self) -> _CheckpointKey:
        return "flat_map", self._operation

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
    def __init__(self, parent: Cursor[T], remaining: int):
        super().__init__(parent)
        self._limit = remaining
        self._remaining = remaining

    def _checkpoint_key(self) -> _CheckpointKey:
        return "take", self._limit

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
    def __init__(self, parent: Cursor[T], remaining: int):
        super().__init__(parent)
        self._limit = remaining
        self._remaining = remaining

    def _checkpoint_key(self) -> _CheckpointKey:
        return "skip", self._limit

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
        parent: Cursor[T],
        size: int,
        drop_last: bool,
    ):
        super().__init__(parent)
        self._size = size
        self._drop_last = drop_last
        self._buffer: list[T] = []

    def _checkpoint_key(self) -> _CheckpointKey:
        return "batch", self._size, self._drop_last

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


class ConcatCursor[T](Cursor[T]):
    def __init__(self, components: tuple[Dataset[T], ...]):
        super().__init__()
        self._components = components
        self._component = 0
        self._active: Cursor[T] | None = None
        self._active_resource = _CloseSlot()
        self.callback(self._active_resource.close)

    def _checkpoint_key(self) -> _CheckpointKey:
        return "concat", tuple(component.explain() for component in self._components)

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


class ZipCursor[T, U](Cursor[tuple[T, U]]):
    def __init__(self, left: Cursor[T], right: Cursor[U], strict: bool):
        super().__init__()
        self._left = self.enter_context(left)
        self._right = self.enter_context(right)
        self._strict = strict
        self._iterator = zip(self._left, self._right, strict=strict)

    def _checkpoint_key(self) -> _CheckpointKey:
        return "zip", self._strict

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
