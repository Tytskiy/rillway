from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack
from operator import index as to_index
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self
from weakref import finalize

if TYPE_CHECKING:
    from .dataset import Dataset, IndexedDataset

type State = dict[str, Any]
type _CheckpointKey = str | int | bool | tuple[_CheckpointKey, ...]

_STATE_VERSION = 2


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
            "state": self._snapshot(),
        }

    def load_state_dict(self, state: State) -> None:
        if self._started:
            raise RuntimeError("cannot restore a cursor after iteration has started")
        if self._closed:
            raise RuntimeError("cannot restore a closed cursor")
        self._require_checkpointable()
        try:
            self._restore(self._checkpoint_payload(state))
        except BaseException as error:
            self._exit(type(error), error, error.__traceback__)
            raise

    def _snapshot(self) -> State:
        raise TypeError(f"{type(self).__name__} is not checkpointable")

    def _restore(self, state: State) -> None:
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
            cursor_type._snapshot is Cursor._snapshot
            or cursor_type._restore is Cursor._restore
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

    def _restore(self, state: State) -> None:
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

    def _snapshot(self) -> State:
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

    def _restore(self, state: State) -> None:
        self._load_local_state(state)
        self._parent._restore(state["parent"])

    def _snapshot(self) -> State:
        return {"parent": self._parent._snapshot(), **self._local_state()}


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
