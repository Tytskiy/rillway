from collections import deque
from operator import index as to_index
from threading import Lock

from .cursor import Cursor, State

_READ_AHEAD_SNAPSHOT_INTERVAL = 64


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

    def consumed(self, count: int = 1) -> None:
        with self._lock:
            self._consumed += count
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
