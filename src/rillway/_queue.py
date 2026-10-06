from collections import deque
from threading import Condition


class _QueueClosed(Exception):
    pass


class _ThreadingQueue[T]:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[T] = deque()
        self._capacity = capacity
        self._condition = Condition()
        self._closed = False

    def put(self, item: T) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or len(self._items) < self._capacity
            )
            if self._closed:
                raise _QueueClosed
            self._items.append(item)
            self._condition.notify()

    def get(self) -> T:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or bool(self._items))
            if self._closed:
                raise _QueueClosed
            item = self._items.popleft()
            self._condition.notify()
            return item

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._items.clear()
            self._condition.notify_all()
