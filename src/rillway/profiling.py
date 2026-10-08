from __future__ import annotations

import sys
from dataclasses import dataclass
from threading import Lock, local
from time import perf_counter_ns
from types import CodeType, FrameType, TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

from .cursor import Cursor

if TYPE_CHECKING:
    from .dataset import Dataset

_TOOL_ID = sys.monitoring.PROFILER_ID
_EVENTS = sys.monitoring.events
_CURSOR_NEXT = Cursor.__next__.__code__


@dataclass(frozen=True)
class ProfileNode:
    description: str
    depth: int
    calls: int
    outputs: int
    failures: int
    wall_seconds: float
    self_seconds: float
    average_seconds: float
    max_seconds: float

    @property
    def throughput(self) -> float:
        return self.outputs / self.wall_seconds if self.wall_seconds else 0.0


@dataclass(frozen=True)
class ProfileReport:
    nodes: tuple[ProfileNode, ...]
    elapsed_seconds: float

    @property
    def outputs(self) -> int:
        return self.nodes[0].outputs if self.nodes else 0

    @property
    def active_seconds(self) -> float:
        return self.nodes[0].wall_seconds if self.nodes else 0.0

    def __str__(self) -> str:
        descriptions = [
            f"{'  ' * node.depth}{node.description}" for node in self.nodes
        ]
        description_width = max((len(value) for value in descriptions), default=0)
        description_width = max(description_width, len("operation"))
        lines = [
            f"{self.outputs:,} outputs in {_format_duration(self.active_seconds)} active "
            f"({_format_duration(self.elapsed_seconds)} elapsed)",
            f"{'operation':{description_width}}  outputs        own  avg/value  max/value",
        ]
        for node, description in zip(self.nodes, descriptions, strict=True):
            failure = f" failures={node.failures}" if node.failures else ""
            lines.append(
                f"{description:{description_width}} {node.outputs:8,d} "
                f"{_format_duration(node.self_seconds):>10} "
                f"{_format_duration(node.average_seconds):>10} "
                f"{_format_duration(node.max_seconds):>10}{failure}"
            )
        return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 0.001:
        return f"{seconds * 1_000:.3f} ms"
    if seconds >= 0.000001:
        return f"{seconds * 1_000_000:.3f} µs"
    return f"{seconds * 1_000_000_000:.0f} ns"


@dataclass
class _Node:
    dataset: Dataset[Any]
    depth: int
    calls: int = 0
    outputs: int = 0
    failures: int = 0
    wall_ns: int = 0
    self_ns: int = 0
    output_ns: int = 0
    max_ns: int = 0


@dataclass
class _Frame:
    node: _Node
    started_ns: int
    child_ns: int = 0


class _Profiler:
    def __init__(self, dataset: Dataset[Any]):
        self._nodes: list[_Node] = []
        self._by_dataset: dict[int, _Node] = {}
        self._codes = {_CURSOR_NEXT}
        self._add_graph(dataset, 0)
        self._lock = Lock()
        self._thread = local()

    def _add_graph(self, dataset: Dataset[Any], depth: int) -> None:
        key = id(dataset)
        if key in self._by_dataset:
            return
        node = _Node(dataset, depth)
        self._nodes.append(node)
        self._by_dataset[key] = node
        read = getattr(type(dataset), "_get", None)
        code = getattr(read, "__code__", None)
        if isinstance(code, CodeType):
            self._codes.add(code)
        for parent in dataset.parents:
            self._add_graph(parent, depth + 1)

    def enable(self) -> None:
        try:
            sys.monitoring.use_tool_id(_TOOL_ID, "rillway")
        except ValueError as error:
            raise RuntimeError("Python's profiler monitoring slot is already in use") from error
        try:
            sys.monitoring.register_callback(_TOOL_ID, _EVENTS.PY_START, self._start)
            sys.monitoring.register_callback(_TOOL_ID, _EVENTS.PY_RETURN, self._return)
            sys.monitoring.register_callback(_TOOL_ID, _EVENTS.PY_UNWIND, self._unwind)
            for code in self._codes:
                sys.monitoring.set_local_events(
                    _TOOL_ID,
                    code,
                    _EVENTS.PY_START | _EVENTS.PY_RETURN,
                )
            sys.monitoring.set_events(_TOOL_ID, _EVENTS.PY_UNWIND)
        except BaseException:
            self.disable()
            raise

    def disable(self) -> None:
        if sys.monitoring.get_tool(_TOOL_ID) != "rillway":
            return
        sys.monitoring.set_events(_TOOL_ID, 0)
        for code in self._codes:
            sys.monitoring.set_local_events(_TOOL_ID, code, 0)
        for event in (_EVENTS.PY_START, _EVENTS.PY_RETURN, _EVENTS.PY_UNWIND):
            sys.monitoring.register_callback(_TOOL_ID, event, None)
        sys.monitoring.free_tool_id(_TOOL_ID)

    def _start(self, code: CodeType, _offset: int) -> None:
        frame = sys._getframe(1)
        node = self._node(code, frame)
        active = self._active_frame()
        measured = None
        if node is not None and (active is None or active.node is not node):
            measured = _Frame(node, perf_counter_ns())
        self._stack().append(measured)

    def _return(self, code: CodeType, _offset: int, _value: object) -> None:
        self._finish(code, output=True, failure=False)

    def _unwind(self, code: CodeType, _offset: int, error: BaseException) -> None:
        if code in self._codes:
            self._finish(
                code,
                output=False,
                failure=not isinstance(error, StopIteration),
            )

    def _finish(self, code: CodeType, *, output: bool, failure: bool) -> None:
        if code not in self._codes:
            return
        stack = self._stack()
        if not stack:
            return
        frame = stack.pop()
        if frame is None:
            return
        elapsed = perf_counter_ns() - frame.started_ns
        parent = self._active_frame()
        if parent is not None:
            parent.child_ns += elapsed
        with self._lock:
            node = frame.node
            node.calls += 1
            node.outputs += output
            node.failures += failure
            node.wall_ns += elapsed
            node.self_ns += max(0, elapsed - frame.child_ns)
            if output:
                node.output_ns += elapsed
                node.max_ns = max(node.max_ns, elapsed)

    def _node(self, code: CodeType, frame: FrameType) -> _Node | None:
        owner = frame.f_locals.get("self")
        dataset = getattr(owner, "_dataset", None) if code is _CURSOR_NEXT else owner
        return self._by_dataset.get(id(dataset))

    def _stack(self) -> list[_Frame | None]:
        stack = getattr(self._thread, "stack", None)
        if stack is None:
            stack = []
            self._thread.stack = stack
        return stack

    def _active_frame(self) -> _Frame | None:
        return next(
            (frame for frame in reversed(self._stack()) if frame is not None),
            None,
        )

    def report(self, elapsed_seconds: float) -> ProfileReport:
        with self._lock:
            nodes = tuple(
                ProfileNode(
                    node.dataset.description,
                    node.depth,
                    node.calls,
                    node.outputs,
                    node.failures,
                    node.wall_ns / 1_000_000_000,
                    node.self_ns / 1_000_000_000,
                    (
                        node.output_ns / node.outputs / 1_000_000_000
                        if node.outputs
                        else 0.0
                    ),
                    node.max_ns / 1_000_000_000,
                )
                for node in self._nodes
            )
        return ProfileReport(nodes, elapsed_seconds)


class Profile:
    def __init__(self, dataset: Dataset[Any]):
        self._dataset = dataset
        self._profiler = _Profiler(dataset)
        self._started_ns: int | None = None
        self._finished_ns: int | None = None

    def report(self) -> ProfileReport:
        if self._started_ns is None:
            raise RuntimeError("profile must be used as a context manager")
        end = perf_counter_ns() if self._finished_ns is None else self._finished_ns
        return self._profiler.report((end - self._started_ns) / 1_000_000_000)

    def _finish(self) -> None:
        if self._finished_ns is None:
            self._finished_ns = perf_counter_ns()

    def __enter__(self) -> Self:
        if self._started_ns is not None:
            raise RuntimeError("profile cannot be reused")
        self._started_ns = perf_counter_ns()
        self._profiler.enable()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._finish()
        self._profiler.disable()
        return False

    def __str__(self) -> str:
        return str(self.report())


def profiling(dataset: Dataset[Any]) -> Profile:
    return Profile(dataset)
