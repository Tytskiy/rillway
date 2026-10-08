from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import accumulate
from math import isfinite
from operator import index as to_index
from typing import Any, cast

from ..cardinality import Bounds, Cardinality, Exact, Infinite, Unknown
from ..cursor import Cursor, State, _CloseSlot
from ..dataset import Dataset, IndexedDataset, RangeDataset, _RangeSlice
from .ordering import _UINT64_MASK, _mix64


@dataclass(frozen=True, slots=True)
class _ConcatRange[T](RangeDataset[T]):
    components: tuple[RangeDataset[T], ...]
    _ends: tuple[int, ...] = field(repr=False)

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Exact:
        return Exact(self._ends[-1] if self._ends else 0)

    @property
    def description(self) -> str:
        return f"Concat(count={len(self.components)})"

    def _open_range(self, start: int, stop: int) -> Cursor[T]:
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
        return _ConcatCursor(self, tuple(selected))


@dataclass(frozen=True, slots=True)
class _ConcatIndexed[T](IndexedDataset[T]):
    components: tuple[IndexedDataset[T], ...]
    _ends: tuple[int, ...] = field(repr=False)

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
    components: tuple[Dataset[T], ...]

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Cardinality:
        return _sum_cardinality(self.components)

    @property
    def description(self) -> str:
        return f"Concat(count={len(self.components)})"

    def cursor(self) -> Cursor[T]:
        return _ConcatCursor(self, self.components)


@dataclass(frozen=True, slots=True)
class _InterleaveDataset[T](Dataset[T]):
    components: tuple[Dataset[T], ...]

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Cardinality:
        return _sum_cardinality(self.components)

    @property
    def description(self) -> str:
        return f"Interleave(count={len(self.components)})"

    def cursor(self) -> Cursor[T]:
        return _InterleaveCursor(self, self.components)


@dataclass(frozen=True, slots=True)
class _MixDataset[T](Dataset[T]):
    components: tuple[Dataset[T], ...]
    weights: tuple[float, ...]
    seed: int

    @property
    def parents(self) -> tuple[Dataset[Any], ...]:
        return self.components

    @property
    def cardinality(self) -> Cardinality:
        return _sum_cardinality(self.components)

    @property
    def description(self) -> str:
        return f"Mix(weights={self.weights!r}, seed={self.seed})"

    def cursor(self) -> Cursor[T]:
        return _MixCursor(self)

    def _select(self, active: list[bool], draw: int) -> int:
        total = sum(
            weight
            for weight, is_active in zip(self.weights, active, strict=True)
            if is_active
        )
        random = _mix64((self.seed & _UINT64_MASK) ^ _mix64(draw)) >> 11
        target = random * (2.0**-53) * total
        cumulative = 0.0
        selected = -1
        for index, (weight, is_active) in enumerate(
            zip(self.weights, active, strict=True)
        ):
            if not is_active:
                continue
            selected = index
            cumulative += weight
            if target < cumulative:
                return index
        assert selected >= 0
        return selected


@dataclass(frozen=True, slots=True)
class _ZipRange[T, U](RangeDataset[tuple[T, U]]):
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

    def _open_range(self, start: int, stop: int) -> Cursor[tuple[T, U]]:
        return _ZipCursor(
            self,
            self.left.open_range(start, stop),
            self.right.open_range(start, stop),
            self.strict,
        )


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
        return _ZipCursor(
            self,
            self.left.cursor(),
            self.right.cursor(),
            self.strict,
        )


class _ConcatCursor[T](Cursor[T]):
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

    def _restore(self, state: State) -> None:
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
            self._active._restore(active_state)

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

    def _snapshot(self) -> State:
        return {
            "component": self._component,
            "active": None if self._active is None else self._active._snapshot(),
        }


class _InterleaveCursor[T](Cursor[T]):
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

    def _snapshot(self) -> State:
        return {
            "component": self._component,
            "active": list(self._active),
            "parents": [cursor._snapshot() for cursor in self._cursors],
        }

    def _restore(self, state: State) -> None:
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
            cursor._restore(parent_state)
        self._component = component
        self._active = active
        self._remaining = sum(active)


class _MixCursor[T](Cursor[T]):
    def __init__(self, dataset: _MixDataset[T]):
        super().__init__(dataset)
        self._mix_dataset = dataset
        self._cursors = tuple(
            self.enter_context(component.cursor()) for component in dataset.components
        )
        self._active = [True] * len(self._cursors)
        self._remaining = len(self._cursors)
        self._draw = 0

    def _next(self) -> T:
        while self._remaining:
            component = self._mix_dataset._select(self._active, self._draw)
            self._draw += 1
            try:
                return next(self._cursors[component])
            except StopIteration:
                self._active[component] = False
                self._remaining -= 1
        raise StopIteration

    def _snapshot(self) -> State:
        return {
            "draw": self._draw,
            "active": list(self._active),
            "parents": [cursor._snapshot() for cursor in self._cursors],
        }

    def _restore(self, state: State) -> None:
        try:
            draw = to_index(state["draw"])
            active = list(state["active"])
            parents = list(state["parents"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid mix checkpoint") from error
        if (
            draw < 0
            or len(active) != len(self._cursors)
            or any(type(value) is not bool for value in active)
            or len(parents) != len(self._cursors)
        ):
            raise ValueError("mix checkpoint has invalid state")
        for cursor, parent_state in zip(self._cursors, parents, strict=True):
            cursor._restore(parent_state)
        self._draw = draw
        self._active = active
        self._remaining = sum(active)


class _ZipCursor[T, U](Cursor[tuple[T, U]]):
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

    def _snapshot(self) -> State:
        return {
            "left": self._left._snapshot(),
            "right": self._right._snapshot(),
        }

    def _restore(self, state: State) -> None:
        self._left._restore(state["left"])
        self._right._restore(state["right"])


def _concat_datasets[T](first: Dataset[T], others: tuple[Dataset[T], ...]) -> Dataset[T]:
    if not others:
        return first
    components = (first, *others)
    if all(isinstance(component, IndexedDataset) for component in components):
        indexed = cast(tuple[IndexedDataset[T], ...], components)
        return _ConcatIndexed(indexed, _component_ends(indexed))
    if all(isinstance(component, RangeDataset) for component in components):
        ranged = cast(tuple[RangeDataset[T], ...], components)
        return _ConcatRange(ranged, _component_ends(ranged))
    return _ConcatDataset(components)


def _component_ends[T](components: tuple[RangeDataset[T], ...]) -> tuple[int, ...]:
    return tuple(accumulate(map(len, components)))


def _mix_datasets[T](
    first: Dataset[T],
    others: tuple[Dataset[T], ...],
    weights: Iterable[float],
    seed: int,
) -> Dataset[T]:
    try:
        normalized_weights = tuple(float(weight) for weight in weights)
    except (TypeError, ValueError) as error:
        raise TypeError("weights must be an iterable of numbers") from error
    components = (first, *others)
    if len(normalized_weights) != len(components):
        raise ValueError("weights must contain one value per dataset")
    if any(weight <= 0 or not isfinite(weight) for weight in normalized_weights):
        raise ValueError("weights must contain only positive finite values")
    if not isfinite(sum(normalized_weights)):
        raise ValueError("weight sum must be finite")
    seed = to_index(seed)
    if not others:
        return first
    return _MixDataset(components, normalized_weights, seed)


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


def _sum_cardinality(components: tuple[Dataset[Any], ...]) -> Cardinality:
    lower = upper = 0
    unknown = False
    for component in components:
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
