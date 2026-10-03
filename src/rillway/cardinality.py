from __future__ import annotations

from dataclasses import dataclass
from operator import index as to_index


class Exact(int):
    def __new__(cls, value: int) -> Exact:
        value = to_index(value)
        if value < 0:
            raise ValueError("exact cardinality must be nonnegative")
        return super().__new__(cls, value)


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper < self.lower:
            raise ValueError("cardinality bounds must satisfy 0 <= lower <= upper")


@dataclass(frozen=True, slots=True)
class Infinite:
    pass


@dataclass(frozen=True, slots=True)
class Unknown:
    pass


type Cardinality = Exact | Bounds | Infinite | Unknown
