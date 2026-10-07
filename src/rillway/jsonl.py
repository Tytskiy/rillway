import json
from dataclasses import dataclass
from operator import index as to_index
from os import PathLike
from pathlib import Path
from typing import BinaryIO, cast

from ._file import (
    _FileIdentity,
    _open_file_identity,
    _parse_file_identity,
    _path_identity,
)
from .cardinality import Cardinality, Unknown
from .cursor import Cursor, State
from .dataset import Dataset

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True, init=False)
class JsonlDataset(Dataset[JsonValue]):
    supports_checkpointing = True

    path: str

    def __init__(self, path: str | PathLike[str]):
        object.__setattr__(self, "path", str(Path(path).absolute()))

    @property
    def cardinality(self) -> Cardinality:
        return Unknown()

    @property
    def description(self) -> str:
        return f"Jsonl(path={self.path!r})"

    def cursor(self) -> Cursor[JsonValue]:
        return _JsonlCursor(self)


class _JsonlCursor(Cursor[JsonValue]):
    def __init__(self, dataset: JsonlDataset):
        super().__init__(dataset)
        self._jsonl_dataset = dataset
        self._offset = 0
        self._reader: BinaryIO | None = None
        self._identity: _FileIdentity | None = None

    def _next(self) -> JsonValue:
        if self._reader is None:
            self._reader = open(self._jsonl_dataset.path, "rb")
            self.callback(self._reader.close)
            identity = _open_file_identity(self._reader)
            if self._identity is not None and identity != self._identity:
                raise ValueError("JSONL source file changed")
            self._identity = identity
            self._reader.seek(self._offset)
        line = self._reader.readline()
        if not line:
            raise StopIteration
        self._offset = self._reader.tell()
        return cast(JsonValue, json.loads(line))

    def _state_dict(self) -> State:
        identity = (
            _path_identity(self._jsonl_dataset.path)
            if self._reader is None
            else _open_file_identity(self._reader)
        )
        if self._identity is not None and identity != self._identity:
            raise RuntimeError("JSONL source file changed")
        self._identity = identity
        return {"offset": self._offset, "source": identity}

    def _load_state_dict(self, state: State) -> None:
        try:
            offset = to_index(state["offset"])
            identity = _parse_file_identity(state["source"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid JSONL checkpoint") from error
        if offset < 0:
            raise ValueError("JSONL checkpoint has an invalid offset")
        if _path_identity(self._jsonl_dataset.path) != identity:
            raise ValueError("JSONL source file changed")
        self._offset = offset
        self._identity = identity
