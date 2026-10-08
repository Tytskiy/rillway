import json
from dataclasses import dataclass
from typing import BinaryIO, cast

from ..cardinality import Cardinality, Unknown
from ..cursor import Cursor, State
from ..dataset import Dataset
from ._file import (
    _File,
    _FileIdentity,
    _FilePath,
    _load_file_position,
    _save_file_position,
    _validate_file,
)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


@dataclass(init=False)
class JsonlDataset(Dataset[JsonValue]):
    path: str
    _file: _File

    def __init__(self, path: _FilePath):
        file = _File.from_path(path)
        self.path = file.uri
        self._file = file

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
            self._reader = self._jsonl_dataset._file.open_binary()
            self.callback(self._reader.close)
            self._identity = _validate_file(
                self._jsonl_dataset._file,
                self._identity,
                self._reader,
            )
            self._reader.seek(self._offset)
        line = self._reader.readline()
        if not line:
            raise StopIteration
        self._offset = self._reader.tell()
        return cast(JsonValue, json.loads(line))

    def _snapshot(self) -> State:
        state, self._identity = _save_file_position(
            self._jsonl_dataset._file,
            self._reader,
            self._identity,
            self._offset,
        )
        return state

    def _restore(self, state: State) -> None:
        self._offset, self._identity = _load_file_position(
            state,
            self._jsonl_dataset._file,
        )
