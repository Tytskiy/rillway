import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TextIO

from ._file import (
    _FileIdentity,
    _load_file_position,
    _save_file_position,
    _validate_open_file,
)
from .cardinality import Cardinality, Unknown
from .cursor import Cursor, State
from .dataset import Dataset


@dataclass(frozen=True, slots=True, init=False)
class CsvDataset(Dataset[dict[str, str]]):
    supports_checkpointing = True

    path: str
    delimiter: str
    columns: tuple[str, ...] | None
    encoding: str

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        delimiter: str = ",",
        columns: Iterable[str] | None = None,
        encoding: str = "utf-8",
    ):
        if not isinstance(delimiter, str):
            raise TypeError("delimiter must be a string")
        if len(delimiter) != 1:
            raise ValueError("delimiter must be one character")
        if isinstance(columns, str):
            raise TypeError("columns must be an iterable of strings")
        normalized_columns = None if columns is None else tuple(columns)
        if normalized_columns is not None:
            if not normalized_columns:
                raise ValueError("columns must not be empty")
            if any(not isinstance(column, str) for column in normalized_columns):
                raise TypeError("columns must contain only strings")
            if len(set(normalized_columns)) != len(normalized_columns):
                raise ValueError("columns must be unique")
        if not isinstance(encoding, str):
            raise TypeError("encoding must be a string")
        if not encoding:
            raise ValueError("encoding must not be empty")
        object.__setattr__(self, "path", str(Path(path).absolute()))
        object.__setattr__(self, "delimiter", delimiter)
        object.__setattr__(self, "columns", normalized_columns)
        object.__setattr__(self, "encoding", encoding)

    @property
    def cardinality(self) -> Cardinality:
        return Unknown()

    @property
    def description(self) -> str:
        return (
            f"Csv(path={self.path!r}, delimiter={self.delimiter!r}, "
            f"columns={self.columns!r}, encoding={self.encoding!r})"
        )

    def cursor(self) -> Cursor[dict[str, str]]:
        return _CsvCursor(self)


class _CsvLines(Iterator[str]):
    def __init__(self, reader: TextIO):
        self._reader = reader

    def __next__(self) -> str:
        line = self._reader.readline()
        if not line:
            raise StopIteration
        return line


class _CsvCursor(Cursor[dict[str, str]]):
    def __init__(self, dataset: CsvDataset):
        super().__init__(dataset)
        self._csv_dataset = dataset
        self._offset = 0
        self._reader: TextIO | None = None
        self._rows: Iterator[list[str]] | None = None
        self._columns: tuple[str, ...] = ()
        self._identity: _FileIdentity | None = None

    def _open(self) -> None:
        self._reader = open(
            self._csv_dataset.path,
            encoding=self._csv_dataset.encoding,
            newline="",
        )
        self.callback(self._reader.close)
        self._identity = _validate_open_file(self._reader, self._identity)
        rows = csv.reader(_CsvLines(self._reader), delimiter=self._csv_dataset.delimiter)
        if self._csv_dataset.columns is None:
            try:
                self._columns = tuple(next(rows))
            except StopIteration:
                self._rows = iter(())
                return
            if len(set(self._columns)) != len(self._columns):
                raise ValueError("CSV header contains duplicate columns")
        else:
            self._columns = self._csv_dataset.columns
        if self._offset:
            self._reader.seek(self._offset)
            rows = csv.reader(_CsvLines(self._reader), delimiter=self._csv_dataset.delimiter)
        self._rows = rows

    def _next(self) -> dict[str, str]:
        if self._rows is None:
            self._open()
        assert self._rows is not None
        row = next(self._rows)
        if len(row) != len(self._columns):
            raise ValueError("CSV row does not match the header")
        assert self._reader is not None
        self._offset = self._reader.tell()
        return dict(zip(self._columns, row, strict=True))

    def _snapshot(self) -> State:
        state, self._identity = _save_file_position(
            self._csv_dataset.path,
            self._reader,
            self._identity,
            self._offset,
        )
        return state

    def _restore(self, state: State) -> None:
        self._offset, self._identity = _load_file_position(
            state,
            self._csv_dataset.path,
        )
