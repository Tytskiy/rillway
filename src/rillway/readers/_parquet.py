from collections.abc import Generator, Iterable
from dataclasses import dataclass
from hashlib import sha256
from operator import index as to_index
from typing import Any, BinaryIO, ClassVar, Protocol, cast

from ..cursor import Cursor, State
from ._file import _File, _FileIdentity, _validate_file


@dataclass(frozen=True)
class _ParquetFile:
    source: _File
    identity: _FileIdentity
    row_groups: tuple[int, ...]

    @property
    def length(self) -> int:
        return sum(self.row_groups)


class _ParquetSource(Protocol):
    columns: tuple[str, ...] | None
    _files: tuple[_ParquetFile, ...]
    _file_ends: tuple[int, ...]
    _fingerprint: str
    _accept_legacy_position_checkpoint: ClassVar[bool]

    def _parquet_module(self) -> Any: ...


def _normalize_columns(columns: Iterable[str] | None) -> tuple[str, ...] | None:
    if isinstance(columns, str):
        raise TypeError("columns must be an iterable of strings")
    normalized = None if columns is None else tuple(columns)
    if normalized is not None:
        if not normalized:
            raise ValueError("columns must not be empty")
        if any(not isinstance(column, str) for column in normalized):
            raise TypeError("columns must contain only strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError("columns must be unique")
    return normalized


def _read_row_groups(parquet: Any, reader: BinaryIO) -> tuple[int, ...]:
    parquet_file = parquet.ParquetFile(reader)
    try:
        metadata = parquet_file.metadata
        return tuple(
            metadata.row_group(index).num_rows
            for index in range(metadata.num_row_groups)
        )
    finally:
        parquet_file.close()


def _parquet_layout(
    files: Iterable[_ParquetFile],
) -> tuple[tuple[int, ...], int]:
    file_ends = []
    length = 0
    for file in files:
        length += file.length
        file_ends.append(length)
    return tuple(file_ends), length


def _parquet_fingerprint(files: Iterable[_ParquetFile]) -> str:
    digest = sha256()
    for file in files:
        digest.update(file.source.uri.encode())
        digest.update(b"\0")
        digest.update(repr(file.identity).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _parquet_path_fingerprint(files: Iterable[_ParquetFile]) -> str:
    return sha256(
        "\0".join(file.source.path for file in files).encode()
    ).hexdigest()[:16]


class _ParquetCursor(Cursor[dict[str, object]]):
    def __init__(self, source: _ParquetSource, start: int, stop: int):
        super().__init__()
        self._source = source
        self._start = start
        self._stop = stop
        self._position = start
        self._rows: Generator[dict[str, object]] | None = None

    def _next(self) -> dict[str, object]:
        if self._position == self._stop:
            raise StopIteration
        if self._rows is None:
            rows = self._iter_rows()
            self.callback(rows.close)
            self._rows = rows
        try:
            row = next(self._rows)
        except StopIteration as error:
            raise RuntimeError("Parquet range ended before its declared stop") from error
        self._position += 1
        return row

    def _iter_rows(self) -> Generator[dict[str, object]]:
        parquet = self._source._parquet_module()
        file_start = 0
        for file, file_stop in zip(
            self._source._files,
            self._source._file_ends,
            strict=True,
        ):
            if self._position >= file_stop:
                file_start = file_stop
                continue
            if file_start >= self._stop:
                break
            local_start = max(self._position, file_start) - file_start
            local_stop = min(self._stop, file_stop) - file_start
            yield from self._read_file(parquet, file, local_start, local_stop)
            file_start = file_stop

    def _read_file(
        self,
        parquet: Any,
        file: _ParquetFile,
        start: int,
        stop: int,
    ) -> Generator[dict[str, object]]:
        first, offset = _locate(file, start)
        last, _ = _locate(file, stop - 1)
        remaining = stop - start
        with file.source.open_binary() as reader:
            _validate_file(file.source, file.identity, reader)
            parquet_file = parquet.ParquetFile(reader)
            try:
                batches = parquet_file.iter_batches(
                    row_groups=list(range(first, last + 1)),
                    columns=(
                        None
                        if self._source.columns is None
                        else list(self._source.columns)
                    ),
                )
                for batch in batches:
                    rows = cast(list[dict[str, object]], batch.to_pylist())
                    for row in rows:
                        if offset:
                            offset -= 1
                        elif remaining:
                            remaining -= 1
                            yield row
                        else:
                            return
            finally:
                parquet_file.close()

    def _snapshot(self) -> State:
        self._validate_position_file(RuntimeError)
        return {
            "position": self._position,
            "source": self._source._fingerprint,
        }

    def _restore(self, state: State) -> None:
        try:
            position = to_index(state["position"])
        except (KeyError, TypeError) as error:
            raise ValueError("invalid Parquet checkpoint") from error
        fingerprint = state.get("source")
        legacy_identity = (
            self._source._files[0].identity
            if len(self._source._files) == 1
            else None
        )
        if fingerprint is None:
            if not self._source._accept_legacy_position_checkpoint:
                raise ValueError("invalid Parquet checkpoint")
        elif (
            fingerprint != self._source._fingerprint
            and fingerprint != legacy_identity
        ):
            raise ValueError("Parquet checkpoint belongs to a different source")
        if not self._start <= position <= self._stop:
            raise ValueError("Parquet checkpoint position is outside the requested range")
        self._position = position
        self._validate_position_file(ValueError)

    def _validate_position_file(self, error_type: type[Exception]) -> None:
        if self._position == self._stop:
            return
        for file, file_stop in zip(
            self._source._files,
            self._source._file_ends,
            strict=True,
        ):
            if self._position < file_stop:
                try:
                    _validate_file(file.source, file.identity)
                except ValueError as error:
                    raise error_type("source file changed") from error
                return
        raise RuntimeError("Parquet metadata does not cover the requested range")


def _locate(file: _ParquetFile, position: int) -> tuple[int, int]:
    row_group_start = 0
    for index, size in enumerate(file.row_groups):
        if position < row_group_start + size:
            return index, position - row_group_start
        row_group_start += size
    raise RuntimeError("Parquet metadata does not cover the requested range")
