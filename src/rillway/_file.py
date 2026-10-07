from operator import index as to_index
from os import fstat, stat
from typing import IO, Any, cast

type _FileIdentity = tuple[int, int, int, int]


def _path_identity(path: str) -> _FileIdentity:
    return _stat_identity(stat(path))


def _open_file_identity(reader: IO[Any]) -> _FileIdentity:
    return _stat_identity(fstat(reader.fileno()))


def _validate_open_file(
    reader: IO[Any],
    expected: _FileIdentity | None,
) -> _FileIdentity:
    identity = _open_file_identity(reader)
    if expected is not None and identity != expected:
        raise ValueError("source file changed")
    return identity


def _save_file_position(
    path: str,
    reader: IO[Any] | None,
    expected: _FileIdentity | None,
    position: int,
) -> tuple[dict[str, Any], _FileIdentity]:
    identity = _path_identity(path) if reader is None else _open_file_identity(reader)
    if expected is not None and identity != expected:
        raise RuntimeError("source file changed")
    return {"position": position, "source": identity}, identity


def _load_file_position(
    state: dict[str, Any],
    path: str,
) -> tuple[int, _FileIdentity]:
    try:
        position = to_index(state["position"])
        identity = cast(_FileIdentity, state["source"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid file checkpoint") from error
    if position < 0:
        raise ValueError("file checkpoint has an invalid position")
    if _path_identity(path) != identity:
        raise ValueError("source file changed")
    return position, identity


def _stat_identity(info: Any) -> _FileIdentity:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
