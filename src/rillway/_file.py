from os import fstat, stat
from typing import IO, Any, cast

type _FileIdentity = tuple[int, int, int, int]


def _path_identity(path: str) -> _FileIdentity:
    return _stat_identity(stat(path))


def _open_file_identity(reader: IO[Any]) -> _FileIdentity:
    return _stat_identity(fstat(reader.fileno()))


def _parse_file_identity(value: object) -> _FileIdentity:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("invalid source file identity")
    return cast(_FileIdentity, value)


def _stat_identity(info: Any) -> _FileIdentity:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
