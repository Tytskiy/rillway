import gc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from operator import index as to_index

import pytest

from rillway import Cursor, Dataset, Exact, IndexedDataset, RangeDataset, State


class PositionCursor(Cursor[int]):
    def __init__(self, start: int, stop: int):
        super().__init__()
        self.start = start
        self.position = start
        self.stop = stop

    def _restore(self, state: State) -> None:
        position = to_index(state["position"])
        if not self.start <= position <= self.stop:
            raise ValueError("cursor position is outside the requested range")
        self.position = position

    def _next(self) -> int:
        if self.position == self.stop:
            raise StopIteration
        value = self.position
        self.position += 1
        return value

    def _snapshot(self) -> State:
        return {"position": self.position}


class RecordingRange(RangeDataset[int]):
    def __init__(self, length: int):
        self.length = length
        self.opens: list[tuple[int, int]] = []

    @property
    def cardinality(self) -> Exact:
        return Exact(self.length)

    @property
    def description(self) -> str:
        return "RecordingRange"

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[int]:
        self.opens.append((start, stop))
        return PositionCursor(start, stop)


class FakeHDFS:
    def __init__(self, records: list[int], *, fail_at: int | None = None):
        self.records = records
        self.fail_at = fail_at
        self.opens: list[tuple[str, int, int, int]] = []
        self.closes: list[tuple[str, int, int]] = []

    @contextmanager
    def scan_rows(
        self,
        path: str,
        start: int,
        stop: int,
        chunk_size: int,
    ) -> Iterator[Iterator[int]]:
        self.opens.append((path, start, stop, chunk_size))
        try:
            yield self._rows(start, stop)
        finally:
            self.closes.append((path, start, stop))

    def _rows(self, start: int, stop: int) -> Iterator[int]:
        for position in range(start, stop):
            if position == self.fail_at:
                raise OSError("HDFS read failed")
            yield self.records[position]


class HDFSCursor(Cursor[int]):
    def __init__(
        self,
        client: FakeHDFS,
        path: str,
        version: str,
        start: int,
        stop: int,
        chunk_size: int,
    ):
        super().__init__()
        self.client = client
        self.path = path
        self.version = version
        self.start = start
        self.stop = stop
        self.chunk_size = chunk_size
        self.position = start
        self.reader: Iterator[int] | None = None

    def _next(self) -> int:
        if self.position == self.stop:
            raise StopIteration
        if self.reader is None:
            self.reader = self.enter_context(
                self.client.scan_rows(
                    self.path,
                    self.position,
                    self.stop,
                    self.chunk_size,
                )
            )
        try:
            value = next(self.reader)
        except StopIteration as error:
            raise RuntimeError("HDFS range ended before its declared stop") from error
        self.position += 1
        return value

    def _snapshot(self) -> State:
        return {
            "path": self.path,
            "version": self.version,
            "position": self.position,
        }

    def _restore(self, state: State) -> None:
        if (state.get("path"), state.get("version")) != (self.path, self.version):
            raise ValueError("checkpoint belongs to a different HDFS source")
        position = to_index(state["position"])
        if not self.start <= position <= self.stop:
            raise ValueError("cursor position is outside the requested range")
        self.position = position


@dataclass(frozen=True, slots=True)
class HDFSDataset(RangeDataset[int]):
    client: FakeHDFS
    path: str
    version: str
    length: int
    chunk_size: int

    @property
    def cardinality(self) -> Exact:
        return Exact(self.length)

    @property
    def description(self) -> str:
        return f"HDFS(path={self.path!r}, version={self.version!r})"

    def _open_range(self, start: int, stop: int) -> Cursor[int]:
        return HDFSCursor(
            self.client,
            self.path,
            self.version,
            start,
            stop,
            self.chunk_size,
        )


def test_hdfs_cursor_opens_on_first_read_at_the_restored_position():
    client = FakeHDFS(list(range(10)))
    source = HDFSDataset(client, "/events", "v1", 10, 4)
    dataset = source.map(lambda value: value * 10).shard(1, 2)

    cursor = dataset.cursor()
    assert client.opens == []
    assert next(cursor) == 50
    checkpoint = cursor.state_dict()
    cursor.close()
    assert client.opens == [("/events", 5, 10, 4)]
    assert client.closes == [("/events", 5, 10)]

    resumed = dataset.cursor()
    resumed.load_state_dict(checkpoint)
    assert client.opens == [("/events", 5, 10, 4)]
    assert next(resumed) == 60
    assert client.opens[-1] == ("/events", 6, 10, 4)
    assert list(resumed) == [70, 80, 90]
    assert client.closes[-1] == ("/events", 6, 10)


def test_hdfs_checkpoint_rejects_a_different_source_or_range():
    client = FakeHDFS(list(range(10)))
    source = HDFSDataset(client, "/events", "v1", 10, 4)
    cursor = source.open_range(2, 8)
    assert next(cursor) == 2
    checkpoint = cursor.state_dict()
    cursor.close()

    mismatches = [
        HDFSDataset(client, "/other", "v1", 10, 4).open_range(2, 8),
        HDFSDataset(client, "/events", "v2", 10, 4).open_range(2, 8),
        source.open_range(3, 8),
    ]
    for resumed in mismatches:
        with pytest.raises(ValueError, match="checkpoint does not match|requested range"):
            resumed.load_state_dict(checkpoint)
        assert resumed.closed

    assert client.opens == [("/events", 2, 8, 4)]


def test_hdfs_reader_closes_on_failure():
    client = FakeHDFS([0, 1, 2], fail_at=1)
    cursor = HDFSDataset(client, "/events", "v1", 3, 2).cursor()

    assert next(cursor) == 0
    with pytest.raises(OSError, match="HDFS read failed"):
        next(cursor)

    assert cursor.closed
    assert client.closes == [("/events", 0, 3)]


def test_abandoned_hdfs_cursor_closes_its_reader():
    client = FakeHDFS([0, 1, 2])
    cursor = HDFSDataset(client, "/events", "v1", 3, 2).cursor()
    assert next(cursor) == 0

    del cursor
    gc.collect()

    assert client.closes == [("/events", 0, 3)]


def test_shards_are_lazy_balanced_disjoint_and_complete():
    source = RecordingRange(10)
    shards = [source.shard(index, 3) for index in range(3)]

    assert source.opens == []
    assert [shard.cardinality for shard in shards] == [Exact(3), Exact(3), Exact(4)]
    assert [list(shard) for shard in shards] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
    ]
    assert source.opens == [(0, 3), (3, 6), (6, 10)]


def test_repeat_preserves_range_access():
    source = RecordingRange(3)
    repeated = source.repeat(3)

    assert isinstance(repeated, RangeDataset)
    assert len(repeated) == 9
    assert list(repeated.open_range(2, 7)) == [2, 0, 1, 2, 0]
    assert source.opens == [(2, 3), (0, 3), (0, 1)]


def test_map_and_shard_propagate_one_range_to_the_source():
    source = RecordingRange(16)
    dataset = source.map(lambda value: value * 2, name="double").shard(2, 4)

    assert isinstance(dataset, RangeDataset)
    assert dataset.cardinality == Exact(4)
    assert source.opens == []
    assert list(dataset) == [16, 18, 20, 22]
    assert source.opens == [(8, 12)]


def test_take_and_skip_translate_nested_ranges():
    source = RecordingRange(12)
    dataset = source.skip(3).take(5).shard(1, 2)

    assert list(dataset) == [5, 6, 7]
    assert source.opens == [(5, 8)]


def test_batch_then_shard_uses_batch_coordinates():
    source = RecordingRange(10)
    dataset = source.batch(4).shard(1, 2)

    assert dataset.cardinality == Exact(2)
    assert list(dataset) == [(4, 5, 6, 7), (8, 9)]
    assert source.opens == [(4, 10)]


def test_shard_then_batch_uses_record_coordinates():
    source = RecordingRange(10)
    dataset = source.shard(1, 2).batch(4)

    assert dataset.cardinality == Exact(2)
    assert list(dataset) == [(5, 6, 7, 8), (9,)]
    assert source.opens == [(5, 10)]


def test_range_pipeline_resumes_at_the_next_record():
    source = RecordingRange(10)
    dataset = source.map(lambda value: value * 10).shard(1, 2)
    cursor = dataset.cursor()

    assert next(cursor) == 50
    checkpoint = cursor.state_dict()
    cursor.close()

    resumed = dataset.cursor()
    resumed.load_state_dict(checkpoint)
    assert list(resumed) == [60, 70, 80, 90]
    assert source.opens == [(5, 10), (5, 10)]


def test_checkpoint_is_bound_to_its_requested_range():
    source = RecordingRange(10)
    first = source.shard(0, 2)
    second = source.shard(1, 2)
    cursor = first.cursor()
    assert next(cursor) == 0
    checkpoint = cursor.state_dict()
    cursor.close()

    resumed = second.cursor()
    with pytest.raises(ValueError, match="checkpoint does not match"):
        resumed.load_state_dict(checkpoint)
    assert resumed.closed


def test_concat_opens_only_overlapping_component_ranges():
    left = RecordingRange(4)
    middle = RecordingRange(5)
    right = RecordingRange(3)
    dataset = left.concat(middle, right)

    assert isinstance(dataset, RangeDataset)
    assert dataset.cardinality == Exact(12)
    assert list(dataset.open_range(2, 10)) == [2, 3, 0, 1, 2, 3, 4, 0]
    assert left.opens == [(2, 4)]
    assert middle.opens == [(0, 5)]
    assert right.opens == [(0, 1)]


def test_zip_and_shard_open_the_same_range_on_both_parents():
    left = RecordingRange(10)
    right = RecordingRange(10)
    dataset = left.zip(right).shard(1, 2)

    assert isinstance(dataset, RangeDataset)
    assert list(dataset) == [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    assert left.opens == [(5, 10)]
    assert right.opens == [(5, 10)]


def test_concat_and_zip_lose_range_access_when_a_parent_is_streamed():
    source = RecordingRange(4)
    stream = source.filter(lambda value: True)

    concatenated = source.concat(stream)
    zipped = source.zip(stream)
    assert isinstance(concatenated, Dataset)
    assert isinstance(zipped, Dataset)
    assert not isinstance(concatenated, (RangeDataset, IndexedDataset))
    assert not isinstance(zipped, (RangeDataset, IndexedDataset))


def test_mixed_indexed_and_range_composition_preserves_range_access():
    indexed = IndexedDataset.from_source([10, 11])
    ranged = RecordingRange(2)

    concatenated = indexed.concat(ranged)
    zipped = indexed.zip(ranged)

    assert isinstance(concatenated, RangeDataset)
    assert not isinstance(concatenated, IndexedDataset)
    assert list(concatenated) == [10, 11, 0, 1]
    assert isinstance(zipped, RangeDataset)
    assert not isinstance(zipped, IndexedDataset)
    assert list(zipped) == [(10, 0), (11, 1)]


def test_data_dependent_operations_lose_range_access():
    source = RecordingRange(5)

    filtered = source.filter(lambda value: value % 2 == 0)
    expanded = source.flat_map(range)

    assert isinstance(filtered, Dataset)
    assert isinstance(expanded, Dataset)
    assert not isinstance(filtered, (RangeDataset, IndexedDataset))
    assert not isinstance(expanded, (RangeDataset, IndexedDataset))
    assert list(filtered) == [0, 2, 4]
    assert list(expanded) == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]


@pytest.mark.parametrize(
    ("start", "stop"),
    [(-1, 1), (2, 1), (0, 4)],
)
def test_open_range_rejects_invalid_bounds(start: int, stop: int):
    with pytest.raises(ValueError, match="0 <= start <= stop <= length"):
        RecordingRange(3).open_range(start, stop)


@pytest.mark.parametrize(("index", "count"), [(-1, 2), (2, 2), (0, 0)])
def test_shard_rejects_invalid_coordinates(index: int, count: int):
    with pytest.raises(ValueError):
        RecordingRange(3).shard(index, count)
