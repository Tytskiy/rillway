import pytest

from rillway import (
    Bounds,
    Dataset,
    Exact,
    IndexedDataset,
    IndexedSource,
    RangeDataset,
    Unknown,
)


def test_exact_is_a_nonnegative_int():
    cardinality = Exact(3)

    assert isinstance(cardinality, int)
    assert cardinality == 3
    assert cardinality + 2 == 5
    with pytest.raises(ValueError):
        Exact(-1)


def test_vertical_slice():
    dataset = IndexedDataset.from_source(range(10)).map(
        lambda value: value * 2,
        name="double",
    )

    assert dataset.cardinality == Exact(10)
    assert dataset[2] == 4
    assert list(dataset[2:5]) == [4, 6, 8]

    stream = dataset.filter(lambda value: value % 4 == 0, name="divisible-by-four")
    assert isinstance(stream, Dataset)
    assert stream.cardinality == Bounds(0, 10)
    assert list(stream) == [0, 4, 8, 12, 16]

    with stream.cursor() as cursor:
        assert next(cursor) == 0
    assert cursor.closed
    with pytest.raises(StopIteration):
        next(cursor)


def test_dataset_is_replayable_and_cursor_is_one_shot():
    dataset = IndexedDataset.from_source([1, 2, 3]).filter(lambda value: True)
    assert list(dataset) == [1, 2, 3]
    assert list(dataset) == [1, 2, 3]

    cursor = dataset.cursor()
    assert list(cursor) == [1, 2, 3]
    assert cursor.closed
    assert list(cursor) == []


def test_slicing_is_lazy_and_composes():
    calls = []
    dataset = IndexedDataset.from_source(range(10)).map(
        lambda value: calls.append(value) or value * 10
    )
    selected = dataset[::-2][1:4]
    assert calls == []
    assert selected.cardinality == Exact(3)
    assert selected[1] == 50
    assert calls == [5]
    assert list(selected) == [70, 50, 30]


def test_index_validation_and_negative_indices():
    dataset = IndexedDataset.from_source([10, 20, 30])
    assert dataset[-1] == 30
    with pytest.raises(IndexError):
        _ = dataset[3]
    with pytest.raises(IndexError):
        _ = dataset[-4]
    with pytest.raises(TypeError):
        _ = dataset[1.5]


def test_indexed_dataset_opens_ranges_without_consuming_the_prefix():
    class RecordingSource:
        def __init__(self):
            self.positions = []

        def __len__(self):
            return 10

        def __getitem__(self, position):
            self.positions.append(position)
            return position

    source = RecordingSource()
    dataset = IndexedDataset.from_source(source)

    assert isinstance(dataset, RangeDataset)
    assert list(dataset.open_range(2, 5)) == [2, 3, 4]
    assert source.positions == [2, 3, 4]

    shard = dataset.shard(1, 3)
    assert isinstance(shard, IndexedDataset)
    assert list(shard) == [3, 4, 5]


def test_source_size_change_is_reported():
    source = [1, 2, 3]
    dataset = IndexedDataset.from_source(source)
    source.pop()
    assert len(dataset) == 3
    with pytest.raises(RuntimeError, match="changed from 3 to 2"):
        _ = dataset[0]
    with pytest.raises(RuntimeError):
        list(dataset)


def test_filter_loses_indexing_but_map_preserves_it():
    indexed = IndexedDataset.from_source(range(5)).map(str)
    assert isinstance(indexed, IndexedDataset)
    assert indexed[2] == "2"

    streamed = indexed.filter(lambda value: value != "2")
    assert isinstance(streamed, Dataset)
    with pytest.raises(TypeError):
        len(streamed)
    assert not hasattr(streamed, "__getitem__")


def test_unbatch_flattens_each_input_iterable():
    dataset = IndexedDataset.from_source([(1, 2), (), (3, 4, 5)]).unbatch()

    assert list(dataset) == [1, 2, 3, 4, 5]
    assert dataset.cardinality == Unknown()
    assert dataset.explain().startswith("Unbatch()")


def test_plan_is_inspectable():
    dataset = (
        IndexedDataset.from_source(range(5))
        .map(str, name="stringify")
        .filter(str.isdigit, name="digits")
    )
    assert dataset.explain() == (
        "Filter(name='digits') [Bounds(0, 5)]\n"
        "  Map(name='stringify') [Exact(5)]\n"
        "    Source(type=range) [Exact(5)]"
    )


def test_source_requires_index_protocol():
    assert isinstance(range(3), IndexedSource)
    assert not isinstance(iter([1, 2, 3]), IndexedSource)
    with pytest.raises(TypeError):
        IndexedDataset.from_source(iter([1, 2, 3]))


def test_indexed_take_and_skip_reuse_lazy_slicing():
    dataset = IndexedDataset.from_source(range(6))
    taken = dataset.take(3)
    skipped = dataset.skip(4)

    assert isinstance(taken, IndexedDataset)
    assert taken.cardinality == Exact(3)
    assert list(taken) == [0, 1, 2]
    assert list(skipped) == [4, 5]
    assert taken.explain().startswith("Slice(")


def test_stream_skip_is_lazy_and_updates_bounds():
    consumed = []
    stream = (
        IndexedDataset.from_source(range(8))
        .filter(lambda value: True)
        .map(lambda value: consumed.append(value) or value)
        .skip(3)
    )
    assert stream.cardinality == Bounds(0, 5)
    assert consumed == []
    assert list(stream.take(2)) == [3, 4]
    assert consumed == [0, 1, 2, 3, 4]


def test_batch_preserves_indexing_and_calculates_cardinality():
    dataset = IndexedDataset.from_source(range(5))
    batches = dataset.batch(2)
    dropped = dataset.batch(2, drop_last=True)

    assert isinstance(batches, IndexedDataset)
    assert batches.cardinality == Exact(3)
    assert batches[1] == (2, 3)
    assert list(batches) == [(0, 1), (2, 3), (4,)]
    assert dropped.cardinality == Exact(2)
    assert list(dropped) == [(0, 1), (2, 3)]


def test_stream_batch_handles_partial_final_batch_and_closes():
    stream = IndexedDataset.from_source(range(5)).filter(lambda value: True)
    batches = stream.batch(2)
    assert batches.cardinality == Bounds(0, 3)

    cursor = batches.cursor()
    assert list(cursor) == [(0, 1), (2, 3), (4,)]
    assert cursor.closed
    assert list(stream.batch(2, drop_last=True)) == [(0, 1), (2, 3)]


@pytest.mark.parametrize(
    "method,value",
    [("take", -1), ("skip", -1), ("batch", 0), ("shuffle", 0)],
)
def test_structural_operations_validate_counts(method, value):
    indexed = IndexedDataset.from_source([1])
    stream = indexed.filter(lambda value: True)
    with pytest.raises(ValueError):
        getattr(indexed, method)(value)
    with pytest.raises(ValueError):
        getattr(stream, method)(value)
