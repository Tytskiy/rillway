import gc

import pytest

from rillway import (
    Bounds,
    Cursor,
    Dataset,
    Exact,
    IndexedDataset,
    IndexedSource,
    RangeDataset,
    Unknown,
)


def resource_stream(values, closed):
    def factory():
        try:
            yield from values
        finally:
            closed.append(True)

    return Dataset.from_factory(factory)


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


def test_cursor_close_is_idempotent_and_propagates():
    closed = []
    cursor = resource_stream([1], closed).map(str).cursor()
    assert next(cursor) == "1"

    cursor.close()
    cursor.close()

    assert cursor.closed
    assert closed == [True]


def test_abandoned_cursor_is_closed_by_finalizer():
    closed = []
    cursor = resource_stream([1], closed).map(str).cursor()
    assert next(cursor) == "1"
    del cursor
    gc.collect()
    assert closed == [True]


def test_abandoned_flat_map_cursor_closes_the_active_child():
    closed = []

    def expand(value):
        try:
            yield value
            yield value + 1
        finally:
            closed.append(value)

    cursor = IndexedDataset.from_source([1]).flat_map(expand).cursor()
    assert next(cursor) == 1
    del cursor
    gc.collect()

    assert closed == [1]


def test_natural_exhaustion_closes_the_cursor_chain():
    closed = []
    cursor = resource_stream([1], closed).map(str).cursor()
    assert list(cursor) == ["1"]
    assert cursor.closed
    assert closed == [True]


def test_traversal_error_closes_the_cursor_chain():
    closed = []

    def fail(_):
        raise RuntimeError("broken callback")

    cursor = resource_stream([1], closed).map(fail).cursor()
    with pytest.raises(RuntimeError, match="broken callback"):
        next(cursor)

    assert cursor.closed
    assert closed == [True]


def test_owned_context_receives_with_block_exception():
    received = []

    class Resource:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _traceback):
            received.append(exc_type)

    class ManagedCursor(Cursor[int]):
        def __init__(self):
            super().__init__()
            self.enter_context(Resource())

        def _next(self):
            raise StopIteration

    with pytest.raises(ValueError, match="failed block"), ManagedCursor():
        raise ValueError("failed block")

    assert received == [ValueError]


def test_owned_context_receives_traversal_exception():
    received = []

    class Resource:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _traceback):
            received.append(exc_type)

    class FailingCursor(Cursor[int]):
        def __init__(self):
            super().__init__()
            self.enter_context(Resource())

        def _next(self):
            raise RuntimeError("failed traversal")

    cursor = FailingCursor()
    with pytest.raises(RuntimeError, match="failed traversal"):
        next(cursor)

    assert cursor.closed
    assert received == [RuntimeError]


def test_filter_loses_indexing_but_map_preserves_it():
    indexed = IndexedDataset.from_source(range(5)).map(str)
    assert isinstance(indexed, IndexedDataset)
    assert indexed[2] == "2"

    streamed = indexed.filter(lambda value: value != "2")
    assert isinstance(streamed, Dataset)
    with pytest.raises(TypeError):
        len(streamed)
    assert not hasattr(streamed, "__getitem__")


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


def test_stream_take_tracks_bounds_and_closes_upstream_early():
    stream = IndexedDataset.from_source(range(10)).filter(lambda value: value % 2)
    limited = stream.take(2)
    assert limited.cardinality == Bounds(0, 2)
    assert list(limited) == [1, 3]

    closed = []
    cursor = resource_stream([1, 3, 5], closed).take(2).cursor()
    assert list(cursor) == [1, 3]
    assert cursor.closed
    assert closed == [True]


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


def test_concat_preserves_indexing_only_when_every_parent_is_indexed():
    left = IndexedDataset.from_source([1, 2])
    middle = IndexedDataset.from_source([])
    right = IndexedDataset.from_source([3, 4, 5])

    indexed = left.concat(middle, right)
    assert isinstance(indexed, IndexedDataset)
    assert indexed.cardinality == Exact(5)
    assert indexed[3] == 4
    assert list(indexed) == [1, 2, 3, 4, 5]

    streamed = left.concat(right.filter(lambda value: value != 4))
    assert isinstance(streamed, Dataset)
    assert streamed.cardinality == Bounds(2, 5)
    assert list(streamed) == [1, 2, 3, 5]


def test_stream_concat_opens_components_lazily():
    second_calls = []
    first = IndexedDataset.from_source([1]).filter(lambda value: True)
    second = (
        IndexedDataset.from_source([2])
        .filter(lambda value: True)
        .map(lambda value: second_calls.append(value) or value)
    )
    cursor = first.concat(second).cursor()

    assert next(cursor) == 1
    assert second_calls == []
    assert next(cursor) == 2
    assert second_calls == [2]
    cursor.close()


def test_nested_indexed_concat_reuses_cached_lengths():
    cardinality_calls = 0

    class CountingIndexed(IndexedDataset[int]):
        @property
        def cardinality(self):
            nonlocal cardinality_calls
            cardinality_calls += 1
            return Exact(1)

        @property
        def description(self):
            return "Counting"

        def _get(self, position):
            return position

    dataset = CountingIndexed()
    for _ in range(12):
        dataset = dataset.concat(dataset)

    cardinality_calls = 0
    assert dataset[0] == 0
    assert cardinality_calls == 0

    assert list(CountingIndexed().cursor()) == [0]
    assert cardinality_calls == 1


def test_zip_preserves_indexing_and_defaults_to_shortest():
    left = IndexedDataset.from_source([1, 2, 3])
    right = IndexedDataset.from_source(["a", "b"])
    zipped = left.zip(right)

    assert isinstance(zipped, IndexedDataset)
    assert zipped.cardinality == Exact(2)
    assert zipped[1] == (2, "b")
    assert list(zipped) == [(1, "a"), (2, "b")]


def test_strict_zip_rejects_known_and_runtime_length_mismatches():
    left = IndexedDataset.from_source([1, 2, 3])
    right = IndexedDataset.from_source(["a", "b"])
    with pytest.raises(ValueError, match="different lengths"):
        left.zip(right, strict=True)

    uncertain_left = left.filter(lambda value: value < 3)
    uncertain_right = right.filter(lambda value: True)
    assert list(uncertain_left.zip(uncertain_right, strict=True)) == [
        (1, "a"),
        (2, "b"),
    ]

    shorter = right.filter(lambda value: value == "a")
    cursor = uncertain_left.zip(shorter, strict=True).cursor()
    with pytest.raises(ValueError):
        list(cursor)
    assert cursor.closed


@pytest.mark.parametrize("method,value", [("take", -1), ("skip", -1), ("batch", 0)])
def test_structural_operations_validate_counts(method, value):
    indexed = IndexedDataset.from_source([1])
    stream = indexed.filter(lambda value: True)
    with pytest.raises(ValueError):
        getattr(indexed, method)(value)
    with pytest.raises(ValueError):
        getattr(stream, method)(value)


def test_factory_stream_is_lazy_replayable_and_explicitly_not_checkpointable():
    calls = []

    def factory():
        calls.append(True)
        return iter([1, 2, 3])

    stream = Dataset.from_factory(factory, name="numbers")
    assert stream.cardinality == Unknown()
    assert not stream.checkpointable
    assert not stream.map(str).checkpointable
    assert calls == []
    assert list(stream) == [1, 2, 3]
    assert list(stream) == [1, 2, 3]
    assert len(calls) == 2
    assert stream.explain() == "Factory(name='numbers') [Unknown]"

    cursor = stream.cursor()
    with pytest.raises(TypeError, match="not checkpointable"):
        cursor.state_dict()
    restored = stream.cursor()
    with pytest.raises(TypeError, match="not checkpointable"):
        restored.load_state_dict({"position": 1})


def test_custom_dataset_must_opt_in_to_checkpointing():
    class CustomDataset(Dataset[int]):
        @property
        def cardinality(self):
            return Unknown()

        @property
        def description(self):
            return "Custom"

        def cursor(self):
            return Cursor.from_iterator(iter([1]))

    class CheckpointableDataset(CustomDataset):
        supports_checkpointing = True

    assert not CustomDataset().checkpointable
    assert CheckpointableDataset().checkpointable


def test_cursor_from_iterator_is_explicitly_one_shot():
    cursor = Cursor.from_iterator(iter([1, 2]))
    assert list(cursor) == [1, 2]
    assert list(cursor) == []


def test_cursor_state_must_be_loaded_before_iteration():
    cursor = IndexedDataset.from_source([1, 2, 3]).cursor()
    assert next(cursor) == 1

    with pytest.raises(RuntimeError, match="after iteration has started"):
        cursor.load_state_dict({"position": 0})

    assert list(cursor) == [2, 3]


def test_cursor_factory_is_the_supported_custom_stateful_entry_point():
    class CounterCursor(Cursor[int]):
        def __init__(self, stop):
            super().__init__()
            self.position = 0
            self.stop = stop

        def _next(self):
            if self.position == self.stop:
                raise StopIteration
            value = self.position
            self.position += 1
            return value

        def state_dict(self):
            return {"position": self.position}

        def _load_state_dict(self, state):
            self.position = state["position"]

    stream = Dataset.from_cursor_factory(
        lambda: CounterCursor(4),
        cardinality=Exact(4),
        name="counter",
    )
    cursor = stream.cursor()
    assert [next(cursor), next(cursor)] == [0, 1]
    state = cursor.state_dict()

    assert stream.checkpointable
    resumed = stream.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == [2, 3]
    assert stream.explain() == "CursorFactory(name='counter') [Exact(4)]"


def test_custom_cursor_can_own_contexts_and_cleanup_callbacks():
    events = []

    class Resource:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_):
            events.append("exit")

    class ManagedCursor(Cursor[int]):
        def __init__(self):
            super().__init__()
            self.enter_context(Resource())
            self.callback(events.append, "callback")

        def _next(self):
            raise StopIteration

    assert list(ManagedCursor()) == []
    assert events == ["enter", "callback", "exit"]


def test_composed_cursor_can_resume_from_state():
    dataset = (
        IndexedDataset.from_source(range(8))
        .filter(lambda value: value % 2)
        .map(lambda value: value * 10)
        .batch(2)
    )
    assert dataset.checkpointable

    cursor = dataset.cursor()
    assert next(cursor) == (10, 30)
    state = cursor.state_dict()
    expected = list(cursor)

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [(50, 70)]


def test_flat_map_checkpoint_tracks_current_input_and_child_offset():
    dataset = IndexedDataset.from_source([1, 2, 3]).flat_map(range, name="expand")
    cursor = dataset.cursor()

    assert next(cursor) == 0
    assert next(cursor) == 0
    state = cursor.state_dict()
    assert state["current"] == 2
    assert state["child_offset"] == 1

    expected = list(cursor)
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [1, 0, 1, 2]


def test_flat_map_restores_after_take_parent_reaches_its_limit():
    dataset = (
        IndexedDataset.from_source([1])
        .filter(lambda value: True)
        .take(1)
        .flat_map(lambda value: [value, value])
    )

    with dataset.cursor() as cursor:
        assert next(cursor) == 1
        state = cursor.state_dict()

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == [1]


def test_concat_and_zip_checkpoint_their_parent_cursors():
    concatenated = (
        IndexedDataset.from_source([1, 2]).filter(bool).concat(IndexedDataset.from_source([3, 4]))
    )
    concat_cursor = concatenated.cursor()
    assert next(concat_cursor) == 1
    concat_state = concat_cursor.state_dict()
    resumed_concat = concatenated.cursor()
    resumed_concat.load_state_dict(concat_state)
    assert list(resumed_concat) == [2, 3, 4]

    zipped = (
        IndexedDataset.from_source([1, 2, 3])
        .filter(bool)
        .zip(IndexedDataset.from_source(["a", "b", "c"]))
    )
    zip_cursor = zipped.cursor()
    assert next(zip_cursor) == (1, "a")
    zip_state = zip_cursor.state_dict()
    resumed_zip = zipped.cursor()
    resumed_zip.load_state_dict(zip_state)
    assert list(resumed_zip) == [(2, "b"), (3, "c")]
