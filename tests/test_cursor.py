import gc
from threading import Event

import pytest

from rillway import Bounds, Cursor, Dataset, Exact, IndexedDataset, Unknown


def resource_stream(values, closed):
    def factory():
        try:
            yield from values
        finally:
            closed.append(True)

    return Dataset.from_factory(factory)


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


def test_parallel_map_bounds_input_and_closes_upstream_on_early_stop():
    consumed = []
    closed = []

    def factory():
        try:
            for value in range(10):
                consumed.append(value)
                yield value
        finally:
            closed.append(True)

    cursor = (
        Dataset.from_factory(factory)
        .parallel_map(lambda value: value, workers=1, buffer_size=1)
        .cursor()
    )

    assert next(cursor) == 0
    assert consumed == [0, 1, 2]
    cursor.close()
    assert closed == [True]


def test_parallel_map_reports_upstream_errors_in_input_order():
    def factory():
        yield 1
        raise RuntimeError("source failed")

    cursor = Dataset.from_factory(factory).parallel_map(lambda value: value * 2, workers=2).cursor()

    assert next(cursor) == 2
    with pytest.raises(RuntimeError, match="source failed"):
        next(cursor)
    assert cursor.closed


def test_parallel_map_failure_closes_upstream():
    closed = []

    def transform(value):
        if value == 1:
            raise RuntimeError("transform failed")
        return value

    cursor = (
        resource_stream([0, 1, 2], closed)
        .parallel_map(transform, workers=2, buffer_size=0)
        .cursor()
    )

    assert next(cursor) == 0
    with pytest.raises(RuntimeError, match="transform failed"):
        next(cursor)
    assert cursor.closed
    assert closed == [True]


def test_prefetch_propagates_failures_and_closes_upstream():
    closed = []

    def factory():
        try:
            yield 1
            raise RuntimeError("source failed")
        finally:
            closed.append(True)

    cursor = Dataset.from_factory(factory).prefetch(1).cursor()

    assert next(cursor) == 1
    with pytest.raises(RuntimeError, match="source failed"):
        next(cursor)
    assert cursor.closed
    assert closed == [True]


def test_prefetch_closes_upstream_when_stopped_or_abandoned():
    closed = []
    third_read = Event()

    def factory():
        try:
            yield 0
            yield 1
            third_read.set()
            yield 2
        finally:
            closed.append(True)

    cursor = Dataset.from_factory(factory).prefetch(1).cursor()
    assert next(cursor) == 0
    assert third_read.wait(2)
    cursor.close()
    assert closed == [True]

    cursor = resource_stream(range(100), closed).prefetch(1).cursor()
    assert next(cursor) == 0
    del cursor
    gc.collect()
    assert closed == [True, True]


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
    assert not cursor.checkpointable
    with pytest.raises(TypeError, match="not checkpointable"):
        cursor.state_dict()
    restored = stream.cursor()
    with pytest.raises(TypeError, match="not checkpointable"):
        restored.load_state_dict({"position": 1})


def test_read_ahead_operators_checkpoint_consumer_progress():
    source = IndexedDataset.from_source(range(100)).filter(lambda value: value % 2 == 0)
    cases = [
        (
            source.parallel_map(str, workers=2, buffer_size=2),
            list(map(str, range(0, 70, 2))),
        ),
        (source.prefetch(4), list(range(0, 70, 2))),
    ]

    for dataset, prefix in cases:
        assert dataset.checkpointable
        cursor = dataset.cursor()
        assert cursor.checkpointable
        assert [next(cursor) for _ in range(35)] == prefix
        state = cursor.state_dict()
        expected = list(cursor)

        assert state["state"]["replay"] == 35
        resumed = dataset.cursor()
        resumed.load_state_dict(state)
        assert list(resumed) == expected


def test_read_ahead_operators_take_periodic_parent_snapshots():
    source = IndexedDataset.from_source(range(100))
    datasets = [
        source.parallel_map(str, workers=2, buffer_size=2),
        source.prefetch(4),
    ]

    for dataset in datasets:
        cursor = dataset.cursor()
        assert len([next(cursor) for _ in range(70)]) == 70
        state = cursor.state_dict()
        expected = list(cursor)

        assert state["state"]["replay"] == 6
        resumed = dataset.cursor()
        resumed.load_state_dict(state)
        assert list(resumed) == expected


def test_nested_read_ahead_operators_resume_together():
    dataset = (
        IndexedDataset.from_source(range(200))
        .filter(lambda value: value % 3 == 0)
        .parallel_map(lambda value: value * 10, workers=2, buffer_size=2, name="scale")
        .prefetch(4)
    )
    cursor = dataset.cursor()
    assert len([next(cursor) for _ in range(65)]) == 65
    state = cursor.state_dict()
    expected = list(cursor)

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [1950, 1980]


def test_read_ahead_operators_remain_uncheckpointable_with_an_uncheckpointable_parent():
    source = Dataset.from_factory(lambda: range(3))
    datasets = [
        source.parallel_map(str, workers=1),
        source.prefetch(1),
    ]

    for dataset in datasets:
        assert not dataset.checkpointable
        cursor = dataset.cursor()
        assert not cursor.checkpointable
        with pytest.raises(TypeError, match="not checkpointable"):
            cursor.state_dict()
        with pytest.raises(TypeError, match="not checkpointable"):
            cursor.load_state_dict({})
        cursor.close()


def test_cursor_checkpointability_follows_ordinary_parent_graphs():
    checkpointable = IndexedDataset.from_source(range(3)).filter(bool).map(str)
    uncheckpointable = Dataset.from_factory(lambda: range(3)).map(str)
    datasets = [
        uncheckpointable,
        uncheckpointable.take(1),
        uncheckpointable.repeat(2),
        uncheckpointable.concat(checkpointable),
        uncheckpointable.zip(checkpointable),
    ]

    assert checkpointable.cursor().checkpointable
    for dataset in datasets:
        with dataset.cursor() as cursor:
            assert not cursor.checkpointable


def test_custom_dataset_and_cursor_define_a_checkpointable_source():
    class CounterCursor(Cursor[int]):
        def __init__(self, dataset: "CounterDataset"):
            super().__init__(dataset)
            self.dataset = dataset
            self.position = 0

        def _next(self):
            if self.position == self.dataset.stop:
                raise StopIteration
            value = self.position
            self.position += 1
            return value

        def _state_dict(self):
            return {"position": self.position}

        def _load_state_dict(self, state):
            self.position = state["position"]

    class CounterDataset(Dataset[int]):
        supports_checkpointing = True

        def __init__(self, stop):
            self.stop = stop

        @property
        def cardinality(self):
            return Exact(self.stop)

        @property
        def description(self):
            return f"Counter(stop={self.stop})"

        def cursor(self):
            return CounterCursor(self)

    dataset = CounterDataset(4)
    cursor = dataset.cursor()
    assert [next(cursor), next(cursor)] == [0, 1]
    state = cursor.state_dict()

    assert dataset.checkpointable
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == [2, 3]
    assert dataset.explain() == "Counter(stop=4) [Exact(4)]"

    different = CounterDataset(5).cursor()
    with pytest.raises(ValueError, match="checkpoint does not match"):
        different.load_state_dict(state)

    class UncheckpointableCounterDataset(CounterDataset):
        supports_checkpointing = False

    uncheckpointable = UncheckpointableCounterDataset(4).cursor()
    assert not uncheckpointable.checkpointable
    with pytest.raises(TypeError, match="not checkpointable"):
        uncheckpointable.state_dict()


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


def test_repeat_cursor_resumes_across_epochs():
    dataset = IndexedDataset.from_source(range(5)).repeat(None, shuffle=True)
    cursor = dataset.cursor()
    assert [next(cursor) for _ in range(7)]
    state = cursor.state_dict()
    expected = [next(cursor) for _ in range(8)]
    cursor.close()

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert [next(resumed) for _ in range(8)] == expected
    resumed.close()


def test_repeat_closes_each_parent_cursor():
    closed = []
    repeated = resource_stream([1], closed).repeat(2)

    assert list(repeated) == [1, 1]
    assert closed == [True, True]

    cursor = resource_stream([1, 2], closed).repeat(None).cursor()
    assert next(cursor) == 1
    cursor.close()
    assert closed == [True, True, True]


def test_checkpoint_rejects_a_different_cursor_configuration():
    source = IndexedDataset.from_source(range(8)).filter(bool, name="truthy")
    datasets = [
        (source.map(str, name="string"), source.map(float, name="float")),
        (
            source.parallel_map(str, workers=1, name="string"),
            source.parallel_map(str, workers=2, name="string"),
        ),
        (source.prefetch(1), source.prefetch(2)),
        (source.take(4), source.take(5)),
        (source.skip(4), source.skip(5)),
        (source.batch(2), source.batch(3)),
        (source.zip(source), source.zip(source, strict=True)),
    ]

    for original, different in datasets:
        checkpoint = original.cursor().state_dict()
        resumed = different.cursor()
        with pytest.raises(ValueError, match="checkpoint does not match"):
            resumed.load_state_dict(checkpoint)
        assert resumed.closed


def test_checkpoint_rejects_an_unsupported_version():
    dataset = IndexedDataset.from_source(range(3))
    checkpoint = dataset.cursor().state_dict()
    checkpoint["version"] = 2

    resumed = dataset.cursor()
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        resumed.load_state_dict(checkpoint)
    assert resumed.closed


def test_flat_map_checkpoint_tracks_current_input_and_child_offset():
    dataset = IndexedDataset.from_source([1, 2, 3]).flat_map(range, name="expand")
    cursor = dataset.cursor()

    assert next(cursor) == 0
    assert next(cursor) == 0
    state = cursor.state_dict()
    assert state["state"]["current"] == 2
    assert state["state"]["child_offset"] == 1

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
