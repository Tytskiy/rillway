import gc
import os
from threading import Event, Lock

import pytest

from rillway import Dataset, Exact, IndexedDataset, RangeDataset


def worker_process_id(value):
    return os.getpid(), value


def resource_stream(values, closed):
    def factory():
        try:
            yield from values
        finally:
            closed.append(True)

    return Dataset.from_factory(factory)


def test_parallel_map_runs_concurrently_and_preserves_input_order():
    second_finished = Event()
    completed = []
    lock = Lock()

    def transform(value):
        if value == 0:
            assert second_finished.wait(2)
        with lock:
            completed.append(value)
        if value == 1:
            second_finished.set()
        return value * 10

    dataset = IndexedDataset.from_source(range(3)).parallel_map(
        transform,
        workers=2,
        buffer_size=0,
    )

    assert isinstance(dataset, Dataset)
    assert not isinstance(dataset, RangeDataset)
    assert dataset.cardinality == Exact(3)
    assert list(dataset) == [0, 10, 20]
    assert completed[:2] == [1, 0]
    assert "workers=2, buffer_size=0" in dataset.explain()


def test_parallel_map_can_use_process_workers():
    parent_process = os.getpid()
    dataset = IndexedDataset.from_source(range(3)).parallel_map(
        worker_process_id,
        workers=2,
        buffer_size=0,
        backend="process",
    )

    results = list(dataset)

    assert [value for _, value in results] == [0, 1, 2]
    assert all(process != parent_process for process, _ in results)


def test_prefetch_runs_its_parent_in_the_background_and_preserves_order():
    second_read = Event()

    def factory():
        yield 0
        second_read.set()
        yield 1

    dataset = Dataset.from_factory(factory, cardinality=Exact(2)).prefetch(1)
    cursor = dataset.cursor()

    assert not second_read.is_set()
    assert next(cursor) == 0
    assert second_read.wait(2)
    assert list(cursor) == [1]
    assert dataset.cardinality == Exact(2)
    assert dataset.explain().startswith("Prefetch(buffer_size=1)")


@pytest.mark.parametrize(
    "options",
    [
        {"workers": 0},
        {"workers": 1, "buffer_size": -1},
        {"workers": 1, "backend": "invalid"},
    ],
)
def test_parallel_map_validates_limits(options):
    with pytest.raises(ValueError):
        IndexedDataset.from_source([1]).parallel_map(str, **options)


def test_prefetch_requires_a_positive_buffer_size():
    with pytest.raises(ValueError, match="positive"):
        IndexedDataset.from_source([1]).prefetch(0)


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
