import sys

import pytest

from rillway import IndexedDataset, ProfileNode, ProfileReport, profiling


def _nodes(report):
    return {node.description: node for node in report.nodes}


def test_profile_requires_its_context_manager():
    profile = profiling(IndexedDataset.from_source(range(3)))

    with pytest.raises(RuntimeError, match="context manager"):
        profile.report()


def test_profile_attributes_indexed_operations():
    dataset = (
        IndexedDataset.from_source(range(6))
        .map(lambda value: value * 2, name="double")
        .batch(2)
    )

    with profiling(dataset) as profile:
        assert list(dataset) == [(0, 2), (4, 6), (8, 10)]

    report = profile.report()
    nodes = _nodes(report)
    assert report.outputs == 3
    assert nodes["Batch(size=2, drop_last=False)"].outputs == 3
    assert nodes["Map(name='double')"].outputs == 6
    assert nodes["Source(type=range)"].outputs == 6
    assert all(node.wall_seconds >= node.self_seconds >= 0 for node in report.nodes)
    assert all(node.max_seconds >= node.average_seconds >= 0 for node in report.nodes)
    assert "3 outputs" in str(report)
    assert "avg/value" in str(report)
    assert str(profile) == str(report)


def test_profile_attributes_streamed_operations():
    dataset = (
        IndexedDataset.from_source(range(6))
        .map(lambda value: value * 2, name="double")
        .filter(lambda value: value % 4 == 0, name="even")
        .batch(2)
    )

    with profiling(dataset) as profile:
        assert list(dataset) == [(0, 4), (8,)]

    nodes = _nodes(profile.report())
    assert nodes["Batch(size=2, drop_last=False)"].outputs == 2
    assert nodes["Filter(name='even')"].outputs == 3
    assert nodes["Map(name='double')"].outputs == 6
    assert nodes["Source(type=range)"].outputs == 6


def test_profile_records_the_failing_operation_and_releases_monitoring():
    def fail(value):
        if value == 1:
            raise RuntimeError("broken")
        return value

    dataset = IndexedDataset.from_source(range(3)).map(fail, name="fail")
    profile = profiling(dataset)
    with pytest.raises(RuntimeError, match="broken"), profile:
        list(dataset)

    nodes = _nodes(profile.report())
    assert nodes["Map(name='fail')"].outputs == 1
    assert nodes["Map(name='fail')"].failures == 1
    assert nodes["Source(type=range)"].outputs == 2
    assert sys.monitoring.get_tool(sys.monitoring.PROFILER_ID) is None


def test_profile_observes_prefetch_producer_work():
    dataset = (
        IndexedDataset.from_source(range(6))
        .filter(lambda value: True, name="keep")
        .prefetch(2)
    )

    with profiling(dataset) as profile:
        assert list(dataset) == list(range(6))

    nodes = _nodes(profile.report())
    assert nodes["Prefetch(buffer_size=2)"].outputs == 6
    assert nodes["Filter(name='keep')"].outputs == 6
    assert nodes["Source(type=range)"].outputs == 6


def test_profile_table_preserves_small_timings_and_full_descriptions():
    description = "An operation whose complete description matters"
    report = ProfileReport(
        nodes=(
            ProfileNode(
                description=description,
                depth=0,
                calls=1,
                outputs=1,
                failures=0,
                wall_seconds=0.002,
                self_seconds=0.0005,
                average_seconds=0.002,
                max_seconds=0.002,
            ),
        ),
        elapsed_seconds=0.002,
    )

    table = str(report)
    assert description in table
    assert "500.000 µs" in table
    assert "2.000 ms" in table
