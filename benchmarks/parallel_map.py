import argparse
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import median
from time import perf_counter, sleep
from typing import Literal

from rillway import Dataset, IndexedDataset

type Backend = Literal["thread", "process"]


@dataclass(frozen=True)
class Workload:
    name: str
    function: Callable[[int], int]
    count: int


def identity(value: int) -> int:
    return value


def wait_5ms(value: int) -> int:
    sleep(0.005)
    return value


def skewed_wait(value: int) -> int:
    sleep(0.04 if value % 16 == 0 else 0.001)
    return value


def python_cpu(value: int) -> int:
    state = value
    for index in range(120_000):
        state = (state * 1_664_525 + index + 1_013_904_223) & 0xFFFFFFFF
    return state


def slow_source() -> Iterable[int]:
    for value in range(32):
        sleep(0.003)
        yield value


def elapsed(dataset: Dataset[int]) -> tuple[float, int]:
    started = perf_counter()
    checksum = sum(dataset)
    return perf_counter() - started, checksum


def repeated(dataset: Dataset[int], repeats: int) -> tuple[float, int]:
    runs = [elapsed(dataset) for _ in range(repeats)]
    checksums = {checksum for _, checksum in runs}
    if len(checksums) != 1:
        raise RuntimeError("benchmark produced inconsistent results")
    return median(duration for duration, _ in runs), checksums.pop()


def print_scaling(workloads: list[Workload], workers: list[int], repeats: int) -> None:
    print("\nScaling and overhead")
    print("workload  backend  workers  items/s  speedup")
    for workload in workloads:
        source = IndexedDataset.from_source(range(workload.count))
        serial_time, expected = repeated(source.map(workload.function), repeats)
        print(
            f"{workload.name:9} {'map':8} {1:7} "
            f"{workload.count / serial_time:8.0f} {1.0:7.2f}x"
        )
        for backend in ("thread", "process"):
            for worker_count in workers:
                dataset = source.parallel_map(
                    workload.function,
                    workers=worker_count,
                    backend=backend,
                )
                duration, checksum = repeated(dataset, repeats)
                if checksum != expected:
                    raise RuntimeError(f"{backend} backend produced an incorrect result")
                print(
                    f"{workload.name:9} {backend:8} {worker_count:7} "
                    f"{workload.count / duration:8.0f} {serial_time / duration:7.2f}x"
                )


def print_buffering(repeats: int) -> None:
    print("\nBuffer size with four thread workers")
    print("workload  buffer  items/s")
    source = IndexedDataset.from_source(range(128))
    for name, function in (("uniform", wait_5ms), ("skewed", skewed_wait)):
        for buffer_size in (0, 4, 16):
            dataset = source.parallel_map(
                function,
                workers=4,
                buffer_size=buffer_size,
            )
            duration, _ = repeated(dataset, repeats)
            print(f"{name:9} {buffer_size:6} {128 / duration:8.0f}")


def latency(dataset: Dataset[int]) -> tuple[float, float]:
    started = perf_counter()
    cursor = dataset.cursor()
    try:
        next(cursor)
        first = perf_counter() - started
        sum(cursor)
        return first, perf_counter() - started
    finally:
        cursor.close()


def print_slow_source(repeats: int) -> None:
    print("\nSlow upstream source")
    print("mode      buffer  first ms  total ms")
    source = Dataset.from_factory(slow_source)
    configurations: list[tuple[str, Dataset[int], int | None]] = [
        ("map", source.map(wait_5ms), None),
        *[
            (
                "thread",
                source.parallel_map(wait_5ms, workers=4, buffer_size=buffer_size),
                buffer_size,
            )
            for buffer_size in (0, 4, 16)
        ],
    ]
    for mode, dataset, buffer_size in configurations:
        runs = [latency(dataset) for _ in range(repeats)]
        first = median(result[0] for result in runs) * 1_000
        total = median(result[1] for result in runs) * 1_000
        label = "-" if buffer_size is None else str(buffer_size)
        print(f"{mode:9} {label:>6} {first:9.1f} {total:9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if args.repeats < 1 or any(worker < 1 for worker in args.workers):
        parser.error("repeats and workers must be positive")

    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python: {sys.version.split()[0]} (GIL {'enabled' if gil_enabled else 'disabled'})")
    print(f"Logical CPUs: {os.cpu_count()}")
    print_scaling(
        [
            Workload("identity", identity, 10_000),
            Workload("wait-5ms", wait_5ms, 128),
            Workload("cpu", python_cpu, 64),
        ],
        args.workers,
        args.repeats,
    )
    print_buffering(args.repeats)
    print_slow_source(args.repeats)


if __name__ == "__main__":
    main()
