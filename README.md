# Rillway

[![CI](https://github.com/Tytskiy/rillway/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/rillway/actions/workflows/ci.yml)

Rillway is a small Python library for building lazy, reusable data pipelines.
It keeps the familiar feel of Python iterators while making it easy to replay a
pipeline, split up a dataset, and resume interrupted work.

```python
from rillway import IndexedDataset

numbers = IndexedDataset.from_source(range(10))

dataset = (
    numbers
    .map(lambda value: value * 2)
    .filter(lambda value: value % 3 == 0)
    .batch(2)
)

assert list(dataset) == [(0, 6), (12, 18)]
assert list(dataset) == [(0, 6), (12, 18)]
```

Rillway requires Python 3.12 or newer.

## Installation

Rillway is not published on PyPI. Install it directly from GitHub:

```shell
uv add "rillway @ git+https://github.com/Tytskiy/rillway.git"
```

## The basic idea

A dataset is a reusable description of some work. Creating one does not read
any data. Each time you iterate over it, Rillway creates a fresh cursor and
runs the pipeline from the beginning.

```python
dataset = IndexedDataset.from_source([1, 2, 3]).map(str)

assert list(dataset) == ["1", "2", "3"]
assert list(dataset) == ["1", "2", "3"]
```

A cursor is a single trip through that dataset. Use it directly when you want
to stop early or save your place:

```python
with dataset.cursor() as cursor:
    first = next(cursor)
```

The context manager makes sure anything opened by the pipeline is cleaned up.
Normal completion and errors are cleaned up as well.

## Working with datasets

Start with `IndexedDataset.from_source()` when your data supports `len()` and
indexing, such as a list or `range`.

```python
dataset = IndexedDataset.from_source([10, 20, 30, 40])

assert dataset[1] == 20
assert list(dataset[1:3]) == [20, 30]
assert list(dataset.shard(1, 2)) == [30, 40]
```

Structured files have their own datasets:

```python
from rillway import CsvDataset, HuggingFaceDataset, JsonlDataset, ParquetDataset

events = JsonlDataset("events.jsonl")
people = CsvDataset("people.csv", delimiter=",")
training = ParquetDataset("training.parquet", columns=("text", "label"))
hub = HuggingFaceDataset(
    "cornell-movie-review-data/rotten_tomatoes",
    columns=("text", "label"),
)
```

The readers yield parsed records and resume from saved file positions. CSV
files use their first row as the column names; pass
`columns=(...)` for a headerless file or `encoding=...` for non-UTF-8 text.
They also understand paths backed by fsspec. Use `UPath` when a filesystem
needs configuration, and reuse the configured root across datasets:

```python
from upath import UPath

root = UPath("s3://training-bucket/data", profile="training")

events = JsonlDataset(root / "events.jsonl")
training = ParquetDataset(root / "train" / "*.parquet")
```

Install the backend needed by the URL, such as `s3fs` for S3 or `gcsfs` for
Google Cloud Storage. Plain local paths and URLs that need no additional
settings can be passed directly. Filesystems must provide a stable `ukey()` so
Rillway can detect changed sources when resuming; immutable or versioned URLs
provide the strongest guarantee. Put credentials in `UPath` options rather
than in the URL so they cannot appear in descriptions or checkpoints.

Parquet support is optional; add it with
`uv add "rillway[parquet] @ git+https://github.com/Tytskiy/rillway.git"`.
Hugging Face support is optional; add it with
`uv add "rillway[huggingface] @ git+https://github.com/Tytskiy/rillway.git"`.
It reads the Hub's Parquet exports through `HfFileSystem`, pins every file to a
commit, and provides exact cardinality and range access without downloading
the complete dataset. Values come directly from Parquet; Hugging Face media
decoders are not applied.

Rillway keeps useful abilities when an operation allows it. Mapping an indexed
dataset still gives you an indexed dataset. Filtering may change how many
items remain, so its result behaves like a stream instead.

You can inspect a pipeline without running it:

```python
print(dataset.map(str).filter(str.isdigit).explain())
```

Independent work can run concurrently with `parallel_map`. Results stay in
input order, and the amount of work waiting in memory is bounded.

```python
dataset = numbers.parallel_map(read_and_decode, workers=8)
```

Parallel mapping uses threads by default. For CPU-heavy Python work, use
`backend="process"`. The function, input values, and results must then be
picklable.

Place `prefetch` after the work you want to overlap with the consumer. It runs
the complete pipeline before that point in one background thread and keeps a
small number of ready elements:

```python
dataset = numbers.map(read_and_decode).batch(32).prefetch(2)
```

Parallel mapping and prefetching are checkpointable when their input is. Their
checkpoints follow consumer progress rather than read-ahead progress; restoring
may replay a small bounded number of input elements per asynchronous boundary.

Repeat a dataset when you need more than one pass. Indexed datasets can be
shuffled differently on each pass while remaining reproducible:

```python
training = numbers.repeat(3, shuffle=True)
```

The default seed is `42`; pass `seed=...` when you want another order. Use
`repeat(None)` for an endless dataset and combine it with `take()` when you
want a fixed number of items.

Indexed datasets can also be shuffled directly. This produces a global
permutation while preserving indexing and range access:

```python
shuffled = numbers.shuffle(seed=42)
```

Use bounded shuffle for streamed data. It keeps at most `buffer_size` items
ready and produces the same order again when given the same seed:

```python
shuffled = dataset.shuffle(buffer_size=10_000, seed=42)
```

Shuffle checkpoints keep only their starting position and output count.
Restoring one replays the earlier shuffle work, so restore time grows with the
number of items already consumed rather than with the buffer size.

`interleave` reads datasets in round-robin order, `unbatch` flattens iterable
items, and streamed datasets can be split into round-robin shards. Range-based
datasets keep their contiguous, storage-friendly shards.

`mix` chooses between datasets using deterministic weights. Finite inputs are
all drained, so weights affect their order rather than their final counts. Use
endless repetition when the weights should control a fixed-size training mix:

```python
training = (
    text.repeat(None)
    .mix(code.repeat(None), weights=(0.8, 0.2), seed=42)
    .take(1_000_000)
)
```

## Resuming work

Some datasets can save their current position and continue with a new cursor.
Treat the saved state as opaque and restore it before reading from the new
cursor.

```python
dataset = IndexedDataset.from_source(range(6)).map(lambda value: value * 10)

with dataset.cursor() as cursor:
    assert next(cursor) == 0
    checkpoint = cursor.state_dict()

resumed = dataset.cursor()
resumed.load_state_dict(checkpoint)

assert list(resumed) == [10, 20, 30, 40, 50]
```

Use `dataset.checkpointable` before opening a pipeline, or
`cursor.checkpointable` after opening it. Checkpoints belong to the pipeline
that created them; Rillway rejects a checkpoint when the pipeline does not
match.

Implement a custom checkpointable source as a `Dataset` and `Cursor` pair. The
dataset owns immutable configuration and returns a fresh stateful cursor for
each traversal. The cursor implements `_snapshot()` and `_restore()` for its
own state; Rillway adds and validates the public checkpoint envelope. Datasets
are checkpointable by default; set `supports_checkpointing = False` when a
custom source cannot restore its cursor.

## Available operations

- `map`
- `parallel_map`
- `prefetch`
- global indexed and bounded stream `shuffle`
- `filter`
- `flat_map`
- `take` and `skip`
- `batch` and `unbatch`
- `repeat`, with deterministic shuffling for indexed datasets
- `concat`, `interleave`, and weighted `mix`
- `zip`
- `shard` for streamed and range-based datasets
- indexing and slicing for indexed datasets

## Development

```shell
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy src/rillway
```
