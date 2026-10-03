# Rillway agent guide

Rillway is a typed Python library for immutable, replayable dataset pipelines.

## Development

- Use Python 3.12+ and `uv`; do not install dependencies with `pip`.
- Install the locked environment with `uv sync --locked`.
- Run tests with `uv run pytest -q`.
- Run linting with `uv run ruff check .`.
- Run type checking with `uv run mypy src/rillway`.
- Before finishing a code change, run all three checks and report their results.

## Structure

- `src/rillway/__init__.py` defines the public API through explicit re-exports.
- `src/rillway/dataset.py` contains dataset types and implementations.
- `src/rillway/cardinality.py` contains cardinality types and calculations.
- `src/rillway/cursor.py` contains cursor lifecycle and checkpoint behavior.
- `tests/` mirrors observable behavior of the public `rillway` API.

## Code style

- Use PEP 695 generics instead of `TypeVar`.
- Use dataclasses for data and plain classes for bases with no fields.
- Use `ClassVar` for class-level constants on dataclasses.
- Add `from __future__ import annotations` only when forward references require it.
- Do not add comments or docstrings that merely narrate clear code. Explain only
  non-obvious reasons, constraints, or trade-offs.

## Working approach

- Trace callers and identify the broken invariant before changing shared behavior.
  Fix the failure class with the smallest structural change at the shared boundary;
  do not scatter symptom guards or reminders across callers.
- Reuse existing code and dependencies before adding abstractions or packages.
- Keep each rule, configuration value, and process in one canonical location;
  reference it elsewhere instead of duplicating it.
- Understand the relevant rationale and trade-offs before structural changes, and
  update affected documentation with the code.
- Preserve unrelated working-tree changes.
- Keep work within the user's requested scope; name relevant non-goals instead of
  quietly expanding it.
- Keep each iteration one coherent transformation. If repeated attempts make no
  progress, reconsider the approach instead of layering on patches.
- Ask before interacting with remotes, publishing releases, or changing repository
  metadata.

## Design constraints

- Enforce contracts and boundaries in code while keeping execution strategy flexible.
- Keep dataset graphs immutable and replayable; cursors are stateful and one-shot.
- Preserve additional access capabilities only when an operation can support them; otherwise return a `Dataset`.
- Keep cardinality propagation accurate without traversing input data.
- Ensure cursor exhaustion, failure, explicit close, and abandonment release owned resources.
- Treat checkpoint state as opaque and reject state that does not match the dataset graph.
- Export public names through `rillway.__all__`; keep implementation-only names private.
- Add or update a focused test for every behavior change or bug fix.

## Evidence

- Verify current APIs and observable runtime behavior rather than relying on memory.
  Distinguish observations, assumptions, and missing information.
- Prefer deterministic checks. Use independent review when the change's risk or
  ambiguity justifies it, and preserve its actual findings.
- Report the exact checks run and their outcomes.
- If a relevant check cannot run, report it as `NOT_RUN` with the reason.
- Treat verification as stale after further edits and rerun the affected checks.
