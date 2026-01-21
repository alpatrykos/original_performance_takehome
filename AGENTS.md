# Repository Guidelines

## Project Structure & Module Organization

This is a small Python codebase for a performance-tuning challenge.

- `perf_takehome.py` holds the primary kernel builder and test harness you should optimize.
- `problem.py` defines the simulator, ISA, and reference kernel logic.
- `tests/submission_tests.py` is the official correctness/speed test entrypoint.
- `watch_trace.py` and `watch_trace.html` support live trace visualization.
- Generated artifacts: `trace.json` (created when running trace tests).

## Build, Test, and Development Commands

- `python perf_takehome.py` runs the local unit tests in `perf_takehome.py`.
- `python perf_takehome.py Tests.test_kernel_cycles` runs the cycle-count benchmark.
- `python perf_takehome.py Tests.test_kernel_trace` generates `trace.json` for visualization.
- `python tests/submission_tests.py` runs the submission-style correctness and speed checks.
- `python watch_trace.py` serves `watch_trace.html` on `http://localhost:8000` for Perfetto.

## Coding Style & Naming Conventions

- Language: Python 3. Keep to existing style (PEP 8-ish).
- Indentation: 4 spaces; avoid tabs.
- Naming: `snake_case` for functions/variables, `CapWords` for classes, `UPPER_CASE` for constants.
- Keep core logic in `KernelBuilder.build_kernel`; avoid scattering performance-critical changes.
- No formatter or linter is configured; keep diffs tidy and consistent with surrounding code.

## Testing Guidelines

- Tests use Python’s built-in `unittest` framework.
- Correctness is validated by `tests/submission_tests.py`; speed is measured by cycle counts.
- Naming: tests follow `test_*` methods inside `unittest.TestCase` classes.
- For trace debugging, run the trace test then open the hot-reload UI via `watch_trace.py`.

## Commit & Pull Request Guidelines

- History is minimal (single initial commit), so no established commit message convention.
- Use clear, imperative commit messages (e.g., `Optimize hash stage scheduling`).
- PRs should include: a brief performance summary (cycles before/after), test commands run, and any trace screenshots if relevant.

## Security & Configuration Notes

- This repo does not ship secrets or config files.
- If you generate `trace.json`, avoid committing it; treat it as a local artifact.
