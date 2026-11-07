# Contributor instructions

This is an independent NavGPT-3 derivative. Do not edit or manage processes owned by
neighboring repositories. Preserve inherited licenses and `provenance.json`.

- One navigation tool implementation: `NavigationSession`; one API model/tool loop.
  Add model profiles instead of model-name branches. Keep actor and Planner adapters
  separate from environment execution and queue orchestration.
- Never automatically retry movement. Preserve uncertain leases and attempt evidence.
- Never send goals, reference paths or evaluation metrics to a model. Collection
  exports require downstream review; evaluation data cannot become training input.
- All service control requires generated ownership identities. Browser input must not
  become arbitrary executable argv. Credentials belong outside experiment snapshots.
- Freeze configuration and source provenance. Never overwrite a completed attempt.
- Add migrations rather than altering applied migration files.
- Run `uv run pytest -q`, `uv run ruff check nav_harness`, `uv run mypy nav_harness`,
  `uv run ruff check --select F821,F823 navgpt`, and `npm --prefix frontend run build`.
  Use Playwright for changed user workflows.
- GPU integration uses a separate deployment and training/dev manifest. Never start
  the original formal benchmark as a side effect of a unit/UI test.
