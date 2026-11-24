# Contributor instructions

This is an independent NavGPT-3 derivative. Do not edit or manage processes owned by
neighboring repositories. Preserve inherited licenses and `provenance.json`.

- One kernel agent loop and one tool implementation per tool set (`point_v1`, `vla_v1`). Add model
  profiles instead of model-name branches. Keep actor and Planner adapters separate from simulator
  state and run scheduling.
- Never automatically retry movement. An uncertain episode ends as `failed_or_uncertain`; keep its
  evidence and never replay it.
- Never send goals, reference paths or evaluation metrics to a model or actor. Collection runs need a
  train dataset and downstream review; evaluation data cannot become training input.
- All service control requires generated ownership identities. Browser input must not become
  arbitrary executable argv. Credentials stay outside specs and records (env-var references).
- Records are create-only; never overwrite a completed episode. Runs freeze the spec and kernel digest.
- Add migrations rather than altering applied migration files.
- Run `uv run pytest -q`, `uv run ruff check open_nav_harness`, `uv run mypy open_nav_harness` and
  `npm --prefix frontend run build`; CI (`.github/workflows/checks.yml`) also runs the Playwright
  tests (`npm --prefix frontend test`), so use them for changed user workflows.
- Tests must not start a real benchmark or depend on a live GPU; use fakes.
- Semantic-map servers and deep_search belong to the neighboring sementic_map_mcp project. Keep only
  generic MCP client integration here; do not import map packages or implement map search/model loops.
