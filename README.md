# adaptive-mas-defense

LLM multi-agent security research harness (prompt-infection attack on a
coordinator → outline → researcher → executor pipeline, with a tiered
security observer).

## Run the experiment

`main.py` is the only experiment runner:

```
python main.py
```

It will prompt for a task, run one episode, and write a post-hoc dump to
`outputs/last_run.jsonl` for the static (Tkinter) dashboard.

## Live Streamlit dashboard (read-only)

A live dashboard can watch the SAME run started by `main.py`. It only
reads the event stream; it never starts, restarts or reruns the
experiment.

Terminal 1 — start the dashboard first:

```
streamlit run streamlit_app.py
```

Terminal 2 — run the experiment normally:

```
python main.py
```

The dashboard updates automatically while `main.py` runs and, when the
run finishes, shows the final status and retains the event history.

### How it works

* `main.py` creates a `LiveEventPublisher` (`security/live_events.py`)
  and hands it to `MASEnvironment(event_publisher=...)`.
* Every existing `MASEvent`, every security observation, and the episode
  lifecycle are appended as JSON lines to `outputs/live_events.jsonl`.
  A `outputs/live_run.json` marker records the current run id and status.
* `streamlit_app.py` tails that JSONL from a tracked byte offset
  (`EventReader`): refreshes never duplicate events, a partially written
  record is never parsed, and events from a previous run are separated by
  `run_id` so an old run is never shown as live.

### Configuration

* `MAS_LIVE_EVENTS=0` — disable streaming in `main.py` (default: on).
* `MAS_LIVE_EVENTS_PATH` / `MAS_LIVE_RUN_MARKER` — override the stream
  and marker paths (useful for tests).

## Tests

```
python -m pytest tests/test_live_events.py tests/test_streamlit_dashboard.py
```

The dashboard tests use mocked events; no LLM experiment or Streamlit
server is started.