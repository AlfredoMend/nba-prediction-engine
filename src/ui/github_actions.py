"""
Trigger and monitor the daily GitHub Actions workflow from the Streamlit app.

Needs a GitHub token in Streamlit secrets (`GITHUB_TOKEN`): a fine-grained
personal access token limited to this repository with "Actions: Read and write".
The token stays server-side; it is never sent to the browser.
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

API = "https://api.github.com"
WORKFLOW = "daily_update.yml"
COOLDOWN_MIN = 15      # refuse a new full run if one started this recently


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def latest_run(token: str, repo: str) -> dict | None:
    """Most recent run of the workflow: status, conclusion, url, minutes since it started."""
    r = requests.get(f"{API}/repos/{repo}/actions/workflows/{WORKFLOW}/runs", params={"per_page": 1},
                     headers=_headers(token), timeout=15)
    r.raise_for_status()
    runs = r.json().get("workflow_runs", [])
    if not runs:
        return None
    run = runs[0]
    started = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
    return {"status": run["status"], "conclusion": run.get("conclusion"), "url": run["html_url"],
            "event": run["event"], "started": started,
            "minutes_ago": (datetime.now(timezone.utc) - started).total_seconds() / 60}


def can_trigger(run: dict | None) -> tuple[bool, str]:
    if run is None:
        return True, ""
    if run["status"] in ("queued", "in_progress", "waiting", "requested", "pending"):
        return False, "A run is already in progress."
    if run["minutes_ago"] < COOLDOWN_MIN:
        return False, f"Last run started {run['minutes_ago']:.0f} min ago; wait {COOLDOWN_MIN} min between runs."
    return True, ""


def trigger_run(token: str, repo: str, ref: str = "main") -> None:
    r = requests.post(f"{API}/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches", json={"ref": ref},
                      headers=_headers(token), timeout=15)
    if r.status_code != 204:
        raise RuntimeError(f"GitHub returned {r.status_code}: {r.text[:200]}")
