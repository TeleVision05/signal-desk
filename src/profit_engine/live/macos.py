"""launchd integration: write, load and remove the two Signal Desk Live agents.

* ``com.signaldesk.live`` runs ``signal-desk-live run`` (the paper-trading loop);
* ``com.signaldesk.dashboard`` runs the localhost dashboard on 127.0.0.1:8765.

Both are per-user LaunchAgents (``~/Library/LaunchAgents``), loaded with
``launchctl bootstrap gui/<uid>``: no sudo, nothing system-wide. ``RunAtLoad`` starts them at
login, ``KeepAlive`` restarts them if they exit, ``ThrottleInterval`` spaces restarts.

``--stay-awake-on-ac`` wraps the runner in ``/usr/bin/caffeinate -s``, which keeps the Mac
from SLEEPING only while it is on AC power (on battery, or with the lid closed and no
external display, macOS can still sleep). Nothing runs while the Mac sleeps; on wake the
runner acts on the newest closed bar only.

Nothing here runs unless the user invokes ``install``/``uninstall``; ``--dry-run`` prints
the files and commands instead.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from profit_engine.live import paths

RUNNER_LABEL = "com.signaldesk.live"
DASHBOARD_LABEL = "com.signaldesk.dashboard"
CAFFEINATE = "/usr/bin/caffeinate"
# Copied into the agents' environment when set at install time. DATABASE_URL is deliberately
# never forwarded: the live store is local SQLite only.
FORWARDED_ENV = (
    "SIGNAL_DESK_HOME",
    "SIGNAL_DESK_LIVE_DB",
    "SIGNAL_DESK_BOOKS",
    "SIGNAL_DESK_MODELS_DIR",
    "SIGNAL_DESK_LOGS",
    "SIGNAL_DESK_NOTIFY",
)
NOT_PATHS = {"SIGNAL_DESK_NOTIFY"}


def _forwarded(name: str, value: str) -> str:
    """Path variables become absolute: the agents run in the repo, the CLI in the user's cwd,
    and both must open the same files. A URL is passed through (``paths.db_path`` refuses it)."""
    if name in NOT_PATHS or "://" in value:
        return value
    return str(Path(value).expanduser().resolve())


@dataclass(frozen=True)
class InstallPlan:
    python: Path
    repo: Path
    home: Path
    logs: Path
    stay_awake_on_ac: bool = False
    dashboard: bool = True
    env: dict = field(default_factory=dict)

    @property
    def agents_dir(self) -> Path:
        return self.home / "Library" / "LaunchAgents"


def default_plan(stay_awake_on_ac: bool = False, dashboard: bool = True) -> InstallPlan:
    """Plan for this checkout and this interpreter (the venv python, symlink kept as is)."""
    return InstallPlan(
        python=Path(sys.executable).absolute(),
        repo=paths.REPO_ROOT,
        home=paths.home(),
        logs=paths.logs_dir(),
        stay_awake_on_ac=stay_awake_on_ac,
        dashboard=dashboard,
        env={k: _forwarded(k, os.environ[k]) for k in FORWARDED_ENV if os.environ.get(k)},
    )


def _agent(plan: InstallPlan, label: str, command: list[str], prefix: list[str], log_name: str) -> dict:
    return {
        "Label": label,
        "ProgramArguments": [*prefix, str(plan.python), "-m", "profit_engine.live.cli", *command],
        "WorkingDirectory": str(plan.repo),
        "EnvironmentVariables": {"PYTHONPATH": str(plan.repo / "src"), "PYTHONUNBUFFERED": "1", **plan.env},
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(plan.logs / f"{log_name}.out.log"),
        "StandardErrorPath": str(plan.logs / f"{log_name}.err.log"),
    }


def runner_plist(plan: InstallPlan) -> dict:
    prefix = [CAFFEINATE, "-s"] if plan.stay_awake_on_ac else []
    # The JSON log already goes to the rotated runner.jsonl; stderr (never rotated) keeps crashes only.
    return _agent(plan, RUNNER_LABEL, ["run", "--no-console-log"], prefix, "runner")


def dashboard_plist(plan: InstallPlan) -> dict:
    return _agent(plan, DASHBOARD_LABEL, ["dashboard"], [], "dashboard")


def agents(plan: InstallPlan) -> list[tuple[str, dict]]:
    out = [(RUNNER_LABEL, runner_plist(plan))]
    if plan.dashboard:
        out.append((DASHBOARD_LABEL, dashboard_plist(plan)))
    return out


def render(plist: dict) -> bytes:
    return plistlib.dumps(plist, sort_keys=True)


Runner = Callable[[list[str]], int]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def _retire(label: str, plan: InstallPlan, dry_run: bool, run: Runner, out: Callable[[str], None], uid: int) -> None:
    """Stop an agent and delete its plist (a no-op if it was never installed)."""
    path = plan.agents_dir / f"{label}.plist"
    bootout = ["launchctl", "bootout", f"gui/{uid}/{label}"]
    if dry_run:
        out(f"Would run: {shlex.join(bootout)}   (ignored if not loaded)")
        out(f"Would delete {path}")
        return
    run(bootout)
    if path.exists():
        path.unlink()
        out(f"Removed {label}.")


def install(
    plan: InstallPlan, dry_run: bool = False, run: Runner = _run, out: Callable[[str], None] = print,
    uid: int | None = None,
) -> int:  # fmt: skip
    """Write the plists and (re)load them. Returns a process exit code."""
    uid = os.getuid() if uid is None else uid
    failed = 0
    for label, plist in agents(plan):
        path = plan.agents_dir / f"{label}.plist"
        bootout = ["launchctl", "bootout", f"gui/{uid}/{label}"]
        bootstrap = ["launchctl", "bootstrap", f"gui/{uid}", str(path)]
        if dry_run:
            out(f"Would write {path}:\n{render(plist).decode()}")
            out(f"Would run: {shlex.join(bootout)}   (ignored if not loaded)")
            out(f"Would run: {shlex.join(bootstrap)}")
            continue
        plan.logs.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render(plist))
        run(bootout)  # unload an older copy first; fails harmlessly if none is loaded
        if run(bootstrap) != 0:
            out(f"Could not start {label}. Check {plan.logs} and try `signal-desk-live install` again.")
            failed += 1
        else:
            out(f"Installed and started {label} ({path}).")
    if not plan.dashboard:  # runner only: stop a dashboard agent left by an earlier full install
        _retire(DASHBOARD_LABEL, plan, dry_run, run, out, uid)
    return 1 if failed else 0


def uninstall(
    plan: InstallPlan, dry_run: bool = False, run: Runner = _run, out: Callable[[str], None] = print,
    uid: int | None = None,
) -> int:  # fmt: skip
    """Stop both agents and delete their plists (data and logs are kept)."""
    uid = os.getuid() if uid is None else uid
    for label in (RUNNER_LABEL, DASHBOARD_LABEL):
        _retire(label, plan, dry_run, run, out, uid)
    return 0
