"""launchd integration: write, load and remove the Signal Desk Live agents.

* ``com.signaldesk.live`` runs ``signal-desk-live run`` (the paper-trading loop);
* ``com.signaldesk.dashboard`` runs the localhost dashboard on 127.0.0.1:8765;
* ``com.signaldesk.retrain`` runs ``signal-desk-live retrain`` at 06:30 local time on the 1st
  and 2nd of each month (launchd runs a missed time on wake). Retraining is a no-op when the
  month's LightGBM models exist, so the second date only catches a run missed on the first.

All are per-user LaunchAgents (``~/Library/LaunchAgents``), loaded with
``launchctl bootstrap gui/<uid>``: no sudo, nothing system-wide. ``RunAtLoad`` starts them at
login, ``KeepAlive`` restarts them if they exit, ``ThrottleInterval`` spaces restarts.

``--stay-awake-on-ac`` wraps the runner in ``/usr/bin/caffeinate -s``, which keeps the Mac
from SLEEPING only while it is on AC power (on battery, or with the lid closed and no
external display, macOS can still sleep). Nothing runs while the Mac sleeps; on wake the
runner acts on the newest closed bar only.

``install`` checks a few seconds after loading that each agent is actually running
(``launchctl print``): a runner that macOS blocks from reading this folder (``~/Documents``
is privacy-protected) still loads cleanly and would otherwise only crash-loop in silence.
launchctl's own output is kept out of the terminal and goes to ``install.jsonl``.

Nothing here runs unless the user invokes ``install``/``uninstall``; ``--dry-run`` prints
the files and commands instead.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from profit_engine.live import paths

RUNNER_LABEL = "com.signaldesk.live"
DASHBOARD_LABEL = "com.signaldesk.dashboard"
RETRAIN_LABEL = "com.signaldesk.retrain"
RETRAIN_TIMES = ({"Day": 1, "Hour": 6, "Minute": 30}, {"Day": 2, "Hour": 6, "Minute": 30})
CAFFEINATE = "/usr/bin/caffeinate"
# Copied into the agents' environment when set at install time. DATABASE_URL is deliberately
# never forwarded: the live store is local SQLite only.
FORWARDED_ENV = (
    "SIGNAL_DESK_HOME",
    "SIGNAL_DESK_LIVE_DB",
    "SIGNAL_DESK_BOOKS",
    "SIGNAL_DESK_MODELS_DIR",
    "SIGNAL_DESK_DATA_DIR",
    "SIGNAL_DESK_LOGS",
    "SIGNAL_DESK_NOTIFY",
)
NOT_PATHS = {"SIGNAL_DESK_NOTIFY"}
EXIT_TIMEOUT_S = 30  # SIGTERM -> SIGKILL grace: one in-flight Binance request (10 s) plus the tick's commit
STARTUP_CHECK_S = 5.0  # how long after loading an agent install looks at whether it is still running
DOCS = "docs/live_runner.md"
log = logging.getLogger("signal_desk.install")


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
        logs=paths.logs_dir().resolve(),  # launchd needs absolute log paths (and the same folder the env names)
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
        "ExitTimeOut": EXIT_TIMEOUT_S,
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


def retrain_plist(plan: InstallPlan) -> dict:
    """A scheduled job, not a service: it runs, exits, and waits for the next calendar time."""
    plist = _agent(plan, RETRAIN_LABEL, ["retrain"], [], "retrain")
    plist.pop("KeepAlive")
    plist["RunAtLoad"] = False
    plist["StartCalendarInterval"] = [dict(t) for t in RETRAIN_TIMES]
    return plist


def agents(plan: InstallPlan) -> list[tuple[str, dict]]:
    out = [(RUNNER_LABEL, runner_plist(plan))]
    if plan.dashboard:
        out.append((DASHBOARD_LABEL, dashboard_plist(plan)))
    out.append((RETRAIN_LABEL, retrain_plist(plan)))
    return out


def render(plist: dict) -> bytes:
    return plistlib.dumps(plist, sort_keys=True)


Runner = Callable[[list[str]], int]
Probe = Callable[[list[str]], tuple[int, str]]


def _capture(cmd: list[str]) -> tuple[int, str]:
    """Run ``cmd`` with its output captured (logged, never printed raw to the user)."""
    done = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if done.returncode:
        log.warning("command failed", extra={"cmd": cmd, "code": done.returncode, "output": done.stderr.strip()})
    return done.returncode, done.stdout


def _run(cmd: list[str]) -> int:
    return _capture(cmd)[0]


# What install uses unless told otherwise; looked up at call time so tests can replace them.
_launchctl_probe: Probe = _capture
_sleep: Callable[[float], None] = time.sleep


def is_running(printed: str) -> bool:
    """Whether ``launchctl print`` output describes a job that is running now."""
    return re.search(r"^\s*state = running\s*$", printed, re.MULTILINE) is not None


def _check_running(label: str, plan: InstallPlan, uid: int, probe: Probe, out: Callable[[str], None]) -> bool:
    code, printed = probe(["launchctl", "print", f"gui/{uid}/{label}"])
    if code == 0 and is_running(printed):
        out(f"Installed and started {label}; it is running.")
        return True
    name = "runner" if label == RUNNER_LABEL else "dashboard"
    out(
        f"{label} was installed but is not staying up, so it is not doing its job yet. macOS often blocks "
        f"background programs from reading ~/Documents. See 'If the agents do not start' in {DOCS}, and "
        f"check {plan.logs / f'{name}.err.log'}. It stays installed and launchd retries every 30 seconds."
    )
    return False


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
    uid: int | None = None, probe: Probe | None = None, pause: Callable[[float], None] | None = None,
) -> int:  # fmt: skip
    """Write the plists, (re)load them and check they stay running. Returns a process exit code."""
    uid = os.getuid() if uid is None else uid
    probe, pause = probe or _launchctl_probe, pause or _sleep
    failed, loaded = 0, []
    for label, plist in agents(plan):
        path = plan.agents_dir / f"{label}.plist"
        bootout = ["launchctl", "bootout", f"gui/{uid}/{label}"]
        bootstrap = ["launchctl", "bootstrap", f"gui/{uid}", str(path)]
        if dry_run:
            out(f"Would write {path}:\n{render(plist).decode()}")
            out(f"Would run: {shlex.join(bootout)}   (ignored if not loaded)")
            out(f"Would run: {shlex.join(bootstrap)}")
            out(f"Would check it is running: launchctl print gui/{uid}/{label}")
            continue
        plan.logs.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render(plist))
        run(bootout)  # unload an older copy first; fails harmlessly (and quietly) if none is loaded
        if run(bootstrap) != 0:
            out(f"Could not start {label}. Check {plan.logs} and try `signal-desk-live install` again.")
            failed += 1
        else:
            loaded.append(label)
    if RETRAIN_LABEL in loaded:  # scheduled, so it is not running now: nothing to check
        loaded.remove(RETRAIN_LABEL)
        out(f"Scheduled {RETRAIN_LABEL}: retrains the LightGBM models on the 1st and 2nd of each month at 06:30.")
    if loaded:
        pause(STARTUP_CHECK_S)  # a blocked or broken agent exits within a second or two
        failed += sum(not _check_running(label, plan, uid, probe, out) for label in loaded)
    if not plan.dashboard:  # runner only: stop a dashboard agent left by an earlier full install
        _retire(DASHBOARD_LABEL, plan, dry_run, run, out, uid)
    return 1 if failed else 0


def uninstall(
    plan: InstallPlan, dry_run: bool = False, run: Runner = _run, out: Callable[[str], None] = print,
    uid: int | None = None,
) -> int:  # fmt: skip
    """Stop the agents and delete their plists (data, models and logs are kept)."""
    uid = os.getuid() if uid is None else uid
    for label in (RUNNER_LABEL, DASHBOARD_LABEL, RETRAIN_LABEL):
        _retire(label, plan, dry_run, run, out, uid)
    return 0
