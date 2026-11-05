"""launchd plists: snapshot, caffeinate wrapper, and install/uninstall that only act when asked."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from profit_engine.live import macos

SNAPSHOTS = Path(__file__).parent / "snapshots"


def _plan(tmp: Path | None = None, **kw) -> macos.InstallPlan:
    home = tmp or Path("/Users/demo")
    return macos.InstallPlan(
        python=Path("/opt/signal-desk/.venv/bin/python"),
        repo=Path("/opt/signal-desk"),
        home=home,
        logs=home / "Library" / "Logs" / "SignalDesk",
        **kw,
    )


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("com.signaldesk.live.plist", lambda: macos.runner_plist(_plan())),
        ("com.signaldesk.live.stay-awake.plist", lambda: macos.runner_plist(_plan(stay_awake_on_ac=True))),
        ("com.signaldesk.dashboard.plist", lambda: macos.dashboard_plist(_plan())),
    ],
)
def test_plist_snapshot(name, build):
    assert macos.render(build()).decode() == (SNAPSHOTS / name).read_text()


def test_runner_plist_contract():
    p = macos.runner_plist(_plan(stay_awake_on_ac=True))
    assert p["ProgramArguments"][:2] == ["/usr/bin/caffeinate", "-s"]
    assert p["ProgramArguments"][2] == "/opt/signal-desk/.venv/bin/python"
    assert p["RunAtLoad"] is True and p["KeepAlive"] is True and p["ThrottleInterval"] == 30
    assert p["StandardOutPath"].startswith("/Users/demo/Library/Logs/SignalDesk/")
    assert p["WorkingDirectory"] == "/opt/signal-desk"
    assert "DATABASE_URL" not in p["EnvironmentVariables"]
    assert macos.runner_plist(_plan())["ProgramArguments"][0] == "/opt/signal-desk/.venv/bin/python"


def test_dry_run_writes_and_runs_nothing(tmp_path):
    ran, said = [], []
    plan = _plan(tmp_path)
    assert macos.install(plan, dry_run=True, run=ran.append, out=said.append, uid=501) == 0
    assert macos.uninstall(plan, dry_run=True, run=ran.append, out=said.append, uid=501) == 0
    assert ran == []
    assert not plan.agents_dir.exists()
    text = "\n".join(said)
    assert "launchctl bootstrap gui/501" in text and "com.signaldesk.live" in text


def test_install_then_uninstall(tmp_path):
    ran: list[list[str]] = []
    plan = _plan(tmp_path)
    assert macos.install(plan, run=lambda cmd: ran.append(cmd) or 0, out=lambda _s: None, uid=501) == 0
    runner = plan.agents_dir / "com.signaldesk.live.plist"
    assert plistlib.loads(runner.read_bytes())["Label"] == "com.signaldesk.live"
    assert (plan.agents_dir / "com.signaldesk.dashboard.plist").exists()
    assert ["launchctl", "bootstrap", "gui/501", str(runner)] in ran
    ran.clear()
    macos.uninstall(plan, run=lambda cmd: ran.append(cmd) or 0, out=lambda _s: None, uid=501)
    assert ["launchctl", "bootout", "gui/501/com.signaldesk.live"] in ran
    assert not runner.exists()
