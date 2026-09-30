"""Contracts for the safe and documented development container."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
JUPYTER_SCRIPT = ROOT / "scripts" / "start_jupyter.sh"
GUIDE = ROOT / "docs" / "DOCKER_SETUP.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")


def test_dockerfile_has_no_persistent_jupyter_security_override() -> None:
    source = DOCKERFILE.read_text()

    for forbidden in (
        "jupyter_server_config",
        "ServerApp.ip",
        "ServerApp.token",
        "ServerApp.password",
        "ServerApp.disable_check_xsrf",
        "ServerApp.allow_origin",
        "ServerApp.allow_root",
    ):
        assert forbidden not in source
    assert "USER devuser" in source
    assert not any(line.endswith(chr(92) * 2) for line in source.splitlines())
    assert (
        'python -m pip install --no-cache-dir -e ".[dev,io,plot]" jupyter ipykernel'
        in source
    )


def test_devcontainer_uses_repository_dockerfile_nonroot_user_and_forwarded_port() -> (
    None
):
    config = json.loads(DEVCONTAINER.read_text())

    assert config["build"] == {"context": "..", "dockerfile": "../Dockerfile"}
    assert config["workspaceFolder"] == "/workspace"
    assert config["remoteUser"] == "devuser"
    assert config["forwardPorts"] == [8888]
    assert config["portsAttributes"]["8888"]["onAutoForward"] == "notify"


def test_jupyter_launcher_is_valid_shell_and_keeps_security_owned_by_jupyter() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(JUPYTER_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source = JUPYTER_SCRIPT.read_text()
    assert "RVE_JUPYTER_HOST:-127.0.0.1" in source
    assert "authentication and XSRF protection remain enabled" in source
    for forbidden in ("token=''", "password=''", "disable_check_xsrf"):
        assert forbidden not in source


def test_container_guide_matches_current_commands_and_discloses_unrun_build() -> None:
    text = GUIDE.read_text()

    assert "`.[dev,io,plot]`".replace("`", chr(96)) in text
    assert "`127.0.0.1`".replace("`", chr(96)) in text
    assert "認証と XSRF は無効にならない" in text
    assert "Docker CLI/\ndaemon がないため" in text
    for forbidden in (
        "make build",
        "make clean",
        "requirements-dev.txt",
        "Jupyterのトークン認証を無効化",
        "root権限でのJupyter実行を許可",
        "sudo chown -R",
    ):
        assert forbidden not in text


def test_container_guide_local_links_resolve() -> None:
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(GUIDE.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://")):
            continue
        if not (GUIDE.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
