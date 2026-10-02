"""Contracts for the safe and documented development container."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
JUPYTER_SCRIPT = ROOT / "scripts" / "start_jupyter.sh"
CONTAINER_SMOKE = ROOT / "scripts" / "smoke_container.sh"
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


def test_docker_build_context_contains_only_declared_package_inputs() -> None:
    active_lines = [
        line
        for raw_line in DOCKERIGNORE.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]

    assert active_lines == [
        "*",
        "!pyproject.toml",
        "!README.md",
        "!src/",
        "!src/**",
        "**/__pycache__/",
        "**/*.pyc",
        "**/*.egg-info/",
    ]


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
    assert "RVE_JUPYTER_ROOT:-${script_project_root}" in source
    assert "RVE_JUPYTER_ROOT must be an absolute path" in source
    assert "authentication and XSRF protection remain enabled" in source
    for forbidden in ("token=''", "password=''", "disable_check_xsrf"):
        assert forbidden not in source


def test_container_smoke_builds_and_checks_nonroot_authenticated_jupyter() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(CONTAINER_SMOKE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source = CONTAINER_SMOKE.read_text()
    for required in (
        "docker info",
        "docker build",
        "id -u",
        "python -c",
        "Authorization: token",
        "docker port",
        "unauthenticated HTTP",
        "::error title=Container smoke failed::stage=${stage}",
        'stage="image build"',
        'stage="authenticated Jupyter readiness"',
        'stage="runtime user validation"',
        "--env RVE_JUPYTER_ROOT=/workspace",
        "dst=/workspace/project,readonly",
        "dst=/workspace/notebooks",
        "--entrypoint /workspace/project/scripts/start_jupyter.sh",
    ):
        assert required in source
    for forbidden in ("exit 0 #", "SKIP", "fallback"):
        assert forbidden not in source
    assert "dst=/workspace,readonly" not in source


def test_container_guide_matches_current_commands_and_verified_execution() -> None:
    text = GUIDE.read_text()

    assert "`.[dev,io,plot]`".replace("`", chr(96)) in text
    assert "`127.0.0.1`".replace("`", chr(96)) in text
    assert "認証と XSRF は無効にならない" in text
    assert "36893481772" in text
    assert "D-161" in text
    assert "手動attach/Ports確認は完了" in text
    assert "scripts/smoke_container.sh" in text
    assert "container-smoke" in text
    for forbidden in (
        "make build",
        "make clean",
        "requirements-dev.txt",
        "Jupyterのトークン認証を無効化",
        "root権限でのJupyter実行を許可",
        "sudo chown -R",
    ):
        assert forbidden not in text


def test_container_guide_documents_wsl_path_translation_recovery() -> None:
    text = GUIDE.read_text()

    for required in (
        "Failed to translate",
        "[automount]",
        "enabled=true",
        "[interop]",
        "appendWindowsPath=true",
        "wsl --shutdown",
    ):
        assert required in text


def test_container_guide_local_links_resolve() -> None:
    missing: list[str] = []
    for match in LOCAL_LINK.finditer(GUIDE.read_text()):
        target = match.group("target").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://")):
            continue
        if not (GUIDE.parent / target).resolve().exists():
            missing.append(target)

    assert missing == []
