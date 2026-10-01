"""Tests for installed command-line entry points."""

from rovibrational_excitation.cli import simulate


def test_simulate_cli_routes_new_run(monkeypatch):
    calls = []

    def fake_run(params, **kwargs):
        calls.append((params, kwargs))

    monkeypatch.setattr(simulate, "run_all_with_checkpoint", fake_run)

    assert simulate.main(["params.py", "--dry-run", "--no-save", "-j", "2"]) == 0
    assert calls == [
        (
            "params.py",
            {
                "nproc": 2,
                "save": False,
                "dry_run": True,
                "checkpoint_interval": 10,
            },
        )
    ]


def test_simulate_cli_routes_resume(monkeypatch):
    calls = []

    def fake_resume(path, **kwargs):
        calls.append((path, kwargs))

    monkeypatch.setattr(simulate, "resume_run", fake_resume)

    assert simulate.main(["--resume", "results/run", "-j", "3"]) == 0
    assert calls == [
        (
            "results/run",
            {"nproc": 3, "checkpoint_interval": 10},
        )
    ]
