"""M0 gate: the harness and instruments reproduce the Architecture B probe's reference numbers."""

import json

import pytest

from snake_agent import reproduce
from snake_agent.__main__ import main

QUICK = [name for name, (_, full_only) in reproduce.SECTIONS.items() if not full_only]
SLOW = [name for name, (_, full_only) in reproduce.SECTIONS.items() if full_only]


@pytest.fixture(scope="module")
def probe():
    return reproduce.load_probe()


def failures(checks):
    return [
        f"{c.section}: {c.name}\n  expected {c.expected!r}\n  got      {c.actual!r}"
        for c in checks
        if not c.ok
    ]


@pytest.mark.parametrize("section", QUICK)
def test_reference_numbers(probe, section):
    checks = reproduce.SECTIONS[section][0](probe, False)
    assert checks and not failures(checks)


@pytest.mark.slow
@pytest.mark.parametrize("section", list(reproduce.SECTIONS))
def test_reference_numbers_in_full(probe, section):
    checks = reproduce.SECTIONS[section][0](probe, True)
    assert checks and not failures(checks)


def test_a_wrong_number_is_reported(probe):
    altered = json.loads(json.dumps(probe))
    altered["exact_state_space"]["4x4"]["body_paths"] += 1
    checks = reproduce.state_space(altered, False)
    assert len(failures(checks)) == 1


def test_command_line(capsys, tmp_path):
    out = tmp_path / "t1.json"
    assert (
        main(["evaluate", "t1", "--agent", "shortcut", "--board", "6x6", "--games", "5", "--json", str(out)])
        == 0
    )
    assert json.loads(out.read_text())["summary"]["won"] == 5
    assert (
        main(
            [
                "evaluate",
                "t1",
                "--agent",
                "greedy",
                "--board",
                "6x6",
                "--games",
                "5",
                "--max-mean-moves",
                "300",
            ]
        )
        == 1
    )
    assert main(
        ["evaluate", "t2", "--agent", "guarded-greedy", "--board", "6x6", "--sources", "10", "--compare"]
    ) in (0, 1)
    assert main(["solve", "--board", "3x3"]) == 0
    assert main(["evaluate", "t1", "--agent", "nobody"]) == 2
    text = capsys.readouterr().out
    assert "T1  shortcut  6x6" in text and "gate failed" in text and "3x3" in text
