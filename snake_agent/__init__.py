"""The Snake agent, milestone M0: the information contract, exact instruments and evaluation harness.

* :mod:`~snake_agent.contract`: what the agent sees and does, and what counts as
  winning: public state only, relative actions, safety strictly first, the
  ``N + L`` fruit timeout and the outcome classes.
* :mod:`~snake_agent.survival`: the survival certificate and the lost-proof search.
* :mod:`~snake_agent.cycles`: cycle order, the cycle-order envelope and the shortcut cycle.
* :mod:`~snake_agent.references`: the reference policies a learned agent must beat.
* :mod:`~snake_agent.recovery`: recovery starts for track T2.
* :mod:`~snake_agent.evaluation`: the harness for tracks T1 to T3.
* :mod:`~snake_agent.exact`: exact solutions of small boards (4x4).
* :mod:`~snake_agent.seeds` and :mod:`~snake_agent.stats`: held-out seeds and confidence intervals.

``python -m snake_agent --help`` lists the command-line tools. See
``snake_agent/README.md`` for the documentation.
"""

from __future__ import annotations

from .contract import NO_LIMIT, TRAINING_RULES, Agent, Outcome, Rules, choose_safely, planning_game
from .cycles import CycleEnvelope, ShortcutCyclePolicy
from .evaluation import (
    AgentSpec,
    Evaluation,
    GameRecord,
    Provenance,
    evaluate_recovery,
    evaluate_standard,
    evaluate_t3,
    recovery_gate,
    run_game,
    standard_gate,
)
from .exact import ExactSolution, evaluate_policy, solve
from .recovery import RecoveryStart, recovery_starts
from .references import REFERENCES, GuardedGreedyPolicy
from .survival import Board, LostProof, MoveClass, classify_moves, prove_lost

__all__ = [
    "NO_LIMIT",
    "REFERENCES",
    "TRAINING_RULES",
    "Agent",
    "AgentSpec",
    "Board",
    "CycleEnvelope",
    "Evaluation",
    "ExactSolution",
    "GameRecord",
    "GuardedGreedyPolicy",
    "LostProof",
    "MoveClass",
    "Outcome",
    "Provenance",
    "RecoveryStart",
    "Rules",
    "ShortcutCyclePolicy",
    "choose_safely",
    "classify_moves",
    "evaluate_policy",
    "evaluate_recovery",
    "evaluate_standard",
    "evaluate_t3",
    "planning_game",
    "prove_lost",
    "recovery_gate",
    "recovery_starts",
    "run_game",
    "solve",
    "standard_gate",
]
