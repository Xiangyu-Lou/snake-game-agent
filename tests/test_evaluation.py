import json

import pytest

from snake_agent import seeds
from snake_agent.contract import NO_LIMIT, TRAINING_RULES, Outcome
from snake_agent.cycles import CycleEnvelope, ShortcutCyclePolicy
from snake_agent.evaluation import (
    STALL_FACTOR,
    AgentSpec,
    GameRun,
    Provenance,
    best_rates,
    evaluate_recovery,
    evaluate_standard,
    evaluate_t3,
    format_summary,
    recovery_gate,
    run_game,
    standard_gate,
)
from snake_agent.recovery import recovery_starts
from snake_agent.references import REFERENCES, GuardedGreedyPolicy
from snake_game import GameConfig, GameState, RelativeAction, SnakeGame
from snake_game.policies import HamiltonianPolicy


def circle(state):
    """Turn right forever: on an open board the snake circles a 2x2 square without eating."""
    return RelativeAction.RIGHT


def standard(side=6, seed=0):
    return SnakeGame(GameConfig(side, side), seed=seed)


def fruit_outside_the_square(side=6):
    for seed in range(100):
        game = standard(side, seed)
        if game.fruit[1] < side // 2 - 1 or game.fruit[1] > side // 2 + 1:
            return seed
    raise AssertionError("no seed puts the fruit away from the start")


def test_a_cycle_strategy_wins():
    record = run_game(standard(), ShortcutCyclePolicy(), envelope=CycleEnvelope.for_state(standard().state()))
    assert record.outcome is Outcome.WON
    assert record.fruits == 33 and record.length == 36 and record.detail is None
    assert record.envelope_exit is None
    assert 0 < record.timeout_margin < 1 and record.longest_without_fruit < 36


def test_collisions_are_losses():
    record = run_game(standard(), lambda state: RelativeAction.STRAIGHT)
    assert record.outcome is Outcome.LOST and record.detail == "wall" and record.moves == 3


def test_loops_time_out_under_the_training_rules():
    seed = fruit_outside_the_square()
    record = run_game(standard(seed=seed), circle, TRAINING_RULES)
    assert record.outcome is Outcome.TIMED_OUT
    assert record.moves == 36 + 3 and record.longest_without_fruit == 39 and record.timeout_margin == 1.0


def test_without_the_timeout_endless_games_are_censored():
    seed = fruit_outside_the_square()
    loop = run_game(standard(seed=seed), circle, NO_LIMIT, detect_loops=True)
    assert (loop.outcome, loop.detail, loop.moves) == (Outcome.CENSORED, "loop", 5)  # state 5 repeats state 1
    stall = run_game(standard(seed=seed), circle, NO_LIMIT, stall_limit=50)
    assert (stall.outcome, stall.detail, stall.moves) == (Outcome.CENSORED, "stall", 51)
    cap = run_game(standard(seed=seed), circle, NO_LIMIT, move_cap=10)
    assert (cap.outcome, cap.detail, cap.moves) == (Outcome.CENSORED, "cap", 10)


def test_memory_keeps_loop_detection_exact():
    # An agent that alternates between circling and going straight is not a function of the state.
    class Alternating:
        def __init__(self):
            self.calls = 0

        def __call__(self, state):
            self.calls += 1
            return RelativeAction.RIGHT if self.calls % 8 else RelativeAction.STRAIGHT

    seed = fruit_outside_the_square(10)
    agent = Alternating()
    fooled = run_game(standard(10, seed), agent, NO_LIMIT, detect_loops=True)
    assert fooled.detail == "loop" and fooled.moves == 5  # wrongly declared endless
    agent = Alternating()
    exact = run_game(standard(10, seed), agent, NO_LIMIT, detect_loops=True, memory=lambda: agent.calls % 8)
    assert exact.moves > 5


def test_leaving_the_envelope_is_recorded():
    game = standard()
    record = run_game(
        game, lambda state: RelativeAction.STRAIGHT, envelope=CycleEnvelope.for_state(game.state())
    )
    assert record.envelope_exit == 2  # into the wall: the cycle turns before it


def test_agents_see_only_the_public_state():
    seen = []

    def agent(state):
        seen.append(type(state))
        return HamiltonianPolicy()(state)

    run_game(standard(4), agent)
    assert set(seen) == {GameState}


def test_relative_and_absolute_agents_play_the_same_game():
    policy = HamiltonianPolicy()

    def relative(state):
        d = policy(state)
        return RelativeAction((int(d) - int(state.direction) + 1) % 4)

    a = run_game(standard(6, 3), HamiltonianPolicy())
    b = run_game(standard(6, 3), relative)
    assert a == b
    with pytest.raises(TypeError):
        run_game(standard(), lambda state: 1)


def test_standard_track_and_gate():
    evaluation = evaluate_standard(REFERENCES["shortcut"], 6, 6, range(20))
    summary = evaluation.summary()
    assert (summary.games, summary.won, summary.lost, summary.timed_out, summary.censored) == (
        20,
        20,
        0,
        0,
        0,
    )
    assert summary.envelope_exits == 0
    assert summary.win_interval[0] > 0.83 and summary.win_interval[1] == 1.0
    assert 150 < summary.mean_moves < 260
    assert summary.moves_interval[0] < summary.mean_moves < summary.moves_interval[1]
    assert standard_gate(evaluation, 297).passed  # below the plain cycle's expected 297 moves
    gate = standard_gate(evaluation, 150)
    assert not gate.passed and "mean moves" in gate.reasons[0]
    greedy = evaluate_standard(REFERENCES["greedy"], 6, 6, range(20))
    reasons = standard_gate(greedy, 10_000).reasons
    assert any("not won" in r for r in reasons) and any("envelope" in r for r in reasons)
    assert "T1" in format_summary(evaluation) and "envelope exits: 0" in format_summary(evaluation)


def test_timeouts_never_bind_for_the_shortcut_cycle():
    with_rule = evaluate_standard(REFERENCES["shortcut"], 8, 8, range(10), TRAINING_RULES)
    without = evaluate_standard(REFERENCES["shortcut"], 8, 8, range(10), NO_LIMIT)
    assert with_rule.records == without.records
    assert with_rule.summary().timeout_margin < 1


def test_recovery_track_gate_and_t3():
    starts = recovery_starts(8, 8, range(30))
    fruit_seeds = range(500, 500 + len(starts))
    assert {s.band for s in starts} == {0.3, 0.5, 0.7}
    references = [
        evaluate_recovery(REFERENCES[n], starts, fruit_seeds) for n in ("hamiltonian", "guarded-greedy")
    ]
    rates = best_rates(references)
    assert set(rates) == {0.3, 0.5, 0.7}
    for reference in references:
        assert not recovery_gate(reference, rates).passed  # a reference cannot beat itself
    assert recovery_gate(references[0], {band: 0.0 for band in rates}).passed == all(
        references[0].summary(b).won > 0 for b in rates
    )
    assert "T2" in format_summary(references[0]) and "70%" in format_summary(references[0])

    with pytest.raises(ValueError):
        evaluate_t3(REFERENCES["shortcut"], 6, 6, range(2), starts, fruit_seeds)
    learned = AgentSpec(
        "self-play", lambda seed: GuardedGreedyPolicy(), Provenance.SELF_PLAY, deterministic=True
    )
    t1, t2 = evaluate_t3(learned, 8, 8, range(3), starts[:2], fruit_seeds)
    assert t1.label == "T3 (T1 starts)" and t2.label == "T3 (T2 starts)"
    assert len(t1.records) == 3 and len(t2.records) == 2
    imitation = AgentSpec("imitation", learned.factory, Provenance.DEMONSTRATIONS)
    assert evaluate_standard(imitation, 6, 6, range(1)).label == "T1 (with demonstrations)"


def test_censored_games_widen_the_win_interval():
    spec = AgentSpec("circle", lambda seed: circle, deterministic=True)
    evaluation = evaluate_standard(spec, 6, 6, [fruit_outside_the_square()], NO_LIMIT)
    summary = evaluation.summary()
    assert summary.censored == 1 and summary.win_interval == (0.0, 1.0)
    timed = evaluate_standard(spec, 6, 6, [fruit_outside_the_square()], TRAINING_RULES).summary()
    assert timed.timed_out == 1 and timed.win_interval[1] < 1.0


def test_results_are_json_ready():
    evaluation = evaluate_standard(REFERENCES["greedy"], 6, 6, range(3), NO_LIMIT)
    data = json.loads(json.dumps(evaluation.to_dict()))
    assert data["track"] == "T1" and data["agent"] == "greedy" and not data["fruit_timeout"]
    assert len(data["games"]) == 3 and data["summary"]["games"] == 3


def test_stall_factor_matches_the_probe():
    assert STALL_FACTOR == 20


def test_seed_ranges_are_held_out():
    assert seeds.TRAIN.stop <= seeds.VALIDATION.start and seeds.VALIDATION.stop <= seeds.TEST.start
    assert seeds.check_training_seed(123) == 123
    for held_out in (seeds.VALIDATION.start, seeds.TEST.start + 5, seeds.TEST.stop - 1):
        with pytest.raises(ValueError):
            seeds.check_training_seed(held_out)
    standard_seeds = seeds.standard_seeds(seeds.TEST, 300)
    sources = seeds.recovery_source_seeds(seeds.TEST, 60)
    fruits = seeds.recovery_fruit_seeds(seeds.TEST, 130)
    blocks = [set(standard_seeds), set(sources), set(fruits)]
    assert all(seed in seeds.TEST for block in blocks for seed in block)
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])
    with pytest.raises(ValueError):
        seeds.TEST.take(10, seeds.TEST.stop - seeds.TEST.start - 5)


def test_games_can_advance_a_move_at_a_time_together():
    spec = REFERENCES["guarded-greedy"]
    expected = [run_game(standard(6, seed), spec.factory(0), seed=seed) for seed in range(4)]
    runs = [GameRun(standard(6, seed), seed=seed) for seed in range(4)]
    agents = [spec.factory(0) for _ in runs]
    while any(run.result is None for run in runs):
        for run, agent in zip(runs, agents, strict=True):
            if (state := run.next_state()) is not None:
                run.play(agent(state))
    assert [run.result for run in runs] == expected
    assert runs[0].next_state() is None
    with pytest.raises(RuntimeError):
        runs[0].play(RelativeAction.STRAIGHT)
