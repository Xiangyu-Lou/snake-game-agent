import dataclasses
import math

import pytest

from snake_agent.contract import (
    NO_LIMIT,
    TRAINING_RULES,
    Outcome,
    as_policy,
    choose_safely,
    planning_game,
    play_move,
    to_direction,
    to_relative,
)
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame


def test_relative_actions_round_trip():
    for heading in Direction:
        for action in RelativeAction:
            assert to_relative(heading, to_direction(heading, action)) is action
        # The engine ignores a reversal and keeps going straight.
        assert to_relative(heading, heading.opposite) is RelativeAction.STRAIGHT


def test_moves_must_say_whether_they_are_relative():
    game = SnakeGame(seed=0)  # heading right
    play_move(game, RelativeAction.LEFT)
    assert game.direction is Direction.UP
    play_move(game, Direction.LEFT)
    assert game.direction is Direction.LEFT
    with pytest.raises(TypeError):
        play_move(game, 0)  # a bare int could be either


def test_as_policy_converts_relative_actions_for_the_window():
    state = SnakeGame(seed=0).state()
    assert as_policy(lambda s: RelativeAction.RIGHT)(state) is Direction.DOWN
    assert as_policy(lambda s: Direction.UP)(state) is Direction.UP


def test_the_fruit_timeout_grows_with_the_snake():
    state = GameState(4, 4, ((2, 2), (1, 2), (0, 2)), Direction.RIGHT, (3, 3), steps_since_fruit=5)
    assert TRAINING_RULES.timeout(state) == 16 + 3
    assert TRAINING_RULES.moves_left(state) == 14
    assert not TRAINING_RULES.timed_out(state)
    assert TRAINING_RULES.timed_out(dataclasses.replace(state, steps_since_fruit=19))
    longer = GameState(4, 4, ((3, 2), (2, 2), (1, 2), (0, 2)), Direction.RIGHT, (3, 3), steps_since_fruit=19)
    assert not TRAINING_RULES.timed_out(longer)  # one more segment, one more move
    assert NO_LIMIT.timeout(state) is None and NO_LIMIT.moves_left(state) is None
    assert not NO_LIMIT.timed_out(dataclasses.replace(state, steps_since_fruit=10**6))


def test_budget_features_follow_decision_3():
    # enabled = 1, configured = (N + L) / N, remaining = (N + L - moves since fruit) / N
    state = GameState(20, 20, ((10, 10), (9, 10), (8, 10)), Direction.RIGHT, (15, 4), steps_since_fruit=100)
    assert TRAINING_RULES.budget_features(state) == (1.0, 403 / 400, 303 / 400)
    assert NO_LIMIT.budget_features(state) == (0.0, 0.0, 0.0)


def test_outcomes():
    assert [o.value for o in Outcome] == ["won", "lost", "timed_out", "censored"]
    assert Outcome.TIMED_OUT.resolved and not Outcome.CENSORED.resolved


def test_safety_comes_strictly_first():
    # A 0.1% risk is never worth any number of moves.
    assert choose_safely([0.999, 1.0, 0.5], [10, 100_000, 1]) == 1
    # Fewer moves only break exact ties.
    assert choose_safely([1.0, 1.0, 0.5], [50, 40, 1]) == 1
    assert choose_safely([0.7, 0.7, 0.7], [3, 3, 9]) == 0
    assert choose_safely([0.2, 0.9, 0.9]) == 1
    # Allowed moves restrict the choice, e.g. to the cycle-order envelope.
    assert choose_safely([1.0, 1.0, 1.0], [3, 2, 1], [True, True, False]) == 1
    with pytest.raises(ValueError):
        choose_safely([1.0, 1.0], allowed=[False, False])
    with pytest.raises(ValueError):
        choose_safely([math.nan, 1.0])


def test_planning_depends_on_the_public_state_and_its_own_seed_only():
    state = SnakeGame(GameConfig(6, 6), seed=3).state()
    live_a, live_b = SnakeGame.from_state(state, seed=1), SnakeGame.from_state(state, seed=2)

    def fruits(game, moves=60):
        seen = []
        for _ in range(moves):
            safe = game.safe_directions()
            if game.done or not safe:
                break
            game.step(safe[0])
            seen.append(game.fruit)
        return seen

    # Two live games in the same public state have different hidden futures...
    assert fruits(live_a.clone()) != fruits(live_b.clone())
    # ...but planning from that state sees the same future, whatever the live game hides.
    plan_a, plan_b = planning_game(live_a.state(), 7), planning_game(live_b.state(), 7)
    assert plan_a.state() == state
    assert fruits(plan_a) == fruits(plan_b)
