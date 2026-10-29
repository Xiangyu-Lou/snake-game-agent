import dataclasses
import random

import numpy as np
import pytest

from snake_agent.contract import NO_LIMIT, TRAINING_RULES
from snake_agent.cycles import CycleEnvelope
from snake_agent.encoding import (
    ACTION_FEATURES,
    GLOBALS,
    PLANES,
    SATURATION,
    canonical_cycle,
    encode,
    mirror,
    moves_ceiling,
    moves_scale,
    stack,
)
from snake_agent.exact import solve, standard_starts
from snake_agent.survival import Board, classify_moves
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame

P = {name: i for i, name in enumerate(PLANES)}
G = {name: i for i, name in enumerate(GLOBALS)}
A = {name: i for i, name in enumerate(ACTION_FEATURES)}
CYCLE_PLANES = [P[n] for n in PLANES if n.startswith("cycle") or n in ("envelope_free", "hole")]
CYCLE_GLOBALS = [G["in_cycle_order"], G["fruit_ahead"], G["tail_ahead"]]
CYCLE_ACTIONS = [A["envelope"], A["jump"]]


def random_states(width, height, games, seed=0):
    """States of games played by random safe moves, which visit every heading."""
    rng = random.Random(seed)
    states = []
    for g in range(games):
        game = SnakeGame(GameConfig(width, height), seed=seed + g)
        while not game.done:
            states.append(game.state())
            safe = game.safe_directions()
            game.step(rng.choice(safe) if safe else game.direction)
    return states


def mirrored(state):
    """The state reflected left to right."""
    w = state.width
    flip = {Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
    return dataclasses.replace(
        state,
        snake=tuple((w - 1 - x, y) for x, y in state.snake),
        direction=flip.get(state.direction, state.direction),
        fruit=None if state.fruit is None else (w - 1 - state.fruit[0], state.fruit[1]),
    )


def frame_cell(encoded, plane):
    """The (x, y) cells of the canvas where ``plane`` is set."""
    ys, xs = np.nonzero(encoded.planes[P[plane]])
    return sorted(zip(xs.tolist(), ys.tolist(), strict=True))


def absolute(encoded, plane, state):
    """``plane`` turned back from the head's frame onto the board, indexed ``[y, x]``."""
    k = int(state.direction)
    rows, columns = (state.height, state.width) if k % 2 == 0 else (state.width, state.height)
    return np.rot90(encoded.planes[P[plane], :rows, :columns], -k)


def test_shapes_and_canvas():
    state = standard_starts(4, 4)[0]
    e = encode(state, TRAINING_RULES, size=6)
    assert e.planes.shape == (len(PLANES), 6, 6) and e.planes.dtype == np.float32
    assert e.actions.shape == (3, len(ACTION_FEATURES)) and e.globals.shape == (len(GLOBALS),)
    assert e.planes[P["valid"]].sum() == 16 and e.planes[P["valid"], :4, :4].all()
    assert not e.planes[:, 4:, :].any() and not e.planes[:, :, 4:].any()
    assert e.planes[P["head"]].reshape(-1)[e.head] == 1.0
    with pytest.raises(ValueError):
        encode(state, TRAINING_RULES, size=3)
    planes, actions, globals_, heads = stack([e, e])
    assert planes.shape == (2, len(PLANES), 6, 6) and heads.tolist() == [e.head, e.head]
    assert actions.shape == (2, 3, len(ACTION_FEATURES)) and globals_.shape == (2, len(GLOBALS))


@pytest.mark.parametrize("heading", list(Direction))
def test_the_frame_turns_the_heading_up(heading):
    # A two-cell snake in the middle of a 5x6 board, the fruit on each target cell in turn.
    head = (2, 2)
    dx, dy = heading.delta
    snake = (head, (head[0] - dx, head[1] - dy))
    for action in RelativeAction:
        tx, ty = heading.turn(action).delta
        state = GameState(5, 6, snake, heading, (head[0] + tx, head[1] + ty))
        e = encode(state, TRAINING_RULES, size=6)
        hy, hx = divmod(e.head, 6)
        expected = {
            RelativeAction.LEFT: (hx - 1, hy),
            RelativeAction.STRAIGHT: (hx, hy - 1),
            RelativeAction.RIGHT: (hx + 1, hy),
        }[action]
        assert frame_cell(e, "fruit") == [expected]
        assert frame_cell(e, "head") == [(hx, hy)]
        assert frame_cell(e, "tail") == [(hx, hy + 1)]  # the neck is behind the head
        assert e.actions[:, A["eats"]].tolist() == [float(a == action) for a in RelativeAction]
        # The walls are counted in cells in the head's frame.
        rows, columns = (6, 5) if heading in (Direction.UP, Direction.DOWN) else (5, 6)
        assert e.planes[P["wall_up"], hy, hx] * 16 == hy
        assert e.planes[P["wall_left"], hy, hx] * 16 == hx
        assert e.planes[P["wall_right"], hy, hx] * 16 == columns - 1 - hx
        assert e.planes[P["wall_down"], hy, hx] * 16 == rows - 1 - hy
        assert e.planes[P["valid"]].sum() == 30 and e.planes[P["valid"], :rows, :columns].all()


def test_body_planes_and_release_times():
    state = random_states(6, 6, 1, seed=3)[60]
    e = encode(state, TRAINING_RULES)
    length, cells = state.length, 36
    assert e.planes[P["body"]].sum() == length - 1
    release = e.planes[P["release"]] * cells
    assert sorted(np.round(release[release > 0]).astype(int).tolist()) == list(range(1, length + 1))
    assert release.reshape(-1)[e.head] == pytest.approx(length)
    ys, xs = np.nonzero(e.planes[P["tail"]])
    assert release[ys[0], xs[0]] == pytest.approx(1)
    cells_plane = e.planes[P["release_cells"]] * 16
    assert cells_plane.max() == pytest.approx(min(length, 16))
    # One direction plane per body segment, none on the head.
    toward = e.planes[P["toward_up"] : P["toward_left"] + 1]
    assert toward.sum() == length - 1 and (toward.sum(axis=0) == e.planes[P["body"]]).all()


def test_reach_distance_and_slack():
    # Head at (0, 1) heading down; the body hooks round the free cell (1, 1).
    snake = ((0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2))
    state = GameState(4, 4, snake, Direction.DOWN, (3, 3))
    e = encode(state, TRAINING_RULES)
    reach = absolute(e, "reach", state)
    assert reach.sum() == 9 and not (reach * absolute(e, "body", state)).any()
    distance = np.round(absolute(e, "distance", state) * 16).astype(int)
    assert distance[1, 1] == distance[2, 0] == 1 and distance[0, 3] == 8  # (1, 1), (0, 2) and (3, 0)
    # Slack: the latest arrival from a reachable neighbour minus the release time.
    slack = np.round(absolute(e, "slack", state) * 16).astype(int)
    assert slack[2, 1] == 4 - 1  # the tail (1, 2), reached last from (1, 3) at distance 3
    assert slack[2, 2] == 7 - 2  # (2, 2) from (3, 2) at distance 6
    assert slack[0, 1] == 2 - 5  # (1, 0) only from (1, 1)
    assert slack[0, 0] == 0  # the neck borders no reachable cell: the head cannot reverse
    assert e.globals[G["certified"]] == float(Board(state).has_survival_certificate()) == 1.0


def test_action_features_follow_the_move_classes():
    for state in random_states(6, 6, 2, seed=5)[::7]:
        e = encode(state, TRAINING_RULES)
        classes = classify_moves(state)
        for a in RelativeAction:
            assert e.actions[a, A[classes[a].value]] == 1.0
            assert e.actions[a, A["fatal"] : A["unknown"] + 1].sum() == 1.0


def test_cycle_features_on_the_standard_start():
    state = SnakeGame(GameConfig(6, 6), seed=0).state()
    envelope = CycleEnvelope.for_state(state)
    e = encode(state, TRAINING_RULES)
    assert e.globals[G["in_cycle_order"]] == 1.0
    mask = envelope.mask(state)
    assert e.actions[:, A["envelope"]].tolist() == [float(m) for m in mask]
    jumps = envelope.moves(state)
    for a in RelativeAction:
        jump = jumps.get(state.direction.turn(a))
        assert e.actions[a, A["jump"]] == pytest.approx(0.0 if jump is None else jump / 36)
    assert e.globals[G["fruit_ahead"]] == pytest.approx(envelope.ahead(state, state.fruit) / 36)
    assert e.globals[G["tail_ahead"]] == pytest.approx(envelope.ahead(state, state.tail) / 36)
    limit = envelope.limit(state)
    assert e.planes[P["envelope_free"]].sum() == limit
    assert e.planes[P["hole"]].sum() == 0  # the free cells behind the tail: none at the start
    ahead = np.round(e.planes[P["cycle_ahead"]] * 36).astype(int)
    assert sorted(ahead[e.planes[P["valid"]] > 0].tolist()) == list(range(36))
    # Every cell has one successor along the cycle.
    assert (e.planes[P["cycle_up"] : P["cycle_left"] + 1].sum(axis=0)[:6, :6] == 1).all()


def test_boards_without_a_cycle_have_no_cycle_features():
    assert canonical_cycle(5, 5) is None and canonical_cycle(4, 4) is not None
    state = SnakeGame(GameConfig(5, 5), seed=0).state()
    e = encode(state, TRAINING_RULES)
    assert not e.planes[CYCLE_PLANES].any() and not e.globals[CYCLE_GLOBALS].any()
    assert not e.actions[:, CYCLE_ACTIONS].any()


def test_afterstates_have_no_fruit_and_no_action_features():
    state = SnakeGame(GameConfig(6, 6), seed=0).state()
    e = encode(state, TRAINING_RULES, afterstate=True)
    assert e.globals[G["afterstate"]] == 1.0 and not e.planes[P["fruit"]].any()
    assert not e.actions.any() and e.globals[G["fruit_ahead"]] == 0.0
    assert encode(state, TRAINING_RULES).globals[G["afterstate"]] == 0.0


def test_budget_features():
    state = dataclasses.replace(SnakeGame(GameConfig(6, 6), seed=0).state(), steps_since_fruit=9)
    budget = encode(state, TRAINING_RULES).globals[G["timeout_enabled"] : G["timeout_left"] + 1]
    assert budget.tolist() == pytest.approx([1.0, 39 / 36, 30 / 36])
    assert not encode(state, NO_LIMIT).globals[G["timeout_enabled"] : G["timeout_left"] + 1].any()


def test_mirror_matches_the_mirror_image():
    states = random_states(6, 6, 3, seed=11)[::5]
    encoded = [encode(s, TRAINING_RULES) for s in states]
    images = [encode(mirrored(s), TRAINING_RULES) for s in states]
    planes, actions, globals_, heads = stack(encoded)
    widths = np.array([6] * len(states))
    policy = np.tile(np.array([0.2, 0.3, 0.5]), (len(states), 1))
    m_planes, m_actions, m_heads, m_policy = mirror(planes, actions, heads, widths, policy)
    keep = [i for i in range(len(PLANES)) if i not in CYCLE_PLANES]
    keep_actions = [i for i in range(len(ACTION_FEATURES)) if i not in CYCLE_ACTIONS]
    for i, image in enumerate(images):
        assert m_heads[i] == image.head
        assert np.array_equal(m_planes[i][keep], image.planes[keep])
        assert np.array_equal(m_actions[i][:, keep_actions], image.actions[:, keep_actions])
    assert m_policy[0].tolist() == [0.5, 0.3, 0.2]
    # Mirroring twice gives the original batch back.
    again = mirror(m_planes, m_actions, m_heads, widths, m_policy)
    assert np.array_equal(again[0], planes) and np.array_equal(again[1], actions)
    assert np.array_equal(again[2], heads) and np.allclose(again[3], policy)
    assert globals_.shape[0] == len(states)


def test_mirror_keeps_smaller_boards_in_their_corner():
    states = random_states(4, 4, 1, seed=2)[::3]
    planes, actions, _, heads = stack([encode(s, TRAINING_RULES, size=6) for s in states])
    m_planes, _, m_heads, _ = mirror(planes, actions, heads, np.array([4] * len(states)))
    assert not m_planes[:, :, :, 4:].any() and not m_planes[:, :, 4:, :].any()
    assert (np.divmod(m_heads, 6)[1] < 4).all()
    images = [encode(mirrored(s), TRAINING_RULES, size=6) for s in states]
    assert m_heads.tolist() == [image.head for image in images]


def test_moves_scale_is_the_plain_cycle_expectation():
    assert moves_scale(33) == 297.0  # 6x6 from the standard start
    assert moves_scale(13) == 52.0  # 4x4
    assert moves_scale(1) == 1.0


def test_moves_ceiling_bounds_every_win():
    assert moves_ceiling(16, 1) * moves_scale(1) == 31  # the last fruit, within N + L = 16 + 15 moves
    assert moves_ceiling(36, 33) == pytest.approx((2 * 36 * 33 - 33 * 34 / 2) / 297)
    assert moves_ceiling(16, 0) == 0.0
    # Late in a game the best certain win takes many times the plain cycle's pace, never beyond the ceiling.
    solution = solve(4, 4)
    ratios = {}
    for length, moves in solution._moves.items():
        free = 16 - length
        finite = moves[np.isfinite(moves)]
        if finite.size:
            assert finite.max() <= moves_ceiling(16, free) * moves_scale(free)
            ratios[free] = finite.max() / moves_scale(free)
    assert ratios[2] > 12


def test_arrival_planes_hold_the_earliest_moves_through_cells_vacated_in_time():
    # Heading up along the left column of a 4x4 board, the tail at (3, 3) (as in the plan agent's tests).
    snake = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3))
    state = GameState(4, 4, snake, Direction.UP, (3, 0))
    bare = encode(state, TRAINING_RULES)
    assert not bare.planes[[PLANES.index(n) for n in ("arrival", "arrival_cells", "unreached")]].any()
    e = encode(state, TRAINING_RULES, arrival=True)
    k = int(state.direction)  # the planes are turned so that the head points up
    arrival = np.rot90(e.planes[PLANES.index("arrival")][:4, :4], -k) * 16
    cells = np.rot90(e.planes[PLANES.index("arrival_cells")][:4, :4], -k) * SATURATION
    assert (
        arrival[0, 3] == pytest.approx(3)
        and arrival[3, 2] == pytest.approx(5)
        and arrival[1, 0] == pytest.approx(7)
    )
    assert cells[3, 2] == pytest.approx(5) and arrival[0, 0] == 0  # the head itself
    assert not e.planes[PLANES.index("unreached")].any()  # every free cell is reached on time here
