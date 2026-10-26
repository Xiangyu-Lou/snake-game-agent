import pytest

from snake_agent.survival import Board, MoveClass, classify_moves, direction_of, prove_lost
from snake_game import Direction, GameConfig, GameState, SnakeGame, Status
from snake_game.policies import GreedyPolicy, RandomPolicy


def random_states(size, seeds, policy=RandomPolicy):
    for seed in seeds:
        game = SnakeGame(GameConfig(*size), seed=seed)
        agent = policy(seed) if policy is RandomPolicy else policy()
        while not game.done:
            state = game.state()
            yield state
            game.step(agent(state))


def test_board_agrees_with_the_engine():
    for state in random_states((7, 6), range(10)):
        board = Board(state)
        game = SnakeGame.from_state(state, seed=0)
        assert [board.target(int(d)) >= 0 for d in Direction] == [game.is_safe(d) for d in Direction]
        assert board.body() == [y * 7 + x for x, y in state.snake]
        assert board.release(board.head) == state.length
        tail = state.tail[1] * 7 + state.tail[0]
        assert board.release(tail) == 1


def test_moves_are_undone_exactly():
    state = SnakeGame(GameConfig(8, 8), seed=1).state()
    board = Board(state)
    before = (list(board.entered), board.head, board.heading, board.length, board.t)
    undo = []
    for d in (1, 2, 2, 3, 0):
        cell = board.target(d)
        assert cell >= 0
        undo.append((cell, board.move(d, cell, grow=d == 2)))
    for cell, record in reversed(undo):
        board.unmove(cell, record)
    assert (board.entered, board.head, board.heading, board.length, board.t) == before


def test_certificate_loops_can_be_followed_forever():
    replayed = 0
    for size, seeds, policy in (((6, 6), range(8), RandomPolicy), ((10, 8), range(2), GreedyPolicy)):
        for state in random_states(size, seeds, policy):
            board = Board(state)
            if not board.has_survival_certificate():
                assert board.certificate_loop() is None
                continue
            loop = board.certificate_loop()
            assert loop is not None and len(loop) >= state.length and len(set(loop)) == len(loop)
            for step in range(2 * len(loop)):  # twice round, without eating
                cell = loop[step % len(loop)]
                d = int(direction_of(board, cell))
                assert board.target(d) == cell
                board.move(d, cell)
            replayed += 1
    assert replayed > 1000


def test_chasing_the_tail_is_certified():
    #  H o .     the head can step onto the tail, which moves away
    #  t o .
    state = GameState(3, 3, ((0, 0), (1, 0), (1, 1), (0, 1)), Direction.LEFT, (2, 2))
    board = Board(state)
    assert board.has_survival_certificate()
    assert board.certificate_loop() == [
        1 * 3 + 0,
        1 * 3 + 1,
        0 * 3 + 1,
        0,
    ]  # tail, then the body back to the head


def test_a_snake_without_safe_moves_is_lost():
    # RandomPolicy only dies when no safe move is left, so its last state is a proof at depth 0.
    states = list(random_states((6, 6), [4]))
    proof = prove_lost(Board(states[-1]), 1000)
    assert proof.lost is True and proof.nodes == 1 and proof.depth == 0


def test_proofs_must_come_before_the_board_can_fill():
    #  H o o     two free cells: (0, 1) holds the fruit, (0, 2) is empty
    #  * o o     Without eating, every continuation collides within two moves,
    #  . o o     but the real snake eats twice and fills the board.
    state = GameState(3, 3, ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0)), Direction.LEFT, (0, 1))
    proof = prove_lost(Board(state), 10_000)
    assert proof.lost is None
    assert proof.depth >= state.free_cells
    game = SnakeGame.from_state(state, seed=0)
    game.step(Direction.DOWN)
    game.step(Direction.DOWN)  # the only free cell is where the next fruit appears
    assert game.won


def test_unfinished_proofs_are_unknown_and_leave_the_board_unchanged():
    state = GameState(3, 3, ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1), (2, 0)), Direction.LEFT, (0, 1))
    board = Board(state)
    before = list(board.entered)
    proof = prove_lost(board, 1)
    assert proof.lost is None and proof.nodes == 2
    assert board.entered == before and board.t == 0
    survivable = SnakeGame(GameConfig(8, 8), seed=0).state()
    assert prove_lost(Board(survivable), 10).lost is False


def test_classify_moves():
    # 2x2: turning left (down) eats the last fruit; straight and right leave the board.
    state = GameState(2, 2, ((0, 0), (1, 0), (1, 1)), Direction.LEFT, (0, 1))
    assert classify_moves(state) == (MoveClass.WON, MoveClass.FATAL, MoveClass.FATAL)
    start = SnakeGame(GameConfig(8, 8), seed=0).state()
    assert classify_moves(start) == (MoveClass.CERTIFIED,) * 3
    with pytest.raises(ValueError):
        classify_moves(
            GameState(2, 2, ((0, 1), (0, 0), (1, 0), (1, 1)), Direction.DOWN, None, status=Status.WON)
        )


def test_lost_moves_need_a_search_budget():
    found = 0
    for state in random_states((6, 6), range(30)):
        quick, searched = classify_moves(state), classify_moves(state, budget=10_000)
        for a, b in zip(quick, searched, strict=True):
            assert a == b or (a is MoveClass.UNKNOWN and b is MoveClass.LOST)
            found += b is MoveClass.LOST
    assert found > 0
