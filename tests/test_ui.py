"""Headless tests of the pygame front-end (SDL's dummy video driver, see conftest)."""

import math

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from snake_game import (  # noqa: E402
    DeathCause,
    Direction,
    GameConfig,
    GameMode,
    GameState,
    SnakeEnv,
    SnakeGame,
    Status,
    StepResult,
    VectorSnakeEnv,
)
from snake_game.__main__ import build_parser, load_agent, main  # noqa: E402
from snake_game.policies import GreedyPolicy, HamiltonianPolicy  # noqa: E402
from snake_game.ui import BoardRenderer, EnvViewer, SnakeApp  # noqa: E402
from snake_game.ui.renderer import is_successor  # noqa: E402


def key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0)


def press(app, key, frames=1):
    app.handle_event(key_event(key))
    for _ in range(frames):
        app.frame(1 / 60)


@pytest.fixture
def app():
    application = SnakeApp(GameConfig(10, 10), seed=1, cell_size=16)
    application.open()
    yield application
    application.close()


def test_is_successor():
    game = SnakeGame(seed=0)
    before = game.state()
    game.step(Direction.UP)
    after = game.state()
    assert is_successor(before, after)
    assert not is_successor(after, before)
    game.step(Direction.UP)
    assert not is_successor(before, game.state())


def test_renderer_draws_every_kind_of_state():
    pygame.font.init()
    surface = pygame.Surface((400, 400))
    renderer = BoardRenderer(8, 8, 20)
    game = SnakeGame(GameConfig(8, 8), seed=4)
    renderer.set_state(game.state())
    policy = GreedyPolicy()
    while not game.done:
        result = game.step(policy(game.state()))
        renderer.set_state(game.state(), result, duration=0.1)
        renderer.update(0.05)
        renderer.draw(surface, (10, 10))
        renderer.draw(surface, (10, 10), progress=0.3)
    assert game.status is Status.LOST
    for _ in range(30):  # death animation
        renderer.update(0.05)
        renderer.draw(surface, (10, 10))
    won = GameState(2, 2, ((0, 1), (0, 0), (1, 0), (1, 1)), Direction.DOWN, None, score=1, status=Status.WON)
    renderer = BoardRenderer(2, 2, 30)
    renderer.set_state(won, StepResult(Direction.DOWN, True, Status.WON))
    for _ in range(20):
        renderer.update(0.1)
        renderer.draw(surface, (0, 0))
    assert renderer.effects.particles  # confetti
    pixels = pygame.surfarray.array3d(surface)
    assert pixels.std() > 0


def test_step_mode_moves_on_key_presses(app):
    assert app.controller.mode is GameMode.STEP
    head = app.game.head
    press(app, pygame.K_UP)
    assert app.game.head == (head[0], head[1] - 1)
    press(app, pygame.K_DOWN)  # reversal: ignored
    assert app.game.steps == 1
    press(app, pygame.K_SPACE)  # straight ahead
    assert app.game.head == (head[0], head[1] - 2)
    for _ in range(30):
        app.frame(1 / 60)
    assert app.game.steps == 2  # never moves by itself


def test_auto_mode_waits_for_the_first_key_and_follows_speed(app):
    press(app, pygame.K_m)
    assert app.controller.mode is GameMode.AUTO and app.controller.paused
    for _ in range(30):
        app.frame(1 / 60)
    assert app.game.steps == 0
    press(app, pygame.K_UP)
    assert not app.controller.paused
    speed = app.controller.speed
    for _ in range(int(60 / speed) + 2):
        app.frame(1 / 60)
    assert app.game.steps >= 1
    press(app, pygame.K_EQUALS)
    assert app.controller.speed > speed
    press(app, pygame.K_MINUS)
    press(app, pygame.K_MINUS)
    assert app.controller.speed < speed
    for _ in range(40):
        press(app, pygame.K_PLUS, frames=0)
    assert math.isinf(app.controller.speed)
    press(app, pygame.K_SPACE)
    assert app.controller.paused


def test_agent_render_toggle_and_help(app):
    press(app, pygame.K_p)
    assert app.autopilot
    press(app, pygame.K_SPACE)  # step mode: the agent makes one move
    assert app.game.steps == 1
    press(app, pygame.K_m)
    assert not app.controller.paused  # the agent does not wait for a key
    press(app, pygame.K_h)
    assert app.show_help and app.controller.paused
    press(app, pygame.K_h)
    assert not app.show_help and not app.controller.paused
    press(app, pygame.K_r)
    assert not app.render_enabled
    app.controller.speed = 1000
    steps = app.game.steps
    for _ in range(10):
        app.frame(1 / 30)
    assert app.game.steps > steps or app.game.done
    press(app, pygame.K_r)
    assert app.render_enabled
    press(app, pygame.K_LEFT)  # steering takes control back from the agent
    assert not app.autopilot


def test_game_over_and_restart(app):
    for _ in range(12):
        press(app, pygame.K_RIGHT)
    assert app.game.done and app.game.death_cause is DeathCause.WALL
    assert app.games == 1
    for _ in range(5):
        app.frame(1 / 60)
    press(app, pygame.K_RETURN)
    assert not app.game.done and app.game.steps == 0


def test_agent_wins_and_auto_restarts():
    app = SnakeApp(
        GameConfig(4, 4),
        seed=0,
        agent=HamiltonianPolicy(),
        agent_name="hamiltonian",
        mode="auto",
        speed=math.inf,
        render=False,
        auto_restart=True,
        cell_size=20,
    )
    app.open()
    try:
        for _ in range(200):
            app.frame(1 / 30)
            if app.games >= 2:
                break
        assert app.games >= 2
        assert app.best == GameConfig(4, 4).max_score
    finally:
        app.close()


def test_run_loop_resize_and_quit(app):
    pygame.display.set_mode((520, 700), pygame.RESIZABLE)
    old_cell = app.renderer.cell
    app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=(520, 700), w=520, h=700))
    assert app.renderer.cell != old_cell
    app.frame(1 / 60)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()
    assert not app.running


def test_run_with_a_frame_limit():
    app = SnakeApp(GameConfig(6, 6), seed=0, mode="auto", agent=GreedyPolicy(), speed=30, cell_size=12)
    app.run(max_frames=10)
    assert app.frames == 10 and not app.running


def test_env_viewer_frames_and_window_keys():
    state = SnakeGame(GameConfig(6, 5), seed=0).state()
    frame = EnvViewer(6, 5, window=False, cell_size=10).frame(state, caption="game 1")
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    viewer = EnvViewer(6, 5, cell_size=10)
    viewer.show(state)
    assert viewer.is_open
    pygame.event.post(key_event(pygame.K_r))
    viewer._last_pump = -1e9
    viewer.pump()
    assert not viewer.enabled
    viewer.show(state)  # disabled: nothing drawn
    pygame.event.post(key_event(pygame.K_EQUALS))
    pygame.event.post(key_event(pygame.K_r))
    viewer._last_pump = -1e9
    viewer.pump()
    assert viewer.enabled and viewer.speed_factor == 2
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    viewer._last_pump = -1e9
    viewer.pump()
    assert not viewer.enabled and not viewer.is_open  # closing the window turns rendering off
    viewer.close()


def test_environment_rendering():
    env = SnakeEnv(6, 6, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(0)
    assert env.render().shape[2] == 3
    env.close()
    venv = VectorSnakeEnv(3, 6, 6, render_mode="rgb_array", render_env=1)
    venv.reset(seed=0)
    assert venv.render().shape[2] == 3
    venv.close()
    with pytest.warns(UserWarning):
        assert SnakeEnv().render() is None


def test_human_rendering_can_be_switched_off():
    env = SnakeEnv(6, 6, render_mode="human", render_fps=500)
    env.reset(seed=0)
    env.step(0)
    assert env.render_enabled and env._viewer.is_open
    env.render_enabled = False
    for _ in range(20):
        _, _, term, trunc, _ = env.step(env.action_masks().argmax())
        if term or trunc:
            env.reset()
    env.render_enabled = True
    env.step(env.action_masks().argmax()) if not env.game.done else env.reset()
    env.close()
    venv = VectorSnakeEnv(4, 6, 6, render_mode="human")
    venv.reset(seed=0)
    for _ in range(5):
        venv.step(venv.action_masks().argmax(axis=1))
    venv.render_enabled = False
    venv.step(venv.action_masks().argmax(axis=1))
    venv.close()


def test_command_line(monkeypatch):
    args = build_parser().parse_args(
        ["--width", "8", "--mode", "auto", "--speed", "max", "--agent", "greedy"]
    )
    assert args.width == 8 and args.mode == "auto" and math.isinf(args.speed) and args.agent == "greedy"
    policy, name = load_agent("snake_game.policies:HamiltonianPolicy")
    assert isinstance(policy, HamiltonianPolicy) and name == "HamiltonianPolicy"
    assert load_agent("random")[1] == "random"
    for bad in ("nonsense", "snake_game.policies:Missing", ":x"):
        with pytest.raises((ValueError, AttributeError)):
            load_agent(bad)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--speed", "0"])
    original_run = SnakeApp.run
    monkeypatch.setattr(SnakeApp, "run", lambda self, max_frames=None: original_run(self, 3))
    assert (
        main(
            ["--width", "6", "--height", "6", "--agent", "hamiltonian", "--mode", "auto", "--cell-size", "10"]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        main(["--agent", "nonsense"])
