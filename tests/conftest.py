import os

import pytest

# Headless pygame for the UI tests (must be set before pygame is imported).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run the slow tests (several minutes)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
