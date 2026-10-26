from snake_agent import seeds
from snake_agent.cycles import aligned_cycles
from snake_agent.recovery import BAND_WIDTH, BANDS, held_out_starts, recovery_starts
from snake_agent.survival import Board


def test_recovery_starts_are_certified_unaligned_and_in_their_band():
    starts = recovery_starts(8, 8, range(12))
    assert starts
    per_source = {}
    for start in starts:
        state = start.state
        assert start.band in BANDS
        assert start.band <= state.length / 64 < start.band + BAND_WIDTH
        assert Board(state).has_survival_certificate()
        assert not aligned_cycles(state)
        assert start.name == f"{start.source_seed}@{state.steps}"
        per_source.setdefault(start.source_seed, []).append(start.band)
    # At most one start per band from each source game, in order of occupancy.
    assert all(bands == sorted(set(bands)) for bands in per_source.values())


def test_held_out_starts_come_from_the_test_range():
    starts = held_out_starts(6, 6, 4)
    assert all(start.source_seed in seeds.TEST for start in starts)
    assert {s.source_seed for s in starts} <= set(seeds.recovery_source_seeds(seeds.TEST, 4))
