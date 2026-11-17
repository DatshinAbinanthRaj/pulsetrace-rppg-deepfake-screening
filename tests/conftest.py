import pytest

from pulsetrace.synthetic import make_clip


@pytest.fixture(scope="session")
def real_clip():
    return make_clip("real", seconds=10, seed=11, bpm=72)


@pytest.fixture(scope="session")
def fake_clips():
    return {k: make_clip(k, seconds=10, seed=21 + i) for i, k in enumerate(["no_pulse", "incoherent", "flicker"])}
