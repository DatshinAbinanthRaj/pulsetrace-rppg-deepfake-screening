import json

import numpy as np
import pytest

from pulsetrace.analysis import (
    VERDICT_FAKE,
    VERDICT_INCONCLUSIVE,
    VERDICT_REAL,
    AnalysisConfig,
    analyze_frames,
    analyze_video,
)
from pulsetrace.synthetic import make_clip
from pulsetrace.video import VideoError, write_video


@pytest.mark.parametrize("method", ["pos", "chrom", "green"])
def test_real_clip_is_real_with_correct_bpm(real_clip, method):
    rep = analyze_frames(real_clip.frames, real_clip.fps, AnalysisConfig(method=method))
    assert rep.verdict == VERDICT_REAL
    assert abs(rep.bpm - 72) < 3
    assert rep.score > 0.8
    assert {r.name for r in rep.rois} == {"forehead", "left_cheek", "right_cheek"}


@pytest.mark.parametrize("kind", ["no_pulse", "incoherent", "flicker"])
def test_fake_clips_are_flagged(fake_clips, kind):
    c = fake_clips[kind]
    rep = analyze_frames(c.frames, c.fps)
    assert rep.verdict == VERDICT_FAKE, rep.reasons
    assert rep.score < 0.4


def test_real_scores_above_every_fake(real_clip, fake_clips):
    real = analyze_frames(real_clip.frames, real_clip.fps).score
    assert all(real > analyze_frames(c.frames, c.fps).score for c in fake_clips.values())


def test_short_clip_is_inconclusive():
    c = make_clip("real", seconds=3, seed=5)
    rep = analyze_frames(c.frames, c.fps)
    assert rep.verdict == VERDICT_INCONCLUSIVE
    assert rep.score is None
    assert any("at least" in r for r in rep.reasons)


def test_no_face_is_inconclusive():
    frames = [np.full((240, 320, 3), 90, np.uint8) for _ in range(240)]
    rep = analyze_frames(frames, 30.0)
    assert rep.verdict == VERDICT_INCONCLUSIVE
    assert any("face" in r for r in rep.reasons)


def test_low_fps_is_inconclusive(real_clip):
    rep = analyze_frames(real_clip.frames[::5], 6.0)
    assert rep.verdict == VERDICT_INCONCLUSIVE


def test_unknown_method_raises(real_clip):
    with pytest.raises(ValueError):
        analyze_frames(real_clip.frames[:10], 30.0, AnalysisConfig(method="nope"))


def test_report_json_serialisable(real_clip, fake_clips):
    for c in [real_clip, *fake_clips.values()]:
        d = analyze_frames(c.frames, c.fps).to_dict()
        s = json.dumps(d)
        assert "signals" not in d and "NaN" not in s and "Infinity" not in s


def test_analyze_video_file_roundtrip(tmp_path, real_clip):
    p = write_video(tmp_path / "real.avi", real_clip.frames, real_clip.fps)
    rep = analyze_video(p)
    assert rep.verdict == VERDICT_REAL
    assert abs(rep.bpm - 72) < 3


def test_analyze_video_missing_file():
    with pytest.raises(VideoError):
        analyze_video("does/not/exist.mp4")


def test_analyze_video_corrupt_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(VideoError):
        analyze_video(bad)
