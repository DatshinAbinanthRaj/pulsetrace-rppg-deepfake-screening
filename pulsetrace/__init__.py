"""PulseTrace: rPPG-based deepfake screening.

Recovers the faint, heartbeat-driven colour change in facial skin
(remote photoplethysmography) and checks whether forehead and both
cheeks share one consistent pulse. Real faces usually do; many
synthetic or face-swapped videos do not.

This is a screening tool. It produces a score and a verdict with a
human-review band, not a forensic certainty.
"""

from pulsetrace.analysis import AnalysisConfig, AnalysisReport, analyze_frames, analyze_video

__all__ = ["AnalysisConfig", "AnalysisReport", "analyze_frames", "analyze_video"]
__version__ = "0.1.0"
