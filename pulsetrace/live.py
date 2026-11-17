"""Live webcam mode: rolling 10 s window, re-analysed every second, overlay drawn on the feed.

Needs a local camera and a GUI-capable OpenCV build (`pip install opencv-python`,
not the headless wheel). Press q to quit.
"""

from __future__ import annotations

import collections
import time

import cv2

from pulsetrace.analysis import AnalysisConfig, analyze_frames, draw_rois
from pulsetrace.face import FaceTracker

COLOURS = {
    "LIKELY_REAL": (80, 200, 80),
    "LIKELY_FAKE": (60, 60, 230),
    "NEEDS_REVIEW": (0, 190, 255),
    "INCONCLUSIVE": (180, 180, 180),
}


def run_live(camera: int = 0, method: str = "pos", window_s: float = 10.0) -> int:  # pragma: no cover
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    buf: collections.deque = collections.deque(maxlen=int(window_s * fps))
    tracker = FaceTracker()
    cfg = AnalysisConfig(method=method, min_seconds=window_s * 0.9)
    status, last = "Collecting...", 0.0
    colour = COLOURS["INCONCLUSIVE"]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) == buf.maxlen and time.time() - last > 1.0:
            rep = analyze_frames(list(buf), fps, cfg)
            bpm = f"{rep.bpm:.0f} BPM" if rep.bpm else "-- BPM"
            score = f"{rep.score:.2f}" if rep.score is not None else "n/a"
            status, colour, last = f"{rep.verdict}  {bpm}  score {score}", COLOURS[rep.verdict], time.time()
        view = draw_rois(frame, tracker.update(frame))
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(view, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
        cv2.imshow("PulseTrace live (q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0
