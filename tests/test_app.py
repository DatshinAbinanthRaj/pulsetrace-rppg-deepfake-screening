"""UI smoke test using Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.mark.parametrize("kind,expected", [("real", "LIKELY_REAL"), ("incoherent", "LIKELY_FAKE")])
def test_synthetic_demo_flow(kind, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("PULSETRACE_REVIEW_LOG", str(tmp_path / "reviews.csv"))
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    at.radio(key="src").set_value("Synthetic demo clip").run()
    at.selectbox(key="kind").set_value(kind).run()
    at.button(key="gen").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert expected.replace("_", " ") in html
    # human-review form writes a log row
    at.radio(key="review_label").set_value("fake" if expected == "LIKELY_FAKE" else "real").run()
    at.button[-1].click().run()
    log = (tmp_path / "reviews.csv").read_text().splitlines()
    assert len(log) == 2 and expected in log[1]
