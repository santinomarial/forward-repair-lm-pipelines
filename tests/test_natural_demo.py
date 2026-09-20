from pathlib import Path

import dspy
import pytest
from streamlit.testing.v1 import AppTest

import natural_analysis as analysis
import natural_study as study
from test_natural_study import mocked_models, small_study  # noqa: F401


APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"


@pytest.fixture
def natural_app(small_study, mocked_models, monkeypatch):
    study.collect(small_study, phase="development", workers=1)
    study.fit(small_study)
    study.collect(small_study, phase="test", workers=1)
    analysis.create_report(small_study)
    monkeypatch.syspath_prepend(str(APP.parent))
    import natural_replay
    monkeypatch.setattr(natural_replay, "NATURAL_STUDY_DIR", small_study)
    natural_replay.snapshot.clear()
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("offline replay called provider"))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key="app_mode").set_value("Natural errors · free").run()
    assert not app.exception
    return app


def test_natural_replay_uses_frozen_policy_and_saved_alternatives(natural_app):
    app = natural_app
    assert app.title[0].value == "Does repair help without injected failures?"
    assert {m.label: m.value for m in app.metric}["New API calls"] == "0"
    selected = app.selectbox(key="natural_case").value
    app.selectbox(key=f"natural_action_{selected}").set_value("accept").run()
    assert not app.exception
    assert any("Saved alternative" in c.value for c in app.caption)
    assert len(app.get("download_button")) == 1
    next(x for x in app.selectbox if x.label == "Natural output").set_value("Harmed").run()
    assert any("No natural cases" in x.value for x in app.info)


def test_natural_replay_missing_artifacts_is_offline(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(APP.parent))
    import natural_replay
    monkeypatch.setattr(natural_replay, "NATURAL_STUDY_DIR", tmp_path)
    natural_replay.snapshot.clear()
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("missing results must not trigger inference"))
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.radio(key="app_mode").set_value("Natural errors · free").run()
    assert not app.exception
    assert any("could not be verified" in i.value for i in app.info)
