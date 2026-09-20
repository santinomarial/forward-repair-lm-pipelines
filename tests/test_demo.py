"""Headless Streamlit integration tests. Provider calls are forbidden."""

from pathlib import Path

import dspy
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


@pytest.fixture
def app(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("demo test called a provider"))
    result = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not result.exception
    return result


def test_replay_default_and_recovery_harm_keep_filters(app):
    assert app.title[0].value == "When does repair actually help?"
    assert {m.label: m.value for m in app.metric}["New API calls"] == "0"
    assert not any(x.label == "OpenAI API key" for x in app.text_input)
    for view, count in (("Recovered", 57), ("Harmed", 4), ("Kept correct", 34), ("Missed recovery", 15)):
        element(app, "selectbox", "Show").select(view).run()
        assert not app.exception
        assert any(x.label == f"Case · {count} matching" for x in app.selectbox)


def test_search_empty_state_and_review_queue(app):
    element(app, "text_input", "Search question or ID").input("this does not match any question").run()
    assert not app.exception
    assert any("No cases match" in item.value for item in app.info)
    element(app, "text_input", "Search question or ID").input("").run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert any(x.label == "Case · 20 matching" for x in app.selectbox)


def test_counterfactual_does_not_change_frozen_review_target(app):
    original = next(c.value for c in app.caption if c.value.startswith("Review target:"))
    element(app, "selectbox", "Compare original with saved action").select("accept").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Recorded added calls"] == "0"
    assert next(c.value for c in app.caption if c.value.startswith("Review target:")) == original
    assert any("Counterfactual view" in c.value for c in app.caption)


def test_explicit_review_validation_and_session_export(app):
    element(app, "button", "Save review to this session").click().run()
    assert any("provide reviewer" in item.value for item in app.error)
    element(app, "text_input", "Reviewer identifier").input("test-fixture")
    element(app, "multiselect", "Initial answer labels").set_value(["uncertain"])
    element(app, "multiselect", "Selected answer labels").set_value(["uncertain"])
    element(app, "text_area", "Evidence-based rationale").input("UI test only; not a real human assessment.")
    element(app, "button", "Save review to this session").click().run()
    assert not app.exception
    assert len(app.session_state["human_reviews"]) == 1
    assert any("Review saved" in item.value for item in app.success)
    assert len(app.get("download_button")) == 2


def test_live_mode_is_explicit_and_does_not_call_provider_on_entry(app):
    app.radio(key="app_mode").set_value("Live playground · API calls").run()
    assert not app.exception
    assert any("incur model charges" in w.value for w in app.warning)
    assert element(app, "button", "Trigger corruption + localized repair").disabled
    app.radio(key="app_mode").set_value("Saved experiments · free").run()
    assert not app.exception


def test_case_deep_link_restored(monkeypatch):
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("replay must be offline"))
    app = AppTest.from_file(str(APP), default_timeout=20)
    case_id = "5a7166395542994082a3e814:natural"
    app.query_params["case"] = case_id
    app.run()
    assert not app.exception
    assert app.selectbox(key="replay_case").value == case_id
