from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_sections_and_interactive_flows():
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project_root / "app" / "main.py"), default_timeout=60).run()
    assert not app.exception

    for page in (
        "Shopping basket recommendations",
        "Frequently bought together",
        "Quantization comparison",
        "Dataset analytics",
        "Project overview",
    ):
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception, f"{page}: {[error.message for error in app.exception]}"

    baseline = project_root / "models" / "baseline_model.pth"
    dynamic = project_root / "models" / "dynamic_int8" / "model.pth"
    static = project_root / "models" / "static_int8" / "model.pth"
    if all(path.is_file() for path in (baseline, dynamic, static)):
        app.sidebar.radio[0].set_value("Shopping basket recommendations")
        app.run()
        app.text_input[0].set_value("Milk, Bread")
        app.checkbox[0].check()
        app.button[0].click()
        app.run()
        assert not app.exception
        assert len(app.dataframe) == 9
        assert app.success

        app.text_input[0].set_value("Unknown Product")
        app.button[0].click()
        app.run()
        assert any("Unknown Product" in error.value for error in app.error)

    if (project_root / "results" / "association_rules.csv").is_file():
        app.sidebar.radio[0].set_value("Frequently bought together")
        app.run()
        app.multiselect[0].set_value(["Milk"])
        app.run()
        assert not app.exception
        assert app.dataframe
