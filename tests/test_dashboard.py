import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_theme_has_readable_light_contrast():
    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (project_root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )
    theme = config["theme"]

    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))

    def contrast_ratio(foreground, background):
        light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
        return (light + 0.05) / (dark + 0.05)

    assert theme["base"] == "light"
    assert contrast_ratio(theme["textColor"], theme["backgroundColor"]) >= 4.5
    assert contrast_ratio(theme["textColor"], theme["secondaryBackgroundColor"]) >= 4.5
    assert contrast_ratio("#ffffff", theme["primaryColor"]) >= 4.5


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
        assert len(app.dataframe) == 3
        assert all("Model" in dataframe.value.columns for dataframe in app.dataframe)
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
