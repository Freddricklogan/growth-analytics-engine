from pathlib import Path

from growth_engine.events import synthetic_dataset
from growth_engine.report import analyse, render_html, write_report


def test_analyse_and_render() -> None:
    ds = synthetic_dataset(n_users=1500, months=10, seed=5)
    a = analyse(ds)
    assert set(a.attribution) == {"last_touch", "first_touch", "linear", "markov"}
    assert set(a.truth_agreement) == set(a.attribution)
    assert a.blended.conversions == sum(1 for j in ds.journeys if j.converted)
    html = render_html(a, "https://example.invalid/pages/")
    assert "<!DOCTYPE html>" in html
    assert "Content-Security-Policy" in html
    assert 'id="report-data"' in html
    assert "Spearman" in html and "Qini" in html
    assert "onclick=" not in html and "style=" not in html


def test_write_report_creates_site(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    index = write_report(out, seed=9, n_users=800, months=8)
    assert index.exists()
    for name in (
        "src/exec-shell.css",
        "src/exec-shell.js",
        "src/report.js",
        "src/report.css",
        "report.json",
    ):
        assert (out / name).exists(), name
    text = index.read_text(encoding="utf-8")
    assert "synthetic event stream, seed 9" in text
    assert "800 prospects" in text
