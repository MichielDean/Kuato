"""Tests for llmem.dream_report module — HTML generation and escaping."""

from pathlib import Path

from llmem.dream import DreamResult, RemPhaseResult, DeepPhaseResult, LightPhaseResult
from llmem.dream_report import generate_dream_report


class TestDreamReport_HtmlEscaping:
    """Test that themes are HTML-escaped in dream reports."""

    def test_theme_with_html_chars_is_escaped(self, tmp_path):
        """Themes containing HTML special characters are properly escaped."""
        result = DreamResult(
            rem=RemPhaseResult(
                total_memories=10,
                active_memories=8,
                themes=["<script>alert('xss')</script>", "normal theme"],
            )
        )
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        # The raw <script> tag must NOT appear in the output
        assert "<script>" not in html
        # The escaped version should appear
        assert "&lt;script&gt;" in html
        # Normal themes should still render
        assert "normal theme" in html

    def test_theme_with_ampersand_is_escaped(self, tmp_path):
        """Themes with ampersands are properly escaped."""
        result = DreamResult(
            rem=RemPhaseResult(
                total_memories=5,
                active_memories=4,
                themes=["A & B"],
            )
        )
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "A &amp; B" in html

    def test_no_themes_shows_placeholder(self, tmp_path):
        """When no themes, a placeholder message is shown."""
        result = DreamResult(
            rem=RemPhaseResult(
                total_memories=0,
                active_memories=0,
                themes=[],
            )
        )
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "No themes extracted" in html


class TestDreamReport_Sections:
    """Test that dream report sections render correctly."""

    def test_light_section(self, tmp_path):
        result = DreamResult(light=LightPhaseResult(duplicate_pairs=3))
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "Light Phase" in html
        assert "3" in html

    def test_deep_section(self, tmp_path):
        result = DreamResult(
            deep=DeepPhaseResult(
                decayed_count=2,
                boosted_count=1,
                promoted_count=3,
                invalidated_count=0,
                merged_count=1,
            )
        )
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "Deep Phase" in html
        assert "Decayed: 2" in html
        assert "Merged: 1" in html

    def test_rem_section(self, tmp_path):
        result = DreamResult(
            rem=RemPhaseResult(
                total_memories=50,
                active_memories=40,
                themes=["reliability", "performance"],
            )
        )
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "REM Phase" in html
        assert "Total memories: 50" in html
        assert "Active memories: 40" in html
        assert "reliability" in html

    def test_empty_result(self, tmp_path):
        result = DreamResult()
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        html = report_path.read_text()
        assert "LLMem Dream Report" in html
