"""Dream report HTML generator for llmem."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .dream import DreamResult

log = logging.getLogger(__name__)


def generate_dream_report(result: DreamResult, report_path: Path) -> None:
    """Generate an HTML dream report.

    Args:
        result: DreamResult from a dream run.
        report_path: Path to write the HTML report to.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build sections
    light_section = ""
    if result.light:
        light_section = f"""<section>
<h2>Light Phase</h2>
<p>Duplicate pairs found: {result.light.duplicate_pairs}</p>
</section>"""

    deep_section = ""
    if result.deep:
        deep_section = f"""<section>
<h2>Deep Phase</h2>
<ul>
<li>Decayed: {result.deep.decayed_count}</li>
<li>Boosted: {result.deep.boosted_count}</li>
<li>Promoted: {result.deep.promoted_count}</li>
<li>Invalidated: {result.deep.invalidated_count}</li>
<li>Merged: {result.deep.merged_count}</li>
</ul>
</section>"""

    rem_section = ""
    if result.rem:
        themes_html = (
            "".join(f"<li>{t}</li>" for t in result.rem.themes)
            if result.rem.themes
            else "<li>No themes extracted</li>"
        )
        rem_section = f"""<section>
<h2>REM Phase</h2>
<p>Total memories: {result.rem.total_memories}</p>
<p>Active memories: {result.rem.active_memories}</p>
<h3>Themes</h3>
<ul>{themes_html}</ul>
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLMem Dream Report — {timestamp}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    background: #0f1117;
    color: #e0e0e0;
}}
h1, h2, h3 {{ color: #90caf9; }}
section {{ margin: 2rem 0; }}
ul {{ list-style: disc; padding-left: 1.5rem; }}
</style>
</head>
<body>
<h1>LLMem Dream Report</h1>
<p>Generated: {timestamp}</p>
{light_section}
{deep_section}
{rem_section}
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html)
    log.info("llmem: dream_report: written to %s", report_path)
