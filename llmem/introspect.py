"""Targeted introspection — analyze failures and produce behavioral lessons.

Unlike the session-transcript introspection in hooks.py (which processes
full session transcripts at session end), targeted introspection is
triggered mid-session when something goes wrong. It takes focused context
about what failed, calls an LLM to analyze it, and stores structured
self_assessment or procedure memories.

Two modes:
- introspect_failure: "Something went wrong, analyze why" → self_assessment
- learn_lesson: "Wrong vs right, distill the lesson" → procedure
"""

import json
import logging

from .store import MemoryStore
from .extract import DEFAULT_MODEL, OLLAMA_BASE
from .taxonomy import SELF_ASSESSMENT_FIELDS, ERROR_TAXONOMY

log = logging.getLogger(__name__)

INTROSPECTION_FAILURE_SOURCE = "introspect"
INTROSPECTION_LESSON_SOURCE = "learn"


def _call_model(
    prompt: str, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE
) -> str | None:
    """Call the introspection model with a prompt and return the response text.

    Returns None if the model is unavailable or the call fails.
    """
    from .url_validate import validate_base_url, safe_urlopen
    import urllib.request

    try:
        base_url = validate_base_url(base_url, module="introspect")
    except ValueError as e:
        log.error("llmem: introspect: invalid base URL: %s", e)
        return None

    url = f"{base_url}/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with safe_urlopen(req, allow_remote=True, timeout=300) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as e:
        log.error("llmem: introspect: model call failed: %s", e)
        return None


def introspect_failure(
    store: MemoryStore,
    what_happened: str,
    category: str | None = None,
    context: str | None = None,
    caught_by: str | None = None,
    proposed_fix: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE,
) -> str | None:
    """Analyze a failure and store a self_assessment memory.

    If the introspection model is available, uses it to expand the bare
    description into a structured self-assessment. If the model is
    unavailable, stores a structured memory directly from the provided
    fields (graceful degradation — the lesson is still captured).

    Args:
        store: MemoryStore to save the memory into.
        what_happened: What went wrong.
        category: Error taxonomy category (e.g. ERROR_HANDLING).
        context: File, task, or situation where it happened.
        caught_by: How the issue was discovered.
        proposed_fix: What should change to prevent recurrence.
        model: Ollama model name for LLM-assisted introspection.
        base_url: Ollama base URL.

    Returns:
        The memory ID of the stored self_assessment, or None on failure.
    """
    from .hooks import IntrospectionAnalyzer

    if category and category not in ERROR_TAXONOMY:
        log.warning(
            "llmem: introspect: unknown category '%s', proceeding anyway", category
        )

    llm_response = None
    analyzer = IntrospectionAnalyzer(model=model, base_url=base_url)
    if analyzer.check_available():
        field_lines = "\n".join(
            f"  {name}: {desc}" for name, desc in SELF_ASSESSMENT_FIELDS
        )
        prompt = f"""Analyze this failure from a coding agent's session and produce a structured self-assessment.

The agent encountered a problem mid-session. Based on the context below, identify what went wrong, why it happened, whether it's a recurring pattern, and what procedural change would prevent it in the future.

Format each field on its own line as "Field: value":

{field_lines}

Failure context:
  What happened: {what_happened}"""

        if category:
            prompt += f"\n  Category: {category}"
        if context:
            prompt += f"\n  Context: {context}"
        if caught_by:
            prompt += f"\n  How caught: {caught_by}"
        if proposed_fix:
            prompt += f"\n  Proposed fix: {proposed_fix}"

        prompt += "\n\nProduce a structured self-assessment. Be specific about what went wrong and what should change. If the failure reveals a recurring pattern, say so."

        llm_response = _call_model(prompt, model, base_url)

    if llm_response:
        content = llm_response
    else:
        content_lines = []
        if category:
            content_lines.append(f"Category: {category}")
        if context:
            content_lines.append(f"Context: {context}")
        content_lines.append(f"What_happened: {what_happened}")
        content_lines.append(f"What_caught_it: {caught_by or 'mid-session introspection'}")
        if proposed_fix:
            content_lines.append(f"Proposed_update: {proposed_fix}")
        content_lines.append("Recurring: unknown")
        content = "\n".join(content_lines)

    mid = store.add(
        type="self_assessment",
        content=content,
        source=INTROSPECTION_FAILURE_SOURCE,
        confidence=0.9,
    )
    log.info("llmem: introspect: stored self_assessment %s", mid)
    return mid


def learn_lesson(
    store: MemoryStore,
    what_was_wrong: str,
    what_is_correct: str,
    context: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE,
) -> str | None:
    """Analyze a wrong→right correction and store a procedure memory.

    If the introspection model is available, uses it to distill the
    correction into a generalizable, actionable procedure. If the model
    is unavailable, stores the lesson directly (graceful degradation).

    Args:
        store: MemoryStore to save the memory into.
        what_was_wrong: What was done incorrectly.
        what_is_correct: The correct behavior.
        context: File, task, or situation where it happened.
        model: Ollama model name for LLM-assisted analysis.
        base_url: Ollama base URL.

    Returns:
        The memory ID of the stored procedure, or None on failure.
    """
    from .hooks import IntrospectionAnalyzer

    llm_response = None
    analyzer = IntrospectionAnalyzer(model=model, base_url=base_url)
    if analyzer.check_available():
        prompt = f"""A coding agent made a mistake and then corrected it. Distill the lesson into an actionable, generalizable procedure that will prevent this mistake in future sessions.

Be specific and practical. The procedure should be a rule the agent can follow — not vague advice.

What was WRONG:
{what_was_wrong}

What is CORRECT:
{what_is_correct}"""

        if context:
            prompt += f"\n\nContext: {context}"

        prompt += "\n\nWrite the lesson as a clear, actionable procedure. Start with the correct behavior, then explain what to avoid. Keep it under 200 words."

        llm_response = _call_model(prompt, model, base_url)

    if llm_response:
        content = llm_response
    else:
        content_lines = [f"WRONG: {what_was_wrong}", f"RIGHT: {what_is_correct}"]
        if context:
            content_lines.append(f"Context: {context}")
        content = "\n".join(content_lines)

    mid = store.add(
        type="procedure",
        content=content,
        source=INTROSPECTION_LESSON_SOURCE,
        confidence=0.85,
    )
    log.info("llmem: learn: stored procedure %s", mid)
    return mid