import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";

/**
 * Retrieve formatted context from llmem memories for a topic.
 *
 * Invokes `llmem search <query> --json --limit 20` and formats
 * results into a context block suitable for LLM injection,
 * truncated to the given character budget.
 * Returns 'No memories found.' sentinel if no results. Returns 'Error: ...' string
 * for non-array JSON or parse failures.
 */
export default tool({
  name: "llmem-context",
  description:
    "Retrieve formatted llmem memory context for a topic. Returns memories formatted for LLM context injection, limited by character budget.",
  args: {
    query: tool.schema.string().describe(
      "Topic or query to recall context for"
    ),
    budget: tool.schema.number().min(0).optional().describe(
      "Character budget for context (default: 4000)"
    ),
  },
  execute: async (args, context) => {
    const budget = args.budget ?? 4000;

    const cmdArgs: string[] = ["search", args.query, "--limit", "20"];

    const result = await runLlmem(cmdArgs, {
      json: true,
      worktree: context?.worktree,
    });

    if (result.exitCode !== 0) {
      return result.stdout;
    }

    // Parse JSON results and format as context
    let memories: Array<Record<string, unknown>>;
    try {
      memories = JSON.parse(result.stdout);
    } catch {
      const snippet = result.stdout.slice(0, 200);
      return `Error: failed to parse llmem JSON output: ${snippet}`;
    }

    if (!Array.isArray(memories)) {
      return `Error: failed to parse llmem JSON output: expected array, got ${typeof memories}`;
    }

    if (memories.length === 0) {
      return "No memories found.";
    }

    // Format each memory similar to Retriever.format_context()
    const lines: string[] = [];
    for (const m of memories) {
      const mType = String(m.type ?? "unknown");
      const content = String(m.content ?? "");
      const summary = m.summary ? String(m.summary) : null;
      let line = `- [${mType}] ${content}`;
      if (summary) {
        line += ` (summary: ${summary})`;
      }
      lines.push(line);
    }

    const contextBlock = lines.join("\n");
    // Use Array-style slice to avoid splitting surrogate pairs —
    // String.slice operates on UTF-16 code units, but [...str] splits
    // on Unicode code points, so [...str].slice(0,n).join('') is safe.
    // Guard against negative budget: Array.slice(0, -N) truncates from
    // the end, violating the character-budget contract. Clamp to 0.
    const safeBudget = Math.max(0, budget);
    return [...contextBlock].slice(0, safeBudget).join("");
  },
});