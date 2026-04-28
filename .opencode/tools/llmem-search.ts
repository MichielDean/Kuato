import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";

/**
 * Search llmem memories using FTS5 full-text search.
 *
 * Invokes `llmem search <query> --json [--type TYPE] [--limit N]`
 * and returns the matching memory objects as a JSON string.
 * On error, returns a string starting with "Error:".
 */
export default tool({
  name: "llmem-search",
  description:
    "Search llmem memories using FTS5 full-text search. Returns matching memory objects as JSON. Filter by type or limit results.",
  args: {
    query: tool.schema.string().describe(
      "Search query for FTS5 full-text search"
    ),
    type: tool.schema.string().optional().describe(
      "Filter by memory type (fact, decision, preference, etc.)"
    ),
    limit: tool.schema.number().min(1).optional().describe(
      "Maximum number of results (default: 20)"
    ),
  },
  execute: async (args, context) => {
    const cmdArgs: string[] = ["search", args.query];

    if (args.type) {
      cmdArgs.push("--type", args.type);
    }
    if (args.limit !== undefined) {
      cmdArgs.push("--limit", String(args.limit));
    }

    const result = await runLlmem(cmdArgs, {
      json: true,
      worktree: context?.worktree,
    });

    if (result.exitCode !== 0) {
      return result.stdout;
    }

    // Validate that the output is valid JSON
    try {
      const parsed = JSON.parse(result.stdout);
      if (!Array.isArray(parsed)) {
        return `Error: failed to parse llmem JSON output: expected array, got ${typeof parsed}`;
      }
      return result.stdout;
    } catch {
      const snippet = result.stdout.slice(0, 200);
      return `Error: failed to parse llmem JSON output: ${snippet}`;
    }
  },
});