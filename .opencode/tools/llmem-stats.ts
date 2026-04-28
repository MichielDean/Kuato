import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";

/**
 * Show llmem memory statistics.
 *
 * Invokes `llmem stats` and returns the statistics output
 * (total, active, expired, by type). On error, returns an
 * error string starting with "Error:".
 */
export default tool({
  name: "llmem-stats",
  description:
    "Show llmem memory statistics (total, active, expired, by type). Returns formatted stats string.",
  args: {},
  execute: async (_args, context) => {
    const result = await runLlmem(["stats"], {
      worktree: context?.worktree,
    });

    if (result.exitCode !== 0) {
      return result.stdout;
    }

    return result.stdout;
  },
});