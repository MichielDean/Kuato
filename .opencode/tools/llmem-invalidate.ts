import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";

/**
 * Invalidate an llmem memory by ID.
 *
 * Invokes `llmem invalidate <ID> [--reason REASON]` and returns
 * confirmation with the memory ID. On error (not found, CLI failure),
 * returns a string starting with "Error:".
 */
export default tool({
  name: "llmem-invalidate",
  description:
    "Invalidate an llmem memory by ID. Returns confirmation string on success, or error string if memory not found.",
  args: {
    id: tool.schema.string().describe(
      "Memory ID to invalidate"
    ),
    reason: tool.schema.string().optional().describe(
      "Reason for invalidation"
    ),
  },
  execute: async (args, context) => {
    // Build optional flags before the -- separator so argparse
    // recognises them. The -- separator prevents the positional
    // ID from being parsed as a flag (argument injection).
    const cmdArgs: string[] = ["invalidate"];

    if (args.reason) {
      cmdArgs.push("--reason", args.reason);
    }
    cmdArgs.push("--", args.id);

    const result = await runLlmem(cmdArgs, {
      worktree: context?.worktree,
    });

    if (result.exitCode !== 0) {
      return result.stdout;
    }

    return result.stdout;
  },
});