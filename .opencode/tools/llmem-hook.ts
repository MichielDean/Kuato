import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";
import path from "path";

/**
 * Run the llmem extraction hook.
 *
 * Invokes `llmem hook [--force] [--no-embed] [--no-introspect] [--file PATH] [--directory PATH]`.
 * File and directory paths are resolved relative to context.directory for
 * relative paths, and context.worktree for project-root paths.
 * Returns extraction result string. On error, returns error string.
 */
export default tool({
  name: "llmem-hook",
  description:
    "Run the llmem extraction hook to process conversation transcripts. Supports force re-extraction, skipping embeddings, and targeting specific files or directories.",
  args: {
    force: tool.schema.boolean().optional().describe(
      "Force re-extraction of already-processed sessions"
    ),
    noEmbed: tool.schema.boolean().optional().describe(
      "Skip embedding generation (faster, no Ollama required)"
    ),
    noIntrospect: tool.schema.boolean().optional().describe(
      "Skip introspection for trivial sessions"
    ),
    file: tool.schema.string().optional().describe(
      "Path to a specific transcript file to process"
    ),
    directory: tool.schema.string().optional().describe(
      "Path to a directory of transcript files"
    ),
  },
  execute: async (args, context) => {
    const cmdArgs: string[] = ["hook"];

    if (args.force) {
      cmdArgs.push("--force");
    }
    if (args.noEmbed) {
      cmdArgs.push("--no-embed");
    }
    if (args.noIntrospect) {
      cmdArgs.push("--no-introspect");
    }
    if (args.file) {
      // Resolve relative paths using context.directory
      let filePath = args.file;
      if (!path.isAbsolute(filePath) && context?.directory) {
        filePath = path.join(context.directory, filePath);
      }
      cmdArgs.push("--file", filePath);
    }
    if (args.directory) {
      // Resolve relative paths using context.directory
      let dirPath = args.directory;
      if (!path.isAbsolute(dirPath) && context?.directory) {
        dirPath = path.join(context.directory, dirPath);
      }
      cmdArgs.push("--directory", dirPath);
    }

    const result = await runLlmem(cmdArgs, {
      worktree: context?.worktree,
    });

    if (result.exitCode !== 0) {
      return result.stdout;
    }

    return result.stdout;
  },
});