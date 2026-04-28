import { tool } from "@opencode-ai/plugin";
import { runLlmem } from "./lib/_llmem";
import path from "path";

/**
 * Resolve a user-supplied path relative to a base directory, rejecting
 * path-traversal attempts (e.g. "../../etc/passwd").
 *
 * - Absolute paths are resolved as-is and checked against the base.
 * - Relative paths are joined to base, then resolved.
 * - If the resolved path escapes the base directory, returns null.
 *
 * @param userPath - The path supplied by the user (file or directory).
 * @param base - The base directory to resolve against (e.g. context.directory).
 * @returns The safe resolved path, or null if traversal is detected.
 */
function resolveContainedPath(userPath: string, base: string): string | null {
  // Normalize base to an absolute path so that startsWith containment
  // checks work reliably. Without this, a relative base like "." produces
  // baseNorm "./" — which no absolute resolved path starts with, making
  // all file/directory arguments incorrectly rejected.
  const normalizedBase = path.resolve(base);
  const resolved = path.resolve(normalizedBase, userPath);
  // Ensure the resolved path is within the base directory.
  // Both must end with separator for prefix matching to avoid
  // /foo/bar matching /foo/barbaz.
  const baseNorm = normalizedBase.endsWith(path.sep) ? normalizedBase : normalizedBase + path.sep;
  if (resolved !== normalizedBase && !resolved.startsWith(baseNorm)) {
    return null;
  }
  return resolved;
}

/**
 * Run the llmem extraction hook.
 *
 * Invokes `llmem hook [--force] [--no-embed] [--no-introspect] [--file PATH] [--directory PATH]`.
 * File and directory paths are resolved relative to context.directory for
 * relative paths. Paths containing ".." that escape context.directory are
 * rejected to prevent path traversal.
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
    const baseDir = context?.directory ?? context?.worktree ?? ".";
    if (args.file) {
      const resolved = resolveContainedPath(args.file, baseDir);
      if (resolved === null) {
        return `Error: llmem hook: file path "${args.file}" escapes directory scope`;
      }
      cmdArgs.push("--file", resolved);
    }
    if (args.directory) {
      const resolved = resolveContainedPath(args.directory, baseDir);
      if (resolved === null) {
        return `Error: llmem hook: directory path "${args.directory}" escapes directory scope`;
      }
      cmdArgs.push("--directory", resolved);
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