/**
 * Error message returned when the llmem CLI is not found on PATH.
 */
export const LLMEM_NOT_FOUND = "Error: llmem CLI not found on PATH";

/**
 * Result of running an llmem CLI command.
 *
 * - On success: exitCode is 0 and stdout contains the command output.
 * - On failure: exitCode is non-zero and stdout contains an error string
 *   starting with "Error:".
 */
export interface RunLlmemResult {
  stdout: string;
  exitCode: number;
}

/**
 * Run an llmem CLI command as a subprocess.
 *
 * Constructs and executes `llmem <args...>` via Bun's shell. If the `json`
 * option is set, appends `--json` to the args. If `worktree` is provided,
 * the subprocess runs with its cwd set to that directory.
 *
 * Never throws — all errors are returned as error strings in stdout
 * with exitCode set to a non-zero value.
 *
 * @param args - Command-line arguments to pass to the llmem CLI (e.g. ["search", "my query", "--json"]).
 * @param options - Optional configuration: json appends --json, worktree sets the cwd.
 * @returns A promise resolving to { stdout, exitCode }. On success exitCode is 0.
 */
export async function runLlmem(
  args: string[],
  options?: { json?: boolean; worktree?: string }
): Promise<RunLlmemResult> {
  const cmdArgs = [...args];
  if (options?.json) {
    cmdArgs.push("--json");
  }

  const cmd = ["llmem", ...cmdArgs];

  try {
    const proc = Bun.spawn(cmd, {
      cwd: options?.worktree,
      stdout: "pipe",
      stderr: "pipe",
    });

    // Read stdout and stderr concurrently to avoid deadlock when the child
    // process fills the pipe buffer on one stream while we await the other.
    const [exitCode, stdout, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);

    if (exitCode !== 0) {
      const stderrSnippet = stderr.slice(0, 500).trim();
      return {
        stdout: `Error: llmem ${args[0] || ""} failed (exit code ${exitCode}): ${stderrSnippet}`,
        exitCode,
      };
    }

    return { stdout: stdout.trimEnd(), exitCode: 0 };
  } catch (err: unknown) {
    // Bun.spawn throws if the binary is not found
    const message = err instanceof Error ? err.message : String(err);
    if (
      message.includes("ENOENT") ||
      message.includes("not found") ||
      message.includes("No such file") ||
      message.includes("spawn") ||
      message.includes("Cannot find")
    ) {
      return { stdout: LLMEM_NOT_FOUND, exitCode: 127 };
    }
    return {
      stdout: `Error: llmem invocation failed: ${message}`,
      exitCode: 1,
    };
  }
}