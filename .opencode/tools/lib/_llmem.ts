/**
 * Default timeout for llmem subprocesses (milliseconds).
 * Prevents indefinite hangs on unresponsive processes.
 */
const DEFAULT_TIMEOUT_MS = 60000;

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
 * @param options - Optional configuration: json appends --json, worktree sets the cwd, timeout sets subprocess timeout in ms (default 60000).
 * @returns A promise resolving to { stdout, exitCode }. On success exitCode is 0.
 */
export async function runLlmem(
  args: string[],
  options?: { json?: boolean; worktree?: string; timeout?: number }
): Promise<RunLlmemResult> {
  const cmdArgs = [...args];
  if (options?.json) {
    cmdArgs.push("--json");
  }

  // Prepend '--' to all llmem invocations to prevent argparse flag injection.
  // User-supplied arguments (query strings, content, IDs) could otherwise
  // be interpreted as CLI flags (e.g. --help, --force).
  const cmd = ["llmem", "--", ...cmdArgs];
  const timeoutMs = options?.timeout ?? DEFAULT_TIMEOUT_MS;

  try {
    const proc = Bun.spawn(cmd, {
      cwd: options?.worktree,
      stdout: "pipe",
      stderr: "pipe",
    });

    // Race the process exit against a timeout to prevent indefinite hangs.
    const [exitCode, stdout, stderr] = await Promise.all([
      Promise.race([
        proc.exited,
        new Promise<number>((_, reject) =>
          setTimeout(() => {
            proc.kill();
            reject(new Error(`llmem subprocess timed out after ${timeoutMs}ms`));
          }, timeoutMs)
        ),
      ]),
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
    if (message.includes("timed out")) {
      return {
        stdout: `Error: llmem ${args[0] || ""} timed out after ${timeoutMs}ms`,
        exitCode: 124,
      };
    }
    return {
      stdout: `Error: llmem invocation failed: ${message}`,
      exitCode: 1,
    };
  }
}