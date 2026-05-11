const LLMEM = "/home/lobsterdog/.local/bin/llmem"
const fs = require("fs")
const child_process = require("child_process")
const path = require("path")

function log(msg) {
  try { fs.appendFileSync("/tmp/opencode-llmem-plugin.log", new Date().toISOString() + " " + msg + "\n") } catch {}
}

function run(cmd, args, timeout) {
  try {
    const result = child_process.execFileSync(cmd, args, { encoding: "utf8", timeout: timeout || 30000 })
    log("exec ok: " + cmd + " " + args.join(" ") + " (" + result.length + " chars)")
    return result
  } catch (e) {
    log("exec error: " + cmd + " " + args.join(" ") + " -> " + (e.message || e).substring(0, 200))
    return ""
  }
}

// Track modified files during this session
const modifiedFiles = new Set()
let trackReviewRan = false
let sessionStartTime = null
let currentSessionId = null

log("plugin loaded")

export const LLMemPlugin = async ({ $, client }) => {
  log("plugin initialized")
  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        const sessionInfo = event.properties?.info
        const sessionId = sessionInfo?.id || ""
        const directory = sessionInfo?.directory || ""
        sessionStartTime = new Date().toISOString()
        currentSessionId = sessionId || currentSessionId
        log("session.created: id=" + sessionId + " dir=" + directory)
        if (sessionId) {
          const context = run(LLMEM, ["context", "--session-id", sessionId])
          const stats = run(LLMEM, ["stats"])
          if (context) {
            try { await client.app.log({ body: { service: "llmem", level: "info", message: context } }) } catch {}
            log("context injected (" + context.length + " chars)")
          } else {
            log("context empty — on_created returned no content")
          }
          if (stats) {
            try { await client.app.log({ body: { service: "llmem", level: "info", message: stats } }) } catch {}
            log("stats injected")
          }
          // Surface behavioral procedures from dream REM phase
          const procedures = run(LLMEM, ["search", "dream_rem", "--type", "procedure", "--limit", "5"])
          if (procedures) {
            try { await client.app.log({ body: { service: "llmem", level: "info", message: "## Dream-Generated Procedures\n\n" + procedures } }) } catch {}
            log("dream procedures injected (" + procedures.length + " chars)")
          }
          // Surface recurring error patterns from self_assessments
          const behavioral = run(LLMEM, ["search", "Category:", "--type", "self_assessment", "--limit", "5"])
          if (behavioral) {
            try { await client.app.log({ body: { service: "llmem", level: "info", message: "## Recent Self-Assessments\n\n" + behavioral } }) } catch {}
            log("behavioral context injected (" + behavioral.length + " chars)")
          }

        }
      }
      if (event.type === "session.idle") {
        const sessionId = event.properties?.sessionID || currentSessionId || ""
        log("session.idle: " + sessionId + (currentSessionId ? " (cached)" : " (from event)"))
        if (sessionId) {
          run(LLMEM, ["hook", "--type", "idle", "--session-id", sessionId])
        } else {
          log("session.idle: no sessionID available")
        }
      }
      if (event.type === "session.ending" || event.type === "session.ended") {
        const sessionId = event.properties?.sessionID || event.properties?.info?.id || currentSessionId || ""
        log("session.ending: " + sessionId + (currentSessionId ? " (cached)" : " (from event)"))
        if (sessionId) {
          // Run extraction + introspection pass at session end
          run(LLMEM, ["hook", "--type", "ending", "--session-id", sessionId], 120000)
          log("session-end hook ending completed")
          // Run track-review if no review was done this session
          if (!trackReviewRan) {
            const filesList = Array.from(modifiedFiles).slice(0, 10).join(", ")
            run(LLMEM, ["track-review"])
            log("session-end track-review completed (no review was done this session)")
          }
        }
      }
    },
    "experimental.session.compacting": async (input, output) => {
      const sessionId = input.sessionID || currentSessionId || ""
      log("session.compacting: " + sessionId)
      const result = run(LLMEM, ["context", "--compacting", "--session-id", sessionId || "unknown"])
      if (result) {
        output.context.push("## LLMem Memory Context\n\n" + result)
      }
    },
    "experimental.chat.system.transform": async (input, output) => {
      const checklist = [
        "## Session-End Checklist (MANDATORY — enforced by plugin)",
        "Before declaring any task done, you MUST:",
        "1. Run critical-code-reviewer on changed files (or confirm no code changes were made)",
        "2. Run llmem track-review to persist findings (even clean reviews produce REVIEW_PASSED)",
        "3. Run test-and-verify to confirm changes work",
        "4. Commit and push all changes",
        "5. Record any skipped steps with llmem introspect --category MISSING_VERIFICATION",
      ].join("\n")
      output.system.push(checklist)
      log("session-end checklist injected into system prompt")
    },
    "tool.execute.after": async (input, output) => {
      // Track file modifications for the self-review enforcement.
      // When the agent writes files, we remember them so we can remind
      // about review at session end.
      const tool = input.tool || ""
      const args = input.args || {}
      // Track file writes from common tools
      if (tool === "write" || tool === "edit" || tool === "Write") {
        const filePath = args.file_path || args.path || args.filePath || ""
        if (filePath) {
          modifiedFiles.add(filePath)
          log("file modified: " + filePath + " (total: " + modifiedFiles.size + ")")
        }
      }
      // Track bash commands that modify files (git operations, etc.)
      if (tool === "bash") {
        const cmd = (args.command || "") + ""
        if (cmd.includes("git commit") || cmd.includes("git push")) {
          log("git operation detected: " + cmd.substring(0, 80))
        }
      }
      // Detect when critical-code-reviewer is invoked
      if (tool === "skill" && (args.name === "critical-code-reviewer" || args.skill === "critical-code-reviewer")) {
        trackReviewRan = true
        log("critical-code-reviewer detected — trackReviewRan = true")
      }
    },
    "command.execute.before": async (input, output) => {
      // Before git commits, check if review was done.
      // This doesn't block the commit, but logs a reminder.
      const command = input.command || ""
      if (command.includes("git commit") || command.includes("git push")) {
        if (modifiedFiles.size > 0 && !trackReviewRan) {
          const reminder = "REMINDER: " + modifiedFiles.size + " files were modified this session but no code review was run. Consider running critical-code-reviewer before committing."
          try {
            await client.app.log({ body: { service: "llmem", level: "warning", message: reminder } })
          } catch {}
          log("git commit reminder: files modified without review (" + modifiedFiles.size + " files)")
        }
      }
    },
  }
}