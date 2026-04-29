const LLMEM = "llmem"
const fs = require("fs")
const child_process = require("child_process")

function log(msg) {
  try { fs.appendFileSync("/tmp/opencode-llmem-plugin.log", new Date().toISOString() + " " + msg + "\n") } catch {}
}

function run(cmd, args) {
  try {
    return child_process.execFileSync(cmd, args, { encoding: "utf8", timeout: 30000 })
  } catch (e) {
    log("exec error: " + (e.message || e))
    return ""
  }
}

log("plugin loaded")

export const LLMemPlugin = async ({ $, client }) => {
  log("plugin initialized")
  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        log("session.created: injecting memory context")
        const context = run(LLMEM, ["context", "session start"])
        const stats = run(LLMEM, ["stats"])
        if (context) {
          try { await client.app.log({ body: { service: "llmem", level: "info", message: context } }) } catch {}
          log("context injected (" + context.length + " chars)")
        }
        if (stats) {
          try { await client.app.log({ body: { service: "llmem", level: "info", message: stats } }) } catch {}
          log("stats injected")
        }
      }
      if (event.type === "session.idle") {
        const sessionId = event.sessionId || event.properties?.sessionId || ""
        log("session.idle: " + sessionId)
        run(LLMEM, ["hook", "idle", sessionId])
      }
    },
    "experimental.session.compacting": async (input, output) => {
      const result = run(LLMEM, ["search", "recent decisions important facts", "--limit", "10"])
      if (result) {
        output.context.push("## LLMem Memory Context\n\n" + result)
      }
    },
  }
}