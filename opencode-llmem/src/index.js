/**
 * opencode-llmem — LLMem session hooks for OpenCode.
 *
 * Registers three session lifecycle hooks that integrate with the
 * llmem memory system:
 * - session.created: inject relevant memories when a session starts
 * - session.idle: extract memories when a session goes idle
 * - session.compacting: inject key memories during compaction
 *
 * Reads llmem configuration from the standard llmem config path.
 * Does not hardcode provider configuration.
 */

var path = require('path');
var fs = require('fs');
var os = require('os');

var created = require('./hooks/created');
var idle = require('./hooks/idle');
var compacting = require('./hooks/compacting');

var LLMEM = "llmem";
var child_process = require('child_process');

function log(msg) {
  try { fs.appendFileSync("/tmp/opencode-llmem-plugin.log", new Date().toISOString() + " " + msg + "\n") } catch {}
}

function run(cmd, args) {
  try {
    return child_process.execFileSync(cmd, args, { encoding: "utf8", timeout: 30000 });
  } catch (e) {
    log("exec error: " + (e.message || e));
    return "";
  }
}

/**
 * Register all three llmem session hooks with the OpenCode session.
 *
 * This function reads the llmem config file and wires up the hooks
 * to delegate to the llmem CLI commands. It does NOT hardcode
 * provider configuration — all settings come from the config file.
 *
 * @param {object} session - The OpenCode session object.
 */
function register(session) {
  var config = _loadConfig();

  if (session && session.on) {
    session.on('created', function(sessionId) {
      created.handle(sessionId, config);
    });
    session.on('idle', function(sessionId) {
      idle.handle(sessionId, config);
    });
    session.on('compacting', function(sessionId) {
      compacting.handle(sessionId, config);
    });
  }
}

/**
 * Load llmem configuration from the standard config path.
 *
 * Returns an empty object if the config file doesn't exist or
 * cannot be parsed.
 *
 * @returns {object} The llmem config, or empty object.
 */
function _loadConfig() {
  var configPath = path.join(
    process.env.LMEM_HOME || path.join(os.homedir(), '.config', 'llmem'),
    'config.yaml'
  );

  try {
    if (fs.existsSync(configPath)) {
      var content = fs.readFileSync(configPath, 'utf8');
      return _parseSimpleYaml(content);
    }
  } catch (err) {
    // Config is optional — return empty object
  }
  return {};
}

// Keys that could prototype-pollute the result object if assigned via YAML keys.
var _DANGEROUS_KEYS = new Set(["__proto__", "constructor", "prototype"]);

function _parseSimpleYaml(text) {
  var result = {};
  var stack = [{ indent: -1, obj: result }];
  var lines = text.split('\n');

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var trimmed = line.trim();
    if (!trimmed || trimmed[0] === '#') {
      continue;
    }
    var indent = line.length - line.trimStart().length;
    var colonIdx = trimmed.indexOf(':');
    if (colonIdx === -1) {
      continue;
    }
    var key = trimmed.substring(0, colonIdx).trim();
    if (_DANGEROUS_KEYS.has(key)) {
      continue;
    }
    var value = trimmed.substring(colonIdx + 1).trim();

    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }
    var parent = stack[stack.length - 1].obj;

    if (value === '') {
      parent[key] = {};
      stack.push({ indent: indent, obj: parent[key] });
    } else {
      if (
        (value[0] === '"' && value[value.length - 1] === '"') ||
        (value[0] === "'" && value[value.length - 1] === "'")
      ) {
        value = value.substring(1, value.length - 1);
      }
      parent[key] = value;
    }
  }
  return result;
}

/**
 * OpenCode plugin API — provides session event hooks and
 * compaction context injection via the llmem CLI.
 */
var LLMemPlugin = async function({ $, client }) {
  log("plugin initialized");
  return {
    event: async function({ event }) {
      if (event.type === "session.created") {
        log("session.created: injecting memory context");
        var context = run(LLMEM, ["context", "session start"]);
        var stats = run(LLMEM, ["stats"]);
        if (context) {
          try { await client.app.log({ body: { service: "llmem", level: "info", message: context } }) } catch {}
          log("context injected (" + context.length + " chars)");
        }
        if (stats) {
          try { await client.app.log({ body: { service: "llmem", level: "info", message: stats } }) } catch {}
          log("stats injected");
        }
      }
      if (event.type === "session.idle") {
        var sessionId = event.sessionId || (event.properties && event.properties.sessionId) || "";
        log("session.idle: " + sessionId);
        run(LLMEM, ["hook", "idle", sessionId]);
      }
    },
    "experimental.session.compacting": async function(input, output) {
      var result = run(LLMEM, ["search", "recent decisions important facts", "--limit", "10"]);
      if (result) {
        output.context.push("## LLMem Memory Context\n\n" + result);
      }
    },
  };
};

module.exports = { register, _parseSimpleYaml, LLMemPlugin };