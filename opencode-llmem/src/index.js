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

const path = require('path');
const fs = require('fs');
const os = require('os');

const created = require('./hooks/created');
const idle = require('./hooks/idle');
const compacting = require('./hooks/compacting');

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
  const config = _loadConfig();

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
    // Prevent prototype pollution: reject keys that could overwrite
    // Object.prototype properties (__proto__, constructor, prototype).
    if (_DANGEROUS_KEYS.has(key)) {
      continue;
    }
    var value = trimmed.substring(colonIdx + 1).trim();

    // Pop stack until we find the parent with less indentation
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }
    var parent = stack[stack.length - 1].obj;

    if (value === '') {
      // Nested section — push onto stack
      parent[key] = {};
      stack.push({ indent: indent, obj: parent[key] });
    } else {
      // Strip optional quotes from value
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

module.exports = { register, _parseSimpleYaml };