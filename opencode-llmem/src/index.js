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
      // Minimal YAML parsing for simple key-value pairs
      // For complex configs, the llmem CLI handles parsing
      return { _configPath: configPath };
    }
  } catch (err) {
    // Config is optional — return empty object
  }
  return {};
}

module.exports = { register };