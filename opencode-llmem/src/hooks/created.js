/**
 * session.created hook — inject relevant memories into a new session.
 *
 * When a new OpenCode session starts, this hook queries the llmem
 * memory store for relevant memories and writes them into the
 * session's context directory. Uses the llmem CLI command to
 * perform the injection.
 */

var child_process = require('child_process');
var path = require('path');
var fs = require('fs');
var os = require('os');

/**
 * Handle the session.created event by injecting memory context.
 *
 * @param {string} sessionId - The OpenCode session ID.
 * @param {object} config - The llmem configuration object.
 */
function handle(sessionId, config) {
  if (!sessionId) {
    return;
  }

  try {
    // Use the llmem CLI to query and format context for this session
    var result = child_process.execSync(
      'llmem context ' + sessionId,
      { encoding: 'utf8', timeout: 30000 }
    );

    if (result && result.trim()) {
      var contextDir = _getContextDir(config);
      if (!fs.existsSync(contextDir)) {
        fs.mkdirSync(contextDir, { recursive: true });
      }
      var contextFile = path.join(contextDir, sessionId + '.md');
      fs.writeFileSync(contextFile, result);
    }
  } catch (err) {
    // Graceful degradation — don't crash the session on failure
    console.error('opencode-llmem: session.created hook failed: ' + err.message);
  }
}

/**
 * Resolve the context directory path from config.
 *
 * @param {object} config - The llmem configuration object.
 * @returns {string} Path to the context directory.
 */
function _getContextDir(config) {
  if (config && config.opencode && config.opencode.context_dir) {
    return config.opencode.context_dir;
  }
  return path.join(
    process.env.LMEM_HOME || path.join(os.homedir(), '.config', 'llmem'),
    'context'
  );
}

module.exports = { handle };