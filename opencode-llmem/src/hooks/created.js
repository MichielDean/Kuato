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
var utils = require('./utils');

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

  // Validate session_id to prevent path traversal attacks
  utils.validateSessionId(sessionId);

  // Rate limit: prevent process-spawn flooding
  if (!utils.canSpawnProcess()) {
    console.error('opencode-llmem: session.created hook rate-limited, skipping');
    return;
  }

  try {
    // Use execFileSync to prevent command injection via sessionId
    var result = child_process.execFileSync(
      'llmem',
      ['context', sessionId],
      { encoding: 'utf8', timeout: 30000 }
    );

    if (result && result.trim()) {
      var contextDir = utils.getContextDir(config);
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

module.exports = { handle };