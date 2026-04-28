/**
 * session.compacting hook — inject key memories during compaction.
 *
 * When an OpenCode session begins compacting (summarizing its context
 * to reduce token count), this hook injects high-confidence key
 * memories (decisions, preferences, procedures, project state) to
 * preserve critical context that should survive compaction.
 */

var child_process = require('child_process');
var path = require('path');
var fs = require('fs');
var utils = require('./utils');

/**
 * Handle the session.compacting event by injecting key memories.
 *
 * @param {string} sessionId - The OpenCode session ID.
 * @param {object} config - The llmem configuration object.
 */
function handle(sessionId, config) {
  if (!sessionId) {
    return;
  }

  try {
    // Use execFileSync to prevent command injection via sessionId
    var result = child_process.execFileSync(
      'llmem',
      ['context', '--compacting', sessionId],
      { encoding: 'utf8', timeout: 30000 }
    );

    if (result && result.trim()) {
      var contextDir = utils.getContextDir(config);
      if (!fs.existsSync(contextDir)) {
        fs.mkdirSync(contextDir, { recursive: true });
      }
      var contextFile = path.join(contextDir, sessionId + '-compact.md');
      fs.writeFileSync(contextFile, result);
    }
  } catch (err) {
    // Graceful degradation — don't block compaction on failure
    console.error('opencode-llmem: session.compacting hook failed: ' + err.message);
  }
}

module.exports = { handle };