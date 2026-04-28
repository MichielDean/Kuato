/**
 * session.idle hook — extract memories when a session goes idle.
 *
 * When an OpenCode session goes idle (debounced 30 seconds), this
 * hook triggers memory extraction and introspection via the llmem CLI.
 * The debounce ensures the extraction doesn't fire on every brief pause.
 */

var child_process = require('child_process');

var DEBOUNCE_MS = 30000; // 30 seconds

// Track last idle time per session for debouncing
var _lastIdleTime = {};

/**
 * Handle the session.idle event by extracting memories (debounced).
 *
 * @param {string} sessionId - The OpenCode session ID.
 * @param {object} config - The llmem configuration object.
 */
function handle(sessionId, config) {
  if (!sessionId) {
    return;
  }

  var now = Date.now();
  var lastTime = _lastIdleTime[sessionId] || 0;

  // Debounce: skip if called within 30 seconds for the same session
  if (now - lastTime < DEBOUNCE_MS) {
    return;
  }

  _lastIdleTime[sessionId] = now;

  try {
    // Use the llmem CLI to extract memories from the session transcript
    child_process.execSync(
      'llmem hook idle ' + sessionId,
      { encoding: 'utf8', timeout: 60000 }
    );
  } catch (err) {
    // Graceful degradation — don't crash on extraction failure
    console.error('opencode-llmem: session.idle hook failed: ' + err.message);
  }
}

module.exports = { handle };