/**
 * session.idle hook — extract memories when a session goes idle.
 *
 * When an OpenCode session goes idle (debounced 30 seconds), this
 * hook triggers memory extraction and introspection via the llmem CLI.
 * The debounce ensures the extraction doesn't fire on every brief pause.
 */

var child_process = require('child_process');
var utils = require('./utils');

var DEBOUNCE_MS = 30000; // 30 seconds

// Maximum age for idle tracking entries before eviction (5 minutes)
var MAX_IDLE_AGE_MS = 300000;

// Track last idle time per session for debouncing
var _lastIdleTime = {};

/**
 * Evict stale entries from the idle time tracker.
 *
 * Removes entries older than MAX_IDLE_AGE_MS to prevent unbounded
 * memory growth in long-running processes.
 */
function _evictStaleEntries() {
  var now = Date.now();
  var keys = Object.keys(_lastIdleTime);
  for (var i = 0; i < keys.length; i++) {
    if (now - _lastIdleTime[keys[i]] > MAX_IDLE_AGE_MS) {
      delete _lastIdleTime[keys[i]];
    }
  }
}

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

  // Validate session_id to prevent path traversal attacks
  utils.validateSessionId(sessionId);

  var now = Date.now();
  var lastTime = _lastIdleTime[sessionId] || 0;

  // Debounce: skip if called within 30 seconds for the same session
  if (now - lastTime < DEBOUNCE_MS) {
    return;
  }

  _lastIdleTime[sessionId] = now;

  // Evict stale entries to prevent unbounded growth
  _evictStaleEntries();

  try {
    // Use execFileSync to prevent command injection via sessionId
    child_process.execFileSync(
      'llmem',
      ['hook', 'idle', sessionId],
      { encoding: 'utf8', timeout: 60000 }
    );
  } catch (err) {
    // Graceful degradation — don't crash on extraction failure
    console.error('opencode-llmem: session.idle hook failed: ' + err.message);
  }
}

module.exports = { handle };