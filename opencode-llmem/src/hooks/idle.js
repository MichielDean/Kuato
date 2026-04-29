/**
 * session.idle hook — extract memories when a session goes idle.
 *
 * When an OpenCode session goes idle (debounced 30 seconds), this
 * hook triggers memory extraction and introspection via the llmem CLI.
 * The debounce ensures the extraction doesn't fire on every brief pause.
 *
 * Rate limiting: concurrent invocations are capped to MAX_CONCURRENT
 * processes to prevent resource exhaustion during rapid idle events.
 */

var child_process = require('child_process');
var utils = require('./utils');

var DEBOUNCE_MS = 30000; // 30 seconds

// Maximum age for idle tracking entries before eviction (5 minutes)
var MAX_IDLE_AGE_MS = 300000;

// Maximum concurrent llmem processes to prevent resource exhaustion
var MAX_CONCURRENT = 3;

// Track last idle time per session for debouncing
var _lastIdleTime = {};

// Track currently running processes for rate limiting
var _activeCount = 0;

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
 * Rate-limits concurrent invocations to MAX_CONCURRENT to prevent
 * resource exhaustion during rapid idle events.
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

  // Rate limit: skip if too many concurrent processes are running
  if (_activeCount >= MAX_CONCURRENT) {
    console.warn(
      'opencode-llmem: session.idle hook skipped: ' +
      _activeCount + ' concurrent processes already running (max ' + MAX_CONCURRENT + ')'
    );
    return;
  }

  _lastIdleTime[sessionId] = now;

  // Evict stale entries to prevent unbounded growth
  _evictStaleEntries();

  _activeCount++;
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
  } finally {
    _activeCount--;
  }
}

module.exports = { handle };