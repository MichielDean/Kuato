/**
 * Shared utilities for session hook handlers.
 *
 * Provides common helper functions used by multiple hook files
 * to avoid duplication.
 */

var path = require('path');
var os = require('os');

// Minimum interval between spawned llmem processes (milliseconds).
// Prevents process-spawn flooding DoS if hooks fire rapidly.
var _MIN_PROCESS_INTERVAL_MS = 5000;

// Track the last time a process was spawned
var _lastProcessTime = 0;

/**
 * Check whether enough time has passed since the last process spawn
 * to allow a new one. Prevents process-spawn flooding.
 *
 * @returns {boolean} True if enough time has passed, false if rate-limited.
 */
function canSpawnProcess() {
  var now = Date.now();
  if (now - _lastProcessTime < _MIN_PROCESS_INTERVAL_MS) {
    return false;
  }
  _lastProcessTime = now;
  return true;
}

/**
 * Validate that a session ID is safe to use in filesystem paths.
 *
 * Rejects session IDs that contain path separators or traversal
 * sequences, which could allow writing files outside the intended
 * context directory.
 *
 * @param {string} sessionId - The session ID to validate.
 * @throws {Error} If sessionId is empty, contains '/', '\', or '..'.
 */
function validateSessionId(sessionId) {
  if (!sessionId) {
    throw new Error('opencode-llmem: session_id must not be empty');
  }
  if (sessionId.indexOf('/') !== -1) {
    throw new Error('opencode-llmem: session_id contains "/" (path traversal risk): ' + sessionId);
  }
  if (sessionId.indexOf('\\') !== -1) {
    throw new Error('opencode-llmem: session_id contains "\\" (path traversal risk): ' + sessionId);
  }
  if (sessionId.indexOf('..') !== -1) {
    throw new Error('opencode-llmem: session_id contains ".." (path traversal risk): ' + sessionId);
  }
  return sessionId;
}

/**
 * Resolve the context directory path from config.
 *
 * @param {object} config - The llmem configuration object.
 * @returns {string} Path to the context directory.
 */
function getContextDir(config) {
  if (config && config.opencode && config.opencode.context_dir) {
    return config.opencode.context_dir;
  }
  return path.join(
    process.env.LMEM_HOME || path.join(os.homedir(), '.config', 'llmem'),
    'context'
  );
}

module.exports = { getContextDir, validateSessionId, canSpawnProcess };