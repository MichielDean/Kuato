/**
 * Shared utilities for session hook handlers.
 *
 * Provides common helper functions used by multiple hook files
 * to avoid duplication.
 */

var path = require('path');
var os = require('os');

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

module.exports = { getContextDir };