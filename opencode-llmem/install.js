#!/usr/bin/env node

/**
 * opencode-llmem postinstall script.
 *
 * Installs the llmem session hooks source into the OpenCode plugin
 * directory so OpenCode can discover and load them.
 *
 * Uses only Node.js built-in modules: fs, path, os.
 */

var fs = require('fs');
var path = require('path');
var os = require('os');

var SRC_DIR = path.join(__dirname, 'src');
var TARGET_DIR = path.join(os.homedir(), '.agents', 'plugins', 'llmem');

function copyDirRecursive(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  var entries = fs.readdirSync(src, { withFileTypes: true });
  for (var i = 0; i < entries.length; i++) {
    var entry = entries[i];
    var srcPath = path.join(src, entry.name);
    var destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function main() {
  // Verify the source directory exists
  if (!fs.existsSync(SRC_DIR)) {
    console.error('opencode-llmem: source directory not found at ' + SRC_DIR);
    process.exit(1);
  }

  // Create target directory if it doesn't exist
  try {
    fs.mkdirSync(TARGET_DIR, { recursive: true });
  } catch (err) {
    console.error('opencode-llmem: failed to create target directory ' + TARGET_DIR + ': ' + err.message);
    process.exit(1);
  }

  // Verify target directory is writable
  try {
    fs.accessSync(TARGET_DIR, fs.constants.W_OK);
  } catch (err) {
    console.error('opencode-llmem: target directory is not writable: ' + TARGET_DIR);
    process.exit(1);
  }

  // Copy source files to the plugin directory
  copyDirRecursive(SRC_DIR, TARGET_DIR);

  console.log('opencode-llmem: installed session hooks to ' + TARGET_DIR);
}

main();