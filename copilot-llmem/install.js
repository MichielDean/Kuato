#!/usr/bin/env node

/**
 * copilot-llmem postinstall script.
 *
 * Copies skill directories from the shared repo root into the user's
 * ~/.agents/skills/ directory so they are available to the Copilot CLI
 * plugin at runtime.
 *
 * Uses only Node.js built-in modules: fs, path, os.
 */

var fs = require('fs');
var path = require('path');
var os = require('os');

var SKILLS_SRC_DIR = path.join(__dirname, '..', 'skills');
var AGENTS_SRC_DIR = path.join(__dirname, 'agents');
var TARGET_SKILLS_DIR = path.join(os.homedir(), '.agents', 'skills');
var TARGET_AGENTS_DIR = path.join(os.homedir(), '.agents', 'agents');

var EXPECTED_SKILLS = ['llmem', 'introspection', 'introspection-review-tracker'];

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
  // Verify the shared skills source directory exists
  if (!fs.existsSync(SKILLS_SRC_DIR)) {
    console.error('copilot-llmem: shared skills directory not found at ' + SKILLS_SRC_DIR);
    process.exit(1);
  }

  // Verify each expected skill exists
  for (var i = 0; i < EXPECTED_SKILLS.length; i++) {
    var skillDir = path.join(SKILLS_SRC_DIR, EXPECTED_SKILLS[i]);
    if (!fs.existsSync(skillDir)) {
      console.error('copilot-llmem: expected skill not found: ' + EXPECTED_SKILLS[i]);
      process.exit(1);
    }
  }

  // Create target skills directory
  try {
    fs.mkdirSync(TARGET_SKILLS_DIR, { recursive: true });
  } catch (err) {
    console.error('copilot-llmem: failed to create target skills directory ' + TARGET_SKILLS_DIR + ': ' + err.message);
    process.exit(1);
  }

  // Verify target skills directory is writable
  try {
    fs.accessSync(TARGET_SKILLS_DIR, fs.constants.W_OK);
  } catch (err) {
    console.error('copilot-llmem: target skills directory is not writable: ' + TARGET_SKILLS_DIR);
    process.exit(1);
  }

  // Copy each skill directory
  for (var s = 0; s < EXPECTED_SKILLS.length; s++) {
    var srcSkillPath = path.join(SKILLS_SRC_DIR, EXPECTED_SKILLS[s]);
    var destSkillPath = path.join(TARGET_SKILLS_DIR, EXPECTED_SKILLS[s]);
    try {
      copyDirRecursive(srcSkillPath, destSkillPath);
    } catch (err) {
      console.error('copilot-llmem: failed to copy skill ' + EXPECTED_SKILLS[s] + ': ' + err.message);
      process.exit(1);
    }
  }

  console.log('copilot-llmem: installed ' + EXPECTED_SKILLS.length + ' skills to ' + TARGET_SKILLS_DIR);

  // Copy agents directory if present
  if (fs.existsSync(AGENTS_SRC_DIR)) {
    try {
      fs.mkdirSync(TARGET_AGENTS_DIR, { recursive: true });
    } catch (err) {
      console.error('copilot-llmem: failed to create target agents directory ' + TARGET_AGENTS_DIR + ': ' + err.message);
      process.exit(1);
    }

    try {
      fs.accessSync(TARGET_AGENTS_DIR, fs.constants.W_OK);
    } catch (err) {
      console.error('copilot-llmem: target agents directory is not writable: ' + TARGET_AGENTS_DIR);
      process.exit(1);
    }

    copyDirRecursive(AGENTS_SRC_DIR, TARGET_AGENTS_DIR);
    console.log('copilot-llmem: installed agents to ' + TARGET_AGENTS_DIR);
  }
}

main();