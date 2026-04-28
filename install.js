#!/usr/bin/env node

/**
 * opencode-kuato postinstall script.
 * Copies skill directories from the npm package into ~/.agents/skills/
 * so OpenCode can discover them via its standard skill discovery paths.
 *
 * Uses only Node.js built-in modules: fs, path, os.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const SKILLS_DIR = path.join(__dirname, 'skills');
const TARGET_DIR = path.join(os.homedir(), '.agents', 'skills');

function copyDirRecursive(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function main() {
  // Verify the skills directory exists in the package
  if (!fs.existsSync(SKILLS_DIR)) {
    console.error('opencode-kuato: skills directory not found at ' + SKILLS_DIR);
    process.exit(1);
  }

  // Create target directory if it doesn't exist
  try {
    fs.mkdirSync(TARGET_DIR, { recursive: true });
  } catch (err) {
    console.error('opencode-kuato: failed to create target directory ' + TARGET_DIR + ': ' + err.message);
    process.exit(1);
  }

  // Verify target directory is writable
  try {
    fs.accessSync(TARGET_DIR, fs.constants.W_OK);
  } catch (err) {
    console.error('opencode-kuato: target directory is not writable: ' + TARGET_DIR);
    process.exit(1);
  }

  // Copy each skill directory
  const skillDirs = fs.readdirSync(SKILLS_DIR, { withFileTypes: true })
    .filter(function(entry) { return entry.isDirectory(); });

  for (const dir of skillDirs) {
    const srcPath = path.join(SKILLS_DIR, dir.name);
    const destPath = path.join(TARGET_DIR, dir.name);
    try {
      copyDirRecursive(srcPath, destPath);
    } catch (err) {
      console.error('opencode-kuato: failed to copy skill ' + dir.name + ': ' + err.message);
      process.exit(1);
    }
  }

  console.log('opencode-kuato: installed ' + skillDirs.length + ' skills to ' + TARGET_DIR);
}

main();