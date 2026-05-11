#!/usr/bin/env node

/**
 * llmem postinstall script.
 *
 * Deploys skills and platform-specific plugins for LLMem integration.
 *
 * 1. Copies skill directories to ~/.agents/skills/ (all platforms)
 * 2. Deploys the OpenCode plugin to ~/.config/opencode/plugins/ (if OpenCode detected)
 * 3. Deploys the Claude Code / Copilot CLI plugin to ~/.claude/plugins/ (if detected)
 *
 * Use --platform to force a specific platform: opencode, claude-code, copilot, both, none
 * Defaults to auto-detecting.
 *
 * Uses only Node.js built-in modules: fs, path, os, child_process.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const child_process = require('child_process');

const SKILLS_DIR = path.join(__dirname, 'skills');
const PLUGINS_DIR = path.join(__dirname, 'plugins');
const TARGET_SKILLS_DIR = path.join(os.homedir(), '.agents', 'skills');
const OPENCODE_PLUGIN_SRC = path.join(PLUGINS_DIR, 'opencode', 'llmem.js');
const OPENCODE_PLUGIN_DEST = path.join(os.homedir(), '.config', 'opencode', 'plugins', 'llmem.js');
const AGENT_PLUGIN_SRC = path.join(PLUGINS_DIR, 'agent');
const CLAUDE_PLUGIN_DEST = path.join(os.homedir(), '.claude', 'plugins', 'llmem');

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

function detectPlatforms() {
  var platforms = [];
  try {
    child_process.execFileSync('which', ['opencode'], { encoding: 'utf8', timeout: 5000 }).trim();
    platforms.push('opencode');
  } catch {}
  try {
    child_process.execFileSync('which', ['claude'], { encoding: 'utf8', timeout: 5000 }).trim();
    platforms.push('claude-code');
  } catch {}
  var home = os.homedir();
  if (fs.existsSync(path.join(home, '.config', 'opencode'))) {
    if (platforms.indexOf('opencode') === -1) platforms.push('opencode');
  }
  if (fs.existsSync(path.join(home, '.claude'))) {
    if (platforms.indexOf('claude-code') === -1) platforms.push('claude-code');
  }
  return platforms.length > 0 ? platforms : ['opencode'];
}

function installSkills() {
  if (!fs.existsSync(SKILLS_DIR)) {
    console.error('llmem: skills directory not found at ' + SKILLS_DIR);
    process.exit(1);
  }
  try {
    fs.mkdirSync(TARGET_SKILLS_DIR, { recursive: true });
  } catch (err) {
    console.error('llmem: failed to create target directory ' + TARGET_SKILLS_DIR + ': ' + err.message);
    process.exit(1);
  }
  try {
    fs.accessSync(TARGET_SKILLS_DIR, fs.constants.W_OK);
  } catch (err) {
    console.error('llmem: target directory is not writable: ' + TARGET_SKILLS_DIR);
    process.exit(1);
  }

  var skillDirs = fs.readdirSync(SKILLS_DIR, { withFileTypes: true })
    .filter(function(entry) { return entry.isDirectory(); });

  for (var i = 0; i < skillDirs.length; i++) {
    var dir = skillDirs[i];
    var srcPath = path.join(SKILLS_DIR, dir.name);
    var destPath = path.join(TARGET_SKILLS_DIR, dir.name);
    try {
      copyDirRecursive(srcPath, destPath);
    } catch (err) {
      console.error('llmem: failed to copy skill ' + dir.name + ': ' + err.message);
      process.exit(1);
    }
  }
  console.log('llmem: installed ' + skillDirs.length + ' skills to ' + TARGET_SKILLS_DIR);
}

function installOpenCodePlugin() {
  if (!fs.existsSync(OPENCODE_PLUGIN_SRC)) {
    console.error('llmem: OpenCode plugin not found at ' + OPENCODE_PLUGIN_SRC);
    return;
  }
  var pluginDir = path.dirname(OPENCODE_PLUGIN_DEST);
  try {
    fs.mkdirSync(pluginDir, { recursive: true });
  } catch (err) {
    if (err.code !== 'EEXIST') {
      console.error('llmem: failed to create OpenCode plugin directory: ' + err.message);
      return;
    }
  }
  try {
    fs.copyFileSync(OPENCODE_PLUGIN_SRC, OPENCODE_PLUGIN_DEST);
    console.log('llmem: installed OpenCode plugin to ' + OPENCODE_PLUGIN_DEST);
  } catch (err) {
    console.error('llmem: failed to install OpenCode plugin: ' + err.message);
  }
}

function installAgentPlugin() {
  if (!fs.existsSync(AGENT_PLUGIN_SRC)) {
    console.error('llmem: agent plugin not found at ' + AGENT_PLUGIN_SRC);
    return;
  }
  var claudeDest = CLAUDE_PLUGIN_DEST;
  try {
    copyDirRecursive(AGENT_PLUGIN_SRC, claudeDest);
    console.log('llmem: installed Claude Code / Copilot CLI plugin to ' + claudeDest);
  } catch (err) {
    console.error('llmem: failed to install agent plugin: ' + err.message);
  }
}

function main() {
  var platformArg = null;
  for (var i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--platform' && i + 1 < process.argv.length) {
      platformArg = process.argv[i + 1];
    }
  }

  installSkills();

  var platforms;
  if (platformArg) {
    if (platformArg === 'none') {
      console.log('llmem: skipping plugin deployment (--platform none)');
      return;
    }
    if (platformArg === 'both') {
      platforms = ['opencode', 'claude-code'];
    } else {
      platforms = [platformArg];
    }
  } else {
    platforms = detectPlatforms();
  }

  for (var j = 0; j < platforms.length; j++) {
    var p = platforms[j];
    if (p === 'opencode') {
      installOpenCodePlugin();
    } else if (p === 'claude-code' || p === 'copilot') {
      installAgentPlugin();
    }
  }

  console.log('llmem: setup complete. Platforms: ' + platforms.join(', '));
  console.log('llmem: run "llmem init" to initialize config and database.');
}

main();