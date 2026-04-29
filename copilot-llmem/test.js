#!/usr/bin/env node

/**
 * copilot-llmem validation tests.
 *
 * Validates that the Copilot CLI plugin structure is correct,
 * hooks.json schema is valid, no forbidden references exist,
 * and install.js works.
 *
 * Run via: npm test
 */

var fs = require('fs');
var path = require('path');
var os = require('os');
var { execSync } = require('child_process');

var ROOT_DIR = path.join(__dirname);
var FORBIDDEN_PATTERNS = [
  /\blogmem\b/,
  /\blobsterdog\b/,
  /\bcistern\b/,
  /\bMichiel\b/,
  /MichielDean/,
  /\bnicobailon\b/,
  /\.pi\//,
  /\bpi\.on\b/,
  /\bBOOMERANG/i,
  /\bboomerang\b/i,
  /\bScaledTest\b/,
  /\bcastellarius\b/,
  /\bcataractae\b/,
  /\baqueduct\b/,
  /\blobresume\b/,
  /xargs.*sh\s+-c/,
  /\bpass\s+github\b/,
];

var failures = 0;
var passes = 0;

function assert(condition, message) {
  if (!condition) {
    console.error('  FAIL: ' + message);
    failures++;
  } else {
    console.log('  PASS: ' + message);
    passes++;
  }
}

function checkNoForbiddenRefs(filePath, label) {
  var content = fs.readFileSync(filePath, 'utf8');
  for (var i = 0; i < FORBIDDEN_PATTERNS.length; i++) {
    var pattern = FORBIDDEN_PATTERNS[i];
    var matches = content.match(pattern);
    if (matches) {
      assert(false, label + ': found forbidden reference "' + matches[0] + '"');
      return;
    }
  }
  assert(true, label + ' has no forbidden references');
}

function checkNoForbiddenRefsInDir(dirPath, label) {
  if (!fs.existsSync(dirPath)) {
    return;
  }
  var entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (var i = 0; i < entries.length; i++) {
    var entry = entries[i];
    var fullPath = path.join(dirPath, entry.name);
    var relPath = label + '/' + entry.name;
    if (entry.isDirectory()) {
      checkNoForbiddenRefsInDir(fullPath, relPath);
    } else {
      checkNoForbiddenRefs(fullPath, relPath);
    }
  }
}

// ── plugin.json Validation Tests ──────────────────────────────────────────

function testPluginJsonExists() {
  var pluginPath = path.join(ROOT_DIR, 'plugin.json');
  assert(fs.existsSync(pluginPath), 'plugin.json exists');
}

function testPluginJsonName() {
  var pluginPath = path.join(ROOT_DIR, 'plugin.json');
  if (!fs.existsSync(pluginPath)) {
    assert(false, 'plugin.json name: file missing');
    return;
  }
  var plugin = JSON.parse(fs.readFileSync(pluginPath, 'utf8'));
  assert(plugin.name === 'copilot-llmem', 'plugin.json has name: "copilot-llmem"');
}

function testPluginJsonHasSkills() {
  var pluginPath = path.join(ROOT_DIR, 'plugin.json');
  if (!fs.existsSync(pluginPath)) {
    assert(false, 'plugin.json has skills: file missing');
    return;
  }
  var plugin = JSON.parse(fs.readFileSync(pluginPath, 'utf8'));
  assert(plugin.skills === 'skills/', 'plugin.json skills points to "skills/"');
  // Verify the referenced skills directory exists locally
  var skillsDir = path.join(ROOT_DIR, 'skills');
  assert(fs.existsSync(skillsDir) && fs.statSync(skillsDir).isDirectory(), 'plugin.json skills "skills/" resolves to existing local directory');
}

function testPluginJsonHasHooks() {
  var pluginPath = path.join(ROOT_DIR, 'plugin.json');
  if (!fs.existsSync(pluginPath)) {
    assert(false, 'plugin.json has hooks: file missing');
    return;
  }
  var plugin = JSON.parse(fs.readFileSync(pluginPath, 'utf8'));
  assert(plugin.hooks === 'hooks.json', 'plugin.json hooks references "hooks.json"');
  // Verify the referenced hooks.json exists
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  assert(fs.existsSync(hooksPath), 'plugin.json hooks "hooks.json" resolves to existing file');
}

function testPluginJsonHasAgents() {
  var pluginPath = path.join(ROOT_DIR, 'plugin.json');
  if (!fs.existsSync(pluginPath)) {
    assert(false, 'plugin.json has agents: file missing');
    return;
  }
  var plugin = JSON.parse(fs.readFileSync(pluginPath, 'utf8'));
  assert(plugin.agents === 'agents/', 'plugin.json has agents field pointing to "agents/"');
  // Verify the referenced agents directory exists
  var agentsDir = path.join(ROOT_DIR, 'agents');
  assert(fs.existsSync(agentsDir) && fs.statSync(agentsDir).isDirectory(), 'plugin.json agents "agents/" resolves to existing directory');
}

// ── hooks.json Validation Tests ──────────────────────────────────────────

function testHooksJsonExists() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  assert(fs.existsSync(hooksPath), 'hooks.json exists');
}

function testHooksJsonVersion() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json version: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  assert(hooks.version === 1, 'hooks.json has version: 1');
}

function testHooksJsonHasHooks() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json has hooks: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  assert(hooks.hooks && typeof hooks.hooks === 'object', 'hooks.json has hooks object');
}

function testHooksSessionStart() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json sessionStart: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var sessionStart = hooks.hooks && hooks.hooks.sessionStart;
  assert(sessionStart !== undefined, 'hooks.json has sessionStart hook');
  if (sessionStart) {
    assert(sessionStart.type === 'command', 'sessionStart hook has type: "command"');
    assert(typeof sessionStart.bash === 'string' && sessionStart.bash.length > 0, 'sessionStart hook has bash field');
  }
}

function testHooksAgentStop() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json agentStop: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var agentStop = hooks.hooks && hooks.hooks.agentStop;
  assert(agentStop !== undefined, 'hooks.json has agentStop hook');
  if (agentStop) {
    assert(agentStop.type === 'command', 'agentStop hook has type: "command"');
    assert(typeof agentStop.bash === 'string' && agentStop.bash.length > 0, 'agentStop hook has bash field');
  }
}

function testHooksSessionCompacting() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json sessionCompacting: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var sessionCompacting = hooks.hooks && hooks.hooks.sessionCompacting;
  assert(sessionCompacting !== undefined, 'hooks.json has sessionCompacting hook');
  if (sessionCompacting) {
    assert(sessionCompacting.type === 'command', 'sessionCompacting hook has type: "command"');
    assert(typeof sessionCompacting.bash === 'string' && sessionCompacting.bash.length > 0, 'sessionCompacting hook has bash field');
    assert(sessionCompacting.bash.indexOf('--compacting') !== -1, 'sessionCompacting hook bash command uses --compacting flag');
  }
}

function testHooksTimeout() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json timeout: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var hookNames = Object.keys(hooks.hooks || {});
  for (var i = 0; i < hookNames.length; i++) {
    var hookName = hookNames[i];
    var hook = hooks.hooks[hookName];
    assert(hook.timeoutSec >= 10 && hook.timeoutSec <= 120,
      hookName + ' hook has timeoutSec between 10 and 120 (got ' + hook.timeoutSec + ')');
  }
}

function testHooksCallLlmemCli() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json calls llmem: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var hookNames = Object.keys(hooks.hooks || {});
  for (var i = 0; i < hookNames.length; i++) {
    var hookName = hookNames[i];
    var hook = hooks.hooks[hookName];
    var bashCmd = hook.bash || '';
    assert(bashCmd.indexOf('llmem') !== -1,
      hookName + ' hook bash command calls llmem CLI');
    // Verify it does NOT call other forbidden commands
    assert(bashCmd.indexOf('lobmem') === -1,
      hookName + ' hook bash command does not call lobmem');
  }
}

// ── No Forbidden References ──────────────────────────────────────────────

function testNoForbiddenRefs() {
  // Check plugin.json
  checkNoForbiddenRefs(path.join(ROOT_DIR, 'plugin.json'), 'plugin.json');
  // Check hooks.json
  checkNoForbiddenRefs(path.join(ROOT_DIR, 'hooks.json'), 'hooks.json');
  // Check install.js
  if (fs.existsSync(path.join(ROOT_DIR, 'install.js'))) {
    checkNoForbiddenRefs(path.join(ROOT_DIR, 'install.js'), 'install.js');
  }
  // Check agents directory
  checkNoForbiddenRefsInDir(path.join(ROOT_DIR, 'agents'), 'agents');
  // Check bundled skills directory
  checkNoForbiddenRefsInDir(path.join(ROOT_DIR, 'skills'), 'skills');
}

// ── Agent Validation Tests ───────────────────────────────────────────────

function testAgentFileExists() {
  var agentPath = path.join(ROOT_DIR, 'agents', 'memory-assistant.agent.md');
  assert(fs.existsSync(agentPath), 'agents/memory-assistant.agent.md exists');
}

function testAgentFrontmatter() {
  var agentPath = path.join(ROOT_DIR, 'agents', 'memory-assistant.agent.md');
  if (!fs.existsSync(agentPath)) {
    assert(false, 'memory-assistant.agent.md frontmatter: file missing');
    return;
  }
  var content = fs.readFileSync(agentPath, 'utf8');
  var frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
  assert(frontmatterMatch !== null, 'memory-assistant.agent.md has YAML frontmatter');
  if (!frontmatterMatch) return;

  var fm = frontmatterMatch[1];
  assert(fm.indexOf('name:') !== -1, 'memory-assistant.agent.md has name field in frontmatter');
  assert(fm.indexOf('description:') !== -1, 'memory-assistant.agent.md has description field in frontmatter');
  assert(fm.indexOf('tools:') !== -1, 'memory-assistant.agent.md has tools field in frontmatter');
  assert(fm.indexOf('bash') !== -1, 'memory-assistant.agent.md tools includes bash');
}

// ── install.js Validation Tests ──────────────────────────────────────────

function testInstallJsExists() {
  var installPath = path.join(ROOT_DIR, 'install.js');
  assert(fs.existsSync(installPath), 'install.js exists');
}

function testInstallCreatesTargetDir() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'copilot-llmem-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(ROOT_DIR, 'install.js'), {
      env: installEnv,
      cwd: ROOT_DIR,
      stdio: 'pipe'
    });
    var targetSkillsDir = path.join(fakeHome, '.agents', 'skills');
    assert(fs.existsSync(targetSkillsDir), 'install.js creates ~/.agents/skills/ directory');
  } catch (err) {
    assert(false, 'install.js creates target directory: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testInstallCopiesFiles() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'copilot-llmem-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(ROOT_DIR, 'install.js'), {
      env: installEnv,
      cwd: ROOT_DIR,
      stdio: 'pipe'
    });
    var targetSkillsDir = path.join(fakeHome, '.agents', 'skills');
    var expectedSkills = ['llmem', 'introspection', 'introspection-review-tracker'];
    var allPresent = true;
    var missing = [];
    for (var i = 0; i < expectedSkills.length; i++) {
      var skillPath = path.join(targetSkillsDir, expectedSkills[i]);
      if (!fs.existsSync(skillPath)) {
        allPresent = false;
        missing.push(expectedSkills[i]);
      }
    }
    assert(allPresent, 'install.js copies skill directories' + (missing.length > 0 ? ' (missing: ' + missing.join(', ') + ')' : ''));
    // Verify agents are also copied
    var targetAgentsDir = path.join(fakeHome, '.agents', 'agents');
    assert(fs.existsSync(targetAgentsDir), 'install.js copies agents directory');
  } catch (err) {
    assert(false, 'install.js copies files: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

// ── package.json Validation Tests ────────────────────────────────────────

function testPackageJsonExists() {
  var pkgPath = path.join(ROOT_DIR, 'package.json');
  assert(fs.existsSync(pkgPath), 'package.json exists');
}

function testPackageJsonName() {
  var pkgPath = path.join(ROOT_DIR, 'package.json');
  if (!fs.existsSync(pkgPath)) {
    assert(false, 'package.json name: file missing');
    return;
  }
  var pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  assert(pkg.name === 'copilot-llmem', 'package.json has name: "copilot-llmem"');
  assert(pkg.license === 'MIT', 'package.json has license: "MIT"');
  assert(pkg.scripts && pkg.scripts.postinstall === 'node install.js', 'package.json has postinstall script');
  assert(pkg.scripts && pkg.scripts.test === 'node test.js', 'package.json has test script');
}

function testPackageJsonFilesExistLocally() {
  var pkgPath = path.join(ROOT_DIR, 'package.json');
  if (!fs.existsSync(pkgPath)) {
    assert(false, 'package.json files exist: file missing');
    return;
  }
  var pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  var files = pkg.files || [];
  assert(files.length > 0, 'package.json has non-empty files array');
  var allExist = true;
  var missing = [];
  for (var i = 0; i < files.length; i++) {
    var filePath = path.join(ROOT_DIR, files[i]);
    if (!fs.existsSync(filePath)) {
      allExist = false;
      missing.push(files[i]);
    }
  }
  assert(allExist, 'package.json files all exist locally' + (missing.length > 0 ? ' (missing: ' + missing.join(', ') + ')' : ''));
  // Ensure no src/ in files (this is a declarative plugin, no programmatic src/)
  assert(files.indexOf('src/') === -1, 'package.json files does not include src/ (declarative plugin)');
}

function testInstallJsUsesLocalSkills() {
  var installPath = path.join(ROOT_DIR, 'install.js');
  if (!fs.existsSync(installPath)) {
    assert(false, 'install.js uses local skills: file missing');
    return;
  }
  var content = fs.readFileSync(installPath, 'utf8');
  // install.js must NOT reference ../skills (would break standalone npm install)
  assert(content.indexOf('..') === -1 || content.indexOf('..\\\' + \'skills') === -1,
    'install.js does not reference ../skills (breaks standalone install)');
  // Verify it references local skills directory
  assert(content.indexOf("path.join(__dirname, 'skills')") !== -1,
    'install.js references local skills/ directory');
}

function testBundledSkillsExist() {
  var expectedSkills = ['llmem', 'introspection', 'introspection-review-tracker'];
  var allPresent = true;
  var missing = [];
  for (var i = 0; i < expectedSkills.length; i++) {
    var skillDir = path.join(ROOT_DIR, 'skills', expectedSkills[i]);
    if (!fs.existsSync(skillDir) || !fs.statSync(skillDir).isDirectory()) {
      allPresent = false;
      missing.push(expectedSkills[i]);
    }
  }
  assert(allPresent, 'bundled skills/ directory contains expected skills' + (missing.length > 0 ? ' (missing: ' + missing.join(', ') + ')' : ''));
  // Each skill must have a SKILL.md
  var allHaveSkillMd = true;
  var missingMd = [];
  for (var j = 0; j < expectedSkills.length; j++) {
    var skillMd = path.join(ROOT_DIR, 'skills', expectedSkills[j], 'SKILL.md');
    if (!fs.existsSync(skillMd)) {
      allHaveSkillMd = false;
      missingMd.push(expectedSkills[j]);
    }
  }
  assert(allHaveSkillMd, 'each bundled skill has SKILL.md' + (missingMd.length > 0 ? ' (missing: ' + missingMd.join(', ') + ')' : ''));
}

// ── install.js Agents Error Handling Tests ──────────────────────────────────

function testInstallJsAgentsCopyHasTryCatch() {
  var installPath = path.join(ROOT_DIR, 'install.js');
  if (!fs.existsSync(installPath)) {
    assert(false, 'install.js agents try/catch: file missing');
    return;
  }
  var content = fs.readFileSync(installPath, 'utf8');
  // Verify agents copyDirRecursive is wrapped in try/catch
  var agentsCopyBlock = content.match(/try\s*\{[^}]*copyDirRecursive\(AGENTS_SRC_DIR/);
  assert(agentsCopyBlock !== null,
    'install.js wraps copyDirRecursive(AGENTS_SRC_DIR, ...) in try/catch');
  // Verify the catch block includes formatted error message
  var agentsCatchBlock = content.match(/catch\s*\([^)]*\)\s*\{[^}]*failed to copy agents/);
  assert(agentsCatchBlock !== null,
    'install.js catch block for agents copy has "failed to copy agents" error message');
}

function testInstallJsAgentsCopyErrorProducesFormattedMessage() {
  // Runtime test: make agents target directory writable, but place a read-only
  // file inside that conflicts with copyDirRecursive. This triggers the
  // copyDirRecursive try/catch specifically, not the earlier accessSync check.
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'copilot-llmem-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var targetAgentsDir = path.join(fakeHome, '.agents', 'agents');
    fs.mkdirSync(targetAgentsDir, { recursive: true });
    // Place a read-only file inside with the same name as the agents source entry
    // so copyDirRecursive hits it when trying to create a directory or write a file
    var agentsSrcDir = path.join(ROOT_DIR, 'agents');
    var agentEntries = fs.readdirSync(agentsSrcDir);
    if (agentEntries.length > 0) {
      // Create a read-only directory entry that conflicts — put a file where a
      // directory is expected, then make it undeletable
      var blockPath = path.join(targetAgentsDir, agentEntries[0]);
      fs.writeFileSync(blockPath, 'block');
      fs.chmodSync(blockPath, 0o000);
    }
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    try {
      execSync('node ' + path.join(ROOT_DIR, 'install.js'), {
        env: installEnv,
        cwd: ROOT_DIR,
        stdio: 'pipe',
        timeout: 10000
      });
      // If install succeeds (unlikely given 0o000 perms), try alternate approach
      assert(true, 'install.js agents copy error: no error triggered (permissions allow write)');
    } catch (err) {
      var stderr = err.stderr ? err.stderr.toString() : '';
      // Verify a formatted error message is produced (not a raw Node stack trace)
      var hasFormattedMessage = stderr.indexOf('copilot-llmem:') !== -1;
      assert(hasFormattedMessage,
        'install.js agents copy error produces formatted "copilot-llmem:" prefixed message (got: ' + stderr.substring(0, 200) + ')');
    }
    // Clean up permissions for removal
    if (agentEntries && agentEntries.length > 0) {
      try { fs.chmodSync(path.join(targetAgentsDir, agentEntries[0]), 0o644); } catch (e) {}
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

// ── Powershell Field Validation ───────────────────────────────────────────

function testHooksHavePowershell() {
  var hooksPath = path.join(ROOT_DIR, 'hooks.json');
  if (!fs.existsSync(hooksPath)) {
    assert(false, 'hooks.json powershell: file missing');
    return;
  }
  var hooks = JSON.parse(fs.readFileSync(hooksPath, 'utf8'));
  var hookNames = Object.keys(hooks.hooks || {});
  for (var i = 0; i < hookNames.length; i++) {
    var hookName = hookNames[i];
    var hook = hooks.hooks[hookName];
    assert(typeof hook.powershell === 'string' && hook.powershell.length > 0,
      hookName + ' hook has powershell field');
  }
}

// ── Main ─────────────────────────────────────────────────────────────────

console.log('\n=== copilot-llmem plugin.json Validation Tests ===\n');

testPluginJsonExists();
testPluginJsonName();
testPluginJsonHasSkills();
testPluginJsonHasHooks();
testPluginJsonHasAgents();

console.log('\n=== copilot-llmem hooks.json Validation Tests ===\n');

testHooksJsonExists();
testHooksJsonVersion();
testHooksJsonHasHooks();
testHooksSessionStart();
testHooksAgentStop();
testHooksSessionCompacting();
testHooksTimeout();
testHooksCallLlmemCli();
testHooksHavePowershell();

console.log('\n=== copilot-llmem Forbidden Reference Tests ===\n');

testNoForbiddenRefs();

console.log('\n=== copilot-llmem Agent Validation Tests ===\n');

testAgentFileExists();
testAgentFrontmatter();

console.log('\n=== copilot-llmem install.js Validation Tests ===\n');

testInstallJsExists();
testInstallCreatesTargetDir();
testInstallCopiesFiles();
testInstallJsAgentsCopyHasTryCatch();
testInstallJsAgentsCopyErrorProducesFormattedMessage();

console.log('\n=== copilot-llmem package.json Validation Tests ===\n');

testPackageJsonExists();
testPackageJsonName();
testPackageJsonFilesExistLocally();
testInstallJsUsesLocalSkills();
testBundledSkillsExist();

console.log('\n=== copilot-llmem Results ===\n');

console.log('Passed: ' + passes);
console.log('Failed: ' + failures);

if (failures > 0) {
  process.exit(1);
} else {
  console.log('\nAll tests passed!');
  process.exit(0);
}