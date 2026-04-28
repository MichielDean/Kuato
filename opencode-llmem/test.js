#!/usr/bin/env node

/**
 * opencode-llmem validation tests.
 *
 * Validates that the npm package structure is correct,
 * no forbidden references exist, and install.js works.
 *
 * Run via: npm test
 */

var fs = require('fs');
var path = require('path');
var os = require('os');
var { execSync } = require('child_process');

var ROOT_DIR = path.join(__dirname);
var SRC_DIR = path.join(ROOT_DIR, 'src');
var HOOKS_DIR = path.join(SRC_DIR, 'hooks');
var EXPECTED_HOOKS = ['created', 'idle', 'compacting'];
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

// ── Structure Validation Tests ──────────────────────────────────────────

function testPackageJsonExists() {
  var pkgPath = path.join(ROOT_DIR, 'package.json');
  assert(fs.existsSync(pkgPath), 'package.json exists');

  if (fs.existsSync(pkgPath)) {
    var pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    assert(pkg.name === 'opencode-llmem', 'package.json name is opencode-llmem');
    assert(pkg.files && pkg.files.indexOf('src/') !== -1, 'package.json files includes "src/"');
    assert(pkg.scripts && pkg.scripts.postinstall === 'node install.js', 'package.json has postinstall script');
  }
}

function testIndexJsExists() {
  var indexPath = path.join(SRC_DIR, 'index.js');
  assert(fs.existsSync(indexPath), 'src/index.js exists');
  if (fs.existsSync(indexPath)) {
    checkNoForbiddenRefs(indexPath, 'src/index.js');
  }
}

function testAllHookFilesExist() {
  for (var i = 0; i < EXPECTED_HOOKS.length; i++) {
    var hookName = EXPECTED_HOOKS[i];
    var hookPath = path.join(HOOKS_DIR, hookName + '.js');
    assert(fs.existsSync(hookPath), 'src/hooks/' + hookName + '.js exists');
    if (fs.existsSync(hookPath)) {
      checkNoForbiddenRefs(hookPath, 'src/hooks/' + hookName + '.js');
    }
  }
}

function testIndexExportsRegister() {
  var indexPath = path.join(SRC_DIR, 'index.js');
  if (!fs.existsSync(indexPath)) {
    assert(false, 'src/index.js exports register function: file missing');
    return;
  }
  var content = fs.readFileSync(indexPath, 'utf8');
  assert(content.indexOf('module.exports') !== -1 || content.indexOf('exports.register') !== -1,
    'src/index.js exports a register function');
}

function testInstallJsExists() {
  var installPath = path.join(ROOT_DIR, 'install.js');
  assert(fs.existsSync(installPath), 'install.js exists');
}

// ── Security & DRY Tests ────────────────────────────────────────────────

function testNoExecSyncInHookFiles() {
  // Hook files must use execFileSync, NOT execSync, to prevent command injection
  for (var i = 0; i < EXPECTED_HOOKS.length; i++) {
    var hookName = EXPECTED_HOOKS[i];
    var hookPath = path.join(HOOKS_DIR, hookName + '.js');
    if (!fs.existsSync(hookPath)) {
      assert(false, 'src/hooks/' + hookName + '.js: file missing');
      continue;
    }
    var content = fs.readFileSync(hookPath, 'utf8');
    assert(content.indexOf('execSync') === -1,
      'src/hooks/' + hookName + '.js: no execSync (use execFileSync to prevent injection)');
    assert(content.indexOf('execFileSync') !== -1,
      'src/hooks/' + hookName + '.js: uses execFileSync for safe argument passing');
  }
}

function testSharedUtilsModuleExists() {
  var utilsPath = path.join(HOOKS_DIR, 'utils.js');
  assert(fs.existsSync(utilsPath), 'src/hooks/utils.js exists');
  if (fs.existsSync(utilsPath)) {
    checkNoForbiddenRefs(utilsPath, 'src/hooks/utils.js');
    var content = fs.readFileSync(utilsPath, 'utf8');
    assert(content.indexOf('getContextDir') !== -1,
      'src/hooks/utils.js exports getContextDir function');
  }
}

function testNoDuplicateGetContextDir() {
  // _getContextDir should NOT exist in individual hook files (DRY)
  for (var i = 0; i < EXPECTED_HOOKS.length; i++) {
    var hookName = EXPECTED_HOOKS[i];
    var hookPath = path.join(HOOKS_DIR, hookName + '.js');
    if (!fs.existsSync(hookPath)) continue;
    var content = fs.readFileSync(hookPath, 'utf8');
    assert(content.indexOf('_getContextDir') === -1,
      'src/hooks/' + hookName + '.js: no _getContextDir (use utils.getContextDir instead)');
  }
}

function testHookFilesUseSharedUtils() {
  // created.js and compacting.js must require('./utils')
  var hooksUsingUtils = ['created', 'compacting'];
  for (var i = 0; i < hooksUsingUtils.length; i++) {
    var hookPath = path.join(HOOKS_DIR, hooksUsingUtils[i] + '.js');
    if (!fs.existsSync(hookPath)) {
      assert(false, 'src/hooks/' + hooksUsingUtils[i] + '.js: file missing');
      continue;
    }
    var content = fs.readFileSync(hookPath, 'utf8');
    assert(content.indexOf("require('./utils')") !== -1,
      'src/hooks/' + hooksUsingUtils[i] + '.js: imports shared utils module');
  }
}

// ── Integration Tests ───────────────────────────────────────────────────

function testInstallCreatesTargetDir() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llmem-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(ROOT_DIR, 'install.js'), {
      env: installEnv,
      cwd: ROOT_DIR,
      stdio: 'pipe'
    });
    var targetDir = path.join(fakeHome, '.agents', 'plugins', 'llmem');
    assert(fs.existsSync(targetDir), 'install.js creates ~/.agents/plugins/llmem/ directory');
  } catch (err) {
    assert(false, 'install.js creates target directory: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testInstallCopiesSourceFiles() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llmem-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(ROOT_DIR, 'install.js'), {
      env: installEnv,
      cwd: ROOT_DIR,
      stdio: 'pipe'
    });
    var targetDir = path.join(fakeHome, '.agents', 'plugins', 'llmem');
    var indexPath = path.join(targetDir, 'index.js');
    assert(fs.existsSync(indexPath), 'install.js copies index.js');
    // Check hook files
    for (var i = 0; i < EXPECTED_HOOKS.length; i++) {
      var hookFile = path.join(targetDir, 'hooks', EXPECTED_HOOKS[i] + '.js');
      assert(fs.existsSync(hookFile), 'install.js copies hooks/' + EXPECTED_HOOKS[i] + '.js');
    }
    // Check shared utils module
    var utilsFile = path.join(targetDir, 'hooks', 'utils.js');
    assert(fs.existsSync(utilsFile), 'install.js copies hooks/utils.js');
  } catch (err) {
    assert(false, 'install.js copies source files: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

// ── Main ─────────────────────────────────────────────────────────────────

console.log('\n=== opencode-llmem Validation Tests ===\n');

testPackageJsonExists();
console.log('');
testIndexJsExists();
testAllHookFilesExist();
testIndexExportsRegister();
console.log('');
testInstallJsExists();

console.log('\n=== opencode-llmem Security & DRY Tests ===\n');

testNoExecSyncInHookFiles();
testSharedUtilsModuleExists();
testNoDuplicateGetContextDir();
testHookFilesUseSharedUtils();

console.log('\n=== opencode-llmem Integration Tests ===\n');

testInstallCreatesTargetDir();
testInstallCopiesSourceFiles();

console.log('\n=== opencode-llmem Results ===\n');
console.log('Passed: ' + passes);
console.log('Failed: ' + failures);

if (failures > 0) {
  process.exit(1);
} else {
  console.log('\nAll tests passed!');
  process.exit(0);
}