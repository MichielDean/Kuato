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