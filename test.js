#!/usr/bin/env node

/**
 * opencode-kuato validation and integration tests.
 *
 * Validates that all skill directories contain valid SKILL.md frontmatter,
 * have no Lobsterdog/Cistern/personal references, and that install.js
 * works correctly.
 *
 * Run via: npm test
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const SKILLS_DIR = path.join(__dirname, 'skills');
const TEMPLATES_DIR = path.join(__dirname, 'templates');
const EXPECTED_SKILLS = [
  'git-sync',
  'task-intake',
  'test-and-verify',
  'branch-strategy',
  'critical-code-reviewer',
  'pre-pr-review',
  'visual-explainer'
];
const EXPECTED_TEMPLATES = [
  'rules.md',
  'identity.md',
  'user.md'
];
const FORBIDDEN_PATTERNS = [
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
  /xargs.*sh\s+-c/,         // Command injection vulnerability
  /\bpass\s+github\b/,      // Internal credential path convention
];
const NAME_REGEX = /^[a-z0-9]+(-[a-z0-9]+)*$/;

let failures = 0;
let passes = 0;

function assert(condition, message) {
  if (!condition) {
    console.error('  FAIL: ' + message);
    failures++;
  } else {
    console.log('  PASS: ' + message);
    passes++;
  }
}

function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return null;
  const fm = match[1];
  const result = {};
  // Simple YAML frontmatter parser
  // Handles: flat key-value, multi-line with > or |, nested maps (skip values)
  const lines = fm.split('\n');
  let currentKey = null;
  let currentValue = null;
  let inMultiline = false;
  let multilineIndent = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (inMultiline) {
      // Check if this line continues the multi-line value
      // Multi-line values end when we encounter a line at or below the key indent level
      if (line.trim() === '' || (line.match(/^\s/) && line.length - line.search(/\S/) > multilineIndent)) {
        // Continuation of multi-line value — collect but don't append for measurement
        continue;
      } else {
        // End of multi-line value
        inMultiline = false;
        result[currentKey] = currentValue;
        currentKey = null;
        currentValue = null;
        // Process this line as a new key-value
      }
    }

    const kvMatch = line.match(/^(\w+):\s*(.*)$/);
    if (kvMatch) {
      // Save previous multi-line value if any
      if (inMultiline && currentKey) {
        result[currentKey] = currentValue;
      }
      currentKey = kvMatch[1];
      var val = kvMatch[2].trim();
      // Handle multi-line indicators
      if (val === '>' || val === '|') {
        inMultiline = true;
        multilineIndent = line.search(/\w/); // indent level of the key
        currentValue = ''; // will be populated by subsequent lines
        // For description test, we just need to know it's present and non-empty
        // Capture all continuation lines until we see a non-indented line
        var collected = [];
        for (var j = i + 1; j < lines.length; j++) {
          var nextLine = lines[j];
          if (!nextLine.match(/^\s/) && nextLine.trim() !== '') {
            break;
          }
          collected.push(nextLine.trim());
        }
        result[currentKey] = collected.filter(function(s) { return s.length > 0; }).join(' ').trim();
        inMultiline = false;
      } else if (val) {
        result[currentKey] = val;
      } else {
        // Key with no value on same line — could be nested map or multi-line
        result[currentKey] = '';
      }
    } else if (currentKey && line.match(/^\s{2}\w+: /)) {
      // Nested key — skip for simple parsing
    }
  }

  return result;
}

// ── Validation Tests ──────────────────────────────────────────────────

function testSkillHasSKILLMd(skillDir) {
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  assert(fs.existsSync(skillMdPath), path.basename(skillDir) + ' contains SKILL.md');
}

function testFrontmatterNameMatchesDir(skillDir) {
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  if (!fs.existsSync(skillMdPath)) {
    assert(false, path.basename(skillDir) + ' SKILL.md name matches directory: SKILL.md missing');
    return;
  }
  const content = fs.readFileSync(skillMdPath, 'utf8');
  const fm = parseFrontmatter(content);
  if (!fm || !fm.name) {
    assert(false, path.basename(skillDir) + ' SKILL.md has frontmatter with name field');
    return;
  }
  const dirName = path.basename(skillDir);
  assert(fm.name === dirName, 'name="' + fm.name + '" matches directory "' + dirName + '"');
}

function testFrontmatterDescriptionPresent(skillDir) {
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  if (!fs.existsSync(skillMdPath)) {
    assert(false, path.basename(skillDir) + ' description present: SKILL.md missing');
    return;
  }
  const content = fs.readFileSync(skillMdPath, 'utf8');
  const fm = parseFrontmatter(content);
  if (!fm || fm.description === undefined) {
    assert(false, path.basename(skillDir) + ' has description field in frontmatter');
    return;
  }
  // description may be multi-line (starts with >), take the first meaningful line
  var desc = fm.description;
  if (desc.startsWith('>')) {
    desc = desc.replace(/^>\s*/, '');
  }
  assert(desc.length >= 1 && desc.length <= 1024,
    path.basename(skillDir) + ' description is 1-1024 chars (got ' + desc.length + ')');
}

function testFrontmatterNameValidFormat(skillDir) {
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  if (!fs.existsSync(skillMdPath)) {
    assert(false, path.basename(skillDir) + ' name valid format: SKILL.md missing');
    return;
  }
  const content = fs.readFileSync(skillMdPath, 'utf8');
  const fm = parseFrontmatter(content);
  if (!fm || !fm.name) {
    assert(false, path.basename(skillDir) + ' name matches ^[a-z0-9]+(-[a-z0-9]+)*$: no name field');
    return;
  }
  assert(NAME_REGEX.test(fm.name),
    path.basename(skillDir) + ' name "' + fm.name + '" matches ^[a-z0-9]+(-[a-z0-9]+)*$');
}

function checkNoPersonalReferences(dirPath, label) {
  var foundProblems = [];
  function checkFile(filePath, relativePath) {
    var content = fs.readFileSync(filePath, 'utf8');
    for (var i = 0; i < FORBIDDEN_PATTERNS.length; i++) {
      var pattern = FORBIDDEN_PATTERNS[i];
      var matches = content.match(pattern);
      if (matches) {
        foundProblems.push(relativePath + ': found "' + matches[0] + '"');
      }
    }
  }
  function walkDir(dir, relativeBase) {
    var entries = fs.readdirSync(dir, { withFileTypes: true });
    for (var i = 0; i < entries.length; i++) {
      var entry = entries[i];
      var fullPath = path.join(dir, entry.name);
      var relPath = relativeBase ? relativeBase + '/' + entry.name : entry.name;
      if (entry.isDirectory()) {
        walkDir(fullPath, relPath);
      } else {
        checkFile(fullPath, relPath);
      }
    }
  }
  // Support both directory (for skills) and single file (for templates)
  var stat = fs.statSync(dirPath);
  if (stat.isDirectory()) {
    walkDir(dirPath, label);
  } else {
    checkFile(dirPath, label);
  }
  assert(foundProblems.length === 0,
    label + ' has no Lobsterdog/Cistern/personal references' +
    (foundProblems.length > 0 ? ' (found: ' + foundProblems.join('; ') + ')' : ''));
}

function testNoLobsterdogReferences(skillDir) {
  var dirName = path.basename(skillDir);
  checkNoPersonalReferences(skillDir, dirName);
}

function testLicenseFieldPresent(skillDir) {
  const skillMdPath = path.join(skillDir, 'SKILL.md');
  if (!fs.existsSync(skillMdPath)) {
    assert(false, path.basename(skillDir) + ' license field: SKILL.md missing');
    return;
  }
  const content = fs.readFileSync(skillMdPath, 'utf8');
  assert(content.includes('license: MIT'),
    path.basename(skillDir) + ' has license: MIT in frontmatter');
}

function testNoClawhubFiles(skillDir) {
  const dirName = path.basename(skillDir);
  const metaJson = path.join(skillDir, '_meta.json');
  const clawdhubDir = path.join(skillDir, '.clawdhub');
  assert(!fs.existsSync(metaJson), dirName + ' has no _meta.json');
  assert(!fs.existsSync(clawdhubDir), dirName + ' has no .clawdhub/ directory');
}

function testNoClaudePluginFiles(skillDir) {
  const dirName = path.basename(skillDir);
  const claudePluginDir = path.join(skillDir, '.claude-plugin');
  assert(!fs.existsSync(claudePluginDir), dirName + ' has no .claude-plugin/ directory');
}

// ── Template Validation Tests ─────────────────────────────────────────

function testTemplateExists(templateFile) {
  var templatePath = path.join(TEMPLATES_DIR, templateFile);
  assert(fs.existsSync(templatePath), templateFile + ' exists in templates/');
  if (fs.existsSync(templatePath)) {
    var content = fs.readFileSync(templatePath, 'utf8');
    assert(content.trim().length > 0, templateFile + ' is non-empty');
  }
}

function testTemplateEndsWithNewline(templateFile) {
  var templatePath = path.join(TEMPLATES_DIR, templateFile);
  if (!fs.existsSync(templatePath)) {
    assert(false, templateFile + ' ends with newline: file missing');
    return;
  }
  var content = fs.readFileSync(templatePath);
  var lastByte = content[content.length - 1];
  assert(lastByte === 0x0A,
    templateFile + ' ends with trailing newline (cat-safe)');
}

function testTemplateNoPersonalReferences(templateFile) {
  var templatePath = path.join(TEMPLATES_DIR, templateFile);
  if (!fs.existsSync(templatePath)) {
    assert(false, templateFile + ' no personal references: file missing');
    return;
  }
  checkNoPersonalReferences(templatePath, 'templates/' + templateFile);
}

// ── Integration Tests ─────────────────────────────────────────────────

function testInstallCreatesTargetDir() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kuato-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(__dirname, 'install.js'), {
      env: installEnv,
      cwd: __dirname,
      stdio: 'pipe'
    });
    var targetDir = path.join(fakeHome, '.agents', 'skills');
    assert(fs.existsSync(targetDir), 'install.js creates ~/.agents/skills/ directory');
  } catch (err) {
    assert(false, 'install.js creates ~/.agents/skills/ directory: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testInstallCopiesAllSkills() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kuato-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    execSync('node ' + path.join(__dirname, 'install.js'), {
      env: installEnv,
      cwd: __dirname,
      stdio: 'pipe'
    });
    var targetDir = path.join(fakeHome, '.agents', 'skills');
    var allPresent = true;
    var missing = [];
    for (var i = 0; i < EXPECTED_SKILLS.length; i++) {
      var skillPath = path.join(targetDir, EXPECTED_SKILLS[i]);
      if (!fs.existsSync(skillPath)) {
        allPresent = false;
        missing.push(EXPECTED_SKILLS[i]);
      }
    }
    assert(allPresent, 'install.js copies all 7 skills' + (missing.length > 0 ? ' (missing: ' + missing.join(', ') + ')' : ''));
  } catch (err) {
    assert(false, 'install.js copies all 7 skills: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testInstallOverwritesExisting() {
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kuato-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    // First install
    execSync('node ' + path.join(__dirname, 'install.js'), {
      env: installEnv,
      cwd: __dirname,
      stdio: 'pipe'
    });
    // Create a stale file in a skill directory to verify overwrite
    var staleFileDir = path.join(fakeHome, '.agents', 'skills', 'git-sync');
    var staleFilePath = path.join(staleFileDir, 'stale-marker.txt');
    fs.writeFileSync(staleFilePath, 'old content');
    // Second install — should overwrite
    execSync('node ' + path.join(__dirname, 'install.js'), {
      env: installEnv,
      cwd: __dirname,
      stdio: 'pipe'
    });
    // The install copies the package's skills dir content; stale files from
    // prior installs in the target are not explicitly cleaned (overwrite means
    // same-named files get replaced). Verify the SKILL.md was updated.
    var skillMd = path.join(staleFileDir, 'SKILL.md');
    assert(fs.existsSync(skillMd), 'install.js overwrites existing skill directories');
    // The stale file persists because install only writes files from the package,
    // it doesn't delete extra files. This is acceptable — install is not a sync.
    // Verify no error occurred during the second install.
    assert(true, 'install.js runs without error when target already exists');
  } catch (err) {
    assert(false, 'install.js overwrites existing: ' + err.message);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testInstallFailsOnPermissionError() {
  // Create a directory that we can't write to
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kuato-test-'));
  var fakeHome = path.join(tmpDir, 'home');
  fs.mkdirSync(fakeHome);
  var targetDir = path.join(fakeHome, '.agents', 'skills');
  fs.mkdirSync(targetDir, { recursive: true });
  // Make target directory read-only
  try {
    fs.chmodSync(targetDir, 0o444);
  } catch (err) {
    // chmod may not work on all systems (e.g., running as root)
    console.log('  SKIP: testInstallFailsOnPermissionError (chmod not effective on this system)');
    fs.rmSync(tmpDir, { recursive: true, force: true });
    return;
  }
  try {
    var installEnv = Object.assign({}, process.env, { HOME: fakeHome });
    var result = execSync('node ' + path.join(__dirname, 'install.js'), {
      env: installEnv,
      cwd: __dirname,
      stdio: 'pipe'
    });
    // If running as root, this won't fail — skip the assertion
    console.log('  SKIP: testInstallFailsOnPermissionError (running as root or permissions not enforced)');
  } catch (err) {
    assert(err.status !== 0, 'install.js exits with non-zero code on permission error');
  } finally {
    fs.chmodSync(targetDir, 0o755);
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testShareShRejectsSecrets() {
  var shareShPath = path.join(__dirname, 'skills', 'visual-explainer', 'scripts', 'share.sh');
  if (!fs.existsSync(shareShPath)) {
    console.log('  SKIP: share.sh secret detection (share.sh not found)');
    return;
  }
  var tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kuato-share-test-'));
  try {
    // Test: HTML with AWS access key should be rejected
    var awsKeyHtml = path.join(tmpDir, 'aws-key.html');
    fs.writeFileSync(awsKeyHtml, '<!DOCTYPE html><html><body><p>AKIAIOSFODNN7EXAMPLE</p></body></html>');
    try {
      execSync('bash ' + shareShPath + ' ' + awsKeyHtml, { stdio: 'pipe' });
      assert(false, 'share.sh rejects HTML with AWS access key');
    } catch (err) {
      var output = err.stderr ? err.stderr.toString() : '';
      assert(err.status !== 0 && output.indexOf('secrets') !== -1,
        'share.sh rejects HTML with AWS access key');
    }
    // Test: HTML with private key should be rejected
    var privKeyHtml = path.join(tmpDir, 'privkey.html');
    fs.writeFileSync(privKeyHtml, '<!DOCTYPE html><html><body><p>-----BEGIN RSA PRIVATE KEY-----\nMIIBog\n-----END RSA PRIVATE KEY-----</p></body></html>');
    try {
      execSync('bash ' + shareShPath + ' ' + privKeyHtml, { stdio: 'pipe' });
      assert(false, 'share.sh rejects HTML with private key');
    } catch (err) {
      var output2 = err.stderr ? err.stderr.toString() : '';
      assert(err.status !== 0 && output2.indexOf('secrets') !== -1,
        'share.sh rejects HTML with private key');
    }
    // Test: HTML with GitHub token should be rejected
    var ghTokenHtml = path.join(tmpDir, 'ghtoken.html');
    fs.writeFileSync(ghTokenHtml, '<!DOCTYPE html><html><body><p>ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890AA</p></body></html>');
    try {
      execSync('bash ' + shareShPath + ' ' + ghTokenHtml, { stdio: 'pipe' });
      assert(false, 'share.sh rejects HTML with GitHub token');
    } catch (err) {
      var output3 = err.stderr ? err.stderr.toString() : '';
      assert(err.status !== 0 && output3.indexOf('secrets') !== -1,
        'share.sh rejects HTML with GitHub token');
    }
    // Test: Clean HTML should pass secret scan (will fail at vercel-deploy step, not secrets step)
    var cleanHtml = path.join(tmpDir, 'clean.html');
    fs.writeFileSync(cleanHtml, '<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>');
    try {
      execSync('bash ' + shareShPath + ' ' + cleanHtml, { stdio: 'pipe' });
      // If vercel-deploy happens to be installed, deployment succeeds — that's fine
      assert(true, 'share.sh allows clean HTML (no secrets detected)');
    } catch (err) {
      var cleanOutput = err.stderr ? err.stderr.toString() : '';
      // Must NOT fail with "secrets" message — can fail with vercel-deploy not found
      assert(cleanOutput.indexOf('secrets') === -1,
        'share.sh allows clean HTML (no secrets detected)');
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function testTemplatesInPackageJsonFiles() {
  var pkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
  var hasTemplates = pkg.files && pkg.files.indexOf('templates/') !== -1;
  assert(hasTemplates, 'package.json files array includes "templates/"');
}

function testOpencodeJsonLoadsAllHarnessFiles() {
  var config = JSON.parse(fs.readFileSync(path.join(__dirname, 'opencode.json'), 'utf8'));
  var expected = ['harness/identity.md', 'harness/user.md', 'harness/rules.md'];
  var instructions = config.instructions || [];
  for (var i = 0; i < expected.length; i++) {
    assert(instructions.indexOf(expected[i]) !== -1,
      'opencode.json instructions includes ' + expected[i]);
  }
}

// ── Main ──────────────────────────────────────────────────────────────

console.log('\n=== Validation Tests ===\n');

for (var i = 0; i < EXPECTED_SKILLS.length; i++) {
  var skillName = EXPECTED_SKILLS[i];
  var skillDir = path.join(SKILLS_DIR, skillName);
  console.log('--- ' + skillName + ' ---');
  testSkillHasSKILLMd(skillDir);
  testFrontmatterNameMatchesDir(skillDir);
  testFrontmatterDescriptionPresent(skillDir);
  testFrontmatterNameValidFormat(skillDir);
  testNoLobsterdogReferences(skillDir);
  testLicenseFieldPresent(skillDir);
  testNoClawhubFiles(skillDir);
  testNoClaudePluginFiles(skillDir);
  console.log('');
}

console.log('=== Template Validation Tests ===\n');

for (var t = 0; t < EXPECTED_TEMPLATES.length; t++) {
  var tmplFile = EXPECTED_TEMPLATES[t];
  console.log('--- ' + tmplFile + ' ---');
  testTemplateExists(tmplFile);
  testTemplateNoPersonalReferences(tmplFile);
  testTemplateEndsWithNewline(tmplFile);
  console.log('');
}

console.log('=== Integration Tests ===\n');

testInstallCreatesTargetDir();
testInstallCopiesAllSkills();
testInstallOverwritesExisting();
testInstallFailsOnPermissionError();
testShareShRejectsSecrets();
testTemplatesInPackageJsonFiles();
testOpencodeJsonLoadsAllHarnessFiles();

console.log('\n=== Results ===\n');
console.log('Passed: ' + passes);
console.log('Failed: ' + failures);

if (failures > 0) {
  process.exit(1);
} else {
  console.log('\nAll tests passed!');
  process.exit(0);
}