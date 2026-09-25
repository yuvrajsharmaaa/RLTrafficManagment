const fs = require('fs');

console.log('--- Testing index.html ---');
const html = fs.readFileSync('index.html', 'utf8');

const requiredStrings = [
  'Delhi Ambulance Dispatch',
  'Adaptive Emergency Routing',
  'ETA to Hospital',
  'Traffic Unpredictability (V)',
  'Search Breadth (β)',
  'Compare Adaptive vs Baseline',
  'Exit Comparison',
  'Start Route Simulation',
  'Zero traffic disruptions detected'
];

let allPassed = true;
for (const s of requiredStrings) {
  if (html.includes(s)) {
    console.log(`[PASS] Found required string: "${s}"`);
  } else {
    console.error(`[FAIL] Missing required string: "${s}"`);
    allPassed = false;
  }
}

const forbiddenStrings = [
  'A reroute has been',
  'has been triggered',
  'was dispatched',
  'Mission Control',
  'quantum swarm',
  'world-class',
  'delivery-stop',
  'Welcome to'
];

for (const f of forbiddenStrings) {
  if (html.toLowerCase().includes(f.toLowerCase())) {
    console.error(`[FAIL] Found forbidden string: "${f}"`);
    allPassed = false;
  } else {
    console.log(`[PASS] Verified absence of: "${f}"`);
  }
}

console.log('\n--- Testing demo.py ---');
const demoPy = fs.readFileSync('demo.py', 'utf8');
const demoRequired = [
  'Ambulance Emergency Route Dispatch',
  'Start Live Simulation',
  'Start Replay',
  'Active Corridor Stops',
  'Decision Event Feed',
  'Live simulation complete',
  'Replay complete.'
];

for (const s of demoRequired) {
  if (demoPy.includes(s)) {
    console.log(`[PASS] Found in demo.py: "${s}"`);
  } else {
    console.error(`[FAIL] Missing in demo.py: "${s}"`);
    allPassed = false;
  }
}

console.log('\n--- Testing export_for_frontend.py ---');
const exportPy = fs.readFileSync('export_for_frontend.py', 'utf8');
if (exportPy.includes('Emergency corridor dispatched. Prioritized fastest path to patient pickup and hospital.')) {
  console.log('[PASS] export_for_frontend.py active voice event text verified.');
} else {
  console.error('[FAIL] export_for_frontend.py missing expected active voice replan text.');
  allPassed = false;
}

console.log('\n--- Testing frontend_data hero JSON files ---');
const heroFiles = [
  'hero_medium_va_qpso.json',
  'hero_high_va_qpso.json',
  'hero_low_va_qpso.json',
  'hero_medium_fixed_beta_qpso.json',
  'hero_high_fixed_beta_qpso.json',
  'hero_low_fixed_beta_qpso.json'
];

for (const file of heroFiles) {
  const content = fs.readFileSync(`frontend_data/${file}`, 'utf8');
  const json = JSON.parse(content);
  if (json.events && json.events.length > 0) {
    for (const ev of json.events) {
      if (ev.detail.toLowerCase().includes('quantum') || ev.detail.toLowerCase().includes('green-wave') || ev.detail.toLowerCase().includes('has been')) {
        console.error(`[FAIL] ${file} event has jargon/filler: "${ev.detail}"`);
        allPassed = false;
      }
    }
    console.log(`[PASS] ${file} events verified clean.`);
  }
}

console.log('\n--- Testing manifest.json ---');
const manifest = JSON.parse(fs.readFileSync('manifest.json', 'utf8'));
if (manifest.description.includes('Quantum')) {
  console.error('[FAIL] manifest.json contains Quantum');
  allPassed = false;
} else {
  console.log(`[PASS] manifest.json verified: "${manifest.name}" - "${manifest.description}"`);
}

if (!allPassed) {
  console.error('\nAudit checks failed.');
  process.exit(1);
} else {
  console.log('\n>>> ALL UI TEXT AUDIT CHECKS PASSED SUCCESSFULLY! <<<');
}
