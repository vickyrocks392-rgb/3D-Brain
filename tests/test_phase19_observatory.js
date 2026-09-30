/**
 * Validate Phase 19 3D Observatory interactions and telemetry overlays via Chrome DevTools Protocol.
 */
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

async function run() {
  const chromeProcess = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
    '--headless=new',
    '--remote-debugging-port=9224',
    '--user-data-dir=/tmp/chrome_p19_prof_' + Date.now(),
    '--window-size=1440,900',
    'http://localhost:8080/visualization/player.html'
  ]);

  let versionData = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      versionData = await new Promise((resolve, reject) => {
        const req = http.get('http://127.0.0.1:9224/json/version', res => {
          let data = '';
          res.on('data', chunk => data += chunk);
          res.on('end', () => resolve(JSON.parse(data)));
        });
        req.on('error', reject);
        req.setTimeout(1000, () => req.destroy());
      });
      if (versionData) break;
    } catch (e) {
      await new Promise(r => setTimeout(r, 500));
    }
  }
  console.log('Chrome Browser Version:', versionData ? versionData.Browser : 'Unknown');

  // Poll for page target
  let pageTarget = null;
  for (let i = 0; i < 15; i++) {
    try {
      const targetsData = await new Promise((resolve, reject) => {
        http.get('http://127.0.0.1:9224/json/list', res => {
          let data = '';
          res.on('data', chunk => data += chunk);
          res.on('end', () => resolve(JSON.parse(data)));
        }).on('error', reject);
      });
      pageTarget = targetsData.find(t => t.type === 'page');
      if (pageTarget) break;
    } catch (e) {}
    await new Promise(r => setTimeout(r, 500));
  }
  if (!pageTarget) throw new Error('No page target found on port 9224');
  console.log('Connected to target:', pageTarget.url);

  const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

  let msgId = 1;
  const callbacks = new Map();
  const consoleErrors = [];
  const exceptions = [];

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && callbacks.has(msg.id)) {
      const cb = callbacks.get(msg.id);
      callbacks.delete(msg.id);
      cb(msg);
    }
    if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map(a => a.value || a.description).join(' ');
      if (msg.params.type === 'error') {
        console.error('BROWSER CONSOLE ERROR:', text);
        consoleErrors.push(text);
      } else {
        console.log('BROWSER CONSOLE:', msg.params.type, text);
      }
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      console.error('BROWSER EXCEPTION:', JSON.stringify(msg.params.exceptionDetails));
      exceptions.push(msg.params.exceptionDetails);
    }
  };

  await new Promise((resolve) => ws.onopen = resolve);

  function sendCommand(method, params = {}) {
    return new Promise((resolve) => {
      const id = msgId++;
      callbacks.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params }));
    });
  }

  await sendCommand('Runtime.enable');
  await sendCommand('Page.enable');

  // Wait for trace to load
  await new Promise(r => setTimeout(r, 3000));

  function getVal(res) {
    if (res && res.result && res.result.result) {
      return res.result.result.value;
    }
    if (res && res.result) {
      return res.result.value;
    }
    return undefined;
  }

  // 1. Startup State
  const startupRes = await sendCommand('Runtime.evaluate', {
    expression: `({
      appMode: window.appMode,
      totalFrames: window.logData ? window.logData.frames.length : 0,
      totalNodes: window.nodeMeshes ? window.nodeMeshes.length : 0,
      totalEdges: window.edgeLines ? window.edgeLines.length : 0
    })`,
    returnByValue: true
  });
  console.log('STARTUP STATE:', JSON.stringify(getVal(startupRes)));

  // 2. Test Credit Horizon Overlay [T]
  console.log('\nTesting [T] Credit Horizon Overlay...');
  const clickRes = await sendCommand('Runtime.evaluate', {
    expression: `(() => {
      const btn = document.getElementById('credit-horizon-btn');
      const overlay = document.getElementById('credit-horizon-overlay');
      btn.click();
      return {
        btnText: btn ? btn.innerText : 'btn not found',
        overlayDisplay: overlay ? overlay.style.display : 'overlay not found',
        creditHorizonActive: typeof creditHorizonActive !== 'undefined' ? creditHorizonActive : 'undef'
      };
    })()`,
    returnByValue: true
  });
  console.log('Credit Horizon Click Result:', JSON.stringify(getVal(clickRes)));
  const creditDisplay = getVal(clickRes).overlayDisplay;
  if (!['flex', 'block'].includes(creditDisplay)) {
    throw new Error(`Credit horizon overlay failed to display! Got: ${creditDisplay}`);
  }

  // Screenshot Credit Horizon
  const shotT = await sendCommand('Page.captureScreenshot');
  fs.writeFileSync('experiments/results/observatory_phase19_credit_horizon.png', Buffer.from(shotT.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_phase19_credit_horizon.png');

  // 3. Test Consolidation Overlay [C]
  console.log('\nTesting [C] Synaptic Consolidation Overlay...');
  const consolClickRes = await sendCommand('Runtime.evaluate', {
    expression: `(() => {
      document.getElementById('close-credit-horizon-btn').click();
      const btn = document.getElementById('consolidation-btn');
      const overlay = document.getElementById('consolidation-overlay');
      btn.click();
      return {
        btnText: btn ? btn.innerText : 'btn not found',
        overlayDisplay: overlay ? overlay.style.display : 'overlay not found',
      };
    })()`,
    returnByValue: true
  });
  const consolDisplay = getVal(consolClickRes).overlayDisplay;
  console.log('Consolidation Overlay Display:', consolDisplay);
  if (!['flex', 'block'].includes(consolDisplay)) {
    throw new Error(`Consolidation overlay failed to display! Got: ${consolDisplay}`);
  }

  const shotC = await sendCommand('Page.captureScreenshot');
  fs.writeFileSync('experiments/results/observatory_phase19_consolidation.png', Buffer.from(shotC.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_phase19_consolidation.png');

  // 4. Test Baseline Comparison Overlay [B]
  console.log('\nTesting [B] Baseline Comparison Overlay...');
  const baselineClickRes = await sendCommand('Runtime.evaluate', {
    expression: `(() => {
      document.getElementById('close-consolidation-btn').click();
      const btn = document.getElementById('baseline-comp-btn');
      const overlay = document.getElementById('baseline-comp-overlay');
      btn.click();
      return {
        btnText: btn ? btn.innerText : 'btn not found',
        overlayDisplay: overlay ? overlay.style.display : 'overlay not found',
      };
    })()`,
    returnByValue: true
  });
  const baselineDisplay = getVal(baselineClickRes).overlayDisplay;
  console.log('Baseline Comparison Overlay Display:', baselineDisplay);
  if (!['flex', 'block'].includes(baselineDisplay)) {
    throw new Error(`Baseline comparison overlay failed to display! Got: ${baselineDisplay}`);
  }

  const shotB = await sendCommand('Page.captureScreenshot');
  fs.writeFileSync('experiments/results/observatory_phase19_baseline_comp.png', Buffer.from(shotB.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_phase19_baseline_comp.png');

  // 5. Test Keyboard Shortcuts 'T', 'C', 'B'
  console.log('\nTesting Keyboard Shortcuts...');
  await sendCommand('Runtime.evaluate', {
    expression: `
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b' }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 't' }));
    `
  });
  await new Promise(r => setTimeout(r, 500));
  const keyStateRes = await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('credit-horizon-overlay').style.display`,
    returnByValue: true
  });
  console.log('Overlay state after "t" keydown:', getVal(keyStateRes));

  // Close overlays
  await sendCommand('Runtime.evaluate', {
    expression: `window.dispatchEvent(new KeyboardEvent('keydown', { key: 't' }));`
  });

  // Verify Zero Console Errors
  console.log('\nVerifying Zero Console Errors & Exceptions...');
  console.log('Console Errors count:', consoleErrors.length);
  console.log('Exceptions count:', exceptions.length);

  ws.close();
  chromeProcess.kill();

  if (consoleErrors.length > 0 || exceptions.length > 0) {
    throw new Error(`Observatory validation failed: ${consoleErrors.length} errors, ${exceptions.length} exceptions`);
  }

  console.log('\n>>> PHASE 19 OBSERVATORY VERIFICATION PASSED WITH ZERO CONSOLE ERRORS! <<<');
}

run().catch(err => {
  console.error('Phase 19 Observatory Test error:', err);
  process.exit(1);
});
