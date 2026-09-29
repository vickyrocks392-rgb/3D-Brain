/**
 * Phase 14 3D Neural Observatory Validation via Chrome DevTools Protocol (CDP)
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CDP_PORT = 9333;
const TARGET_URL = 'http://localhost:8080/visualization/player.html';
const ARTIFACTS_DIR = '/Users/kalachanddash/.gemini/antigravity-ide/brain/eb759292-4107-4822-949e-502322b8fd8e';
const VIS_SCREENSHOTS_DIR = path.resolve(__dirname, 'screenshots');

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function httpGetJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.msgId = 1;
    this.callbacks = new Map();
    this.consoleLogs = [];
    this.consoleErrors = [];
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.method === 'Runtime.consoleAPICalled') {
          const type = msg.params.type;
          const text = msg.params.args.map(a => a.value || JSON.stringify(a)).join(' ');
          this.consoleLogs.push({ type, text });
          if (type === 'error') {
            this.consoleErrors.push(text);
          }
        }
        if (msg.method === 'Runtime.exceptionThrown') {
          const desc = msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text;
          this.consoleErrors.push(desc);
        }
        if (msg.id && this.callbacks.has(msg.id)) {
          const { res, rej } = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) rej(new Error(msg.error.message));
          else res(msg.result);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.msgId++;
      this.callbacks.set(id, { res: resolve, rej: reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    if (res.exceptionDetails) {
      throw new Error(res.exceptionDetails.text || 'Eval error');
    }
    return res.result?.value;
  }

  async captureScreenshot(filepath) {
    const res = await this.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false,
      fromSurface: true
    });
    const buffer = Buffer.from(res.data, 'base64');
    fs.writeFileSync(filepath, buffer);
    console.log(`Saved screenshot: ${filepath} (${buffer.length} bytes)`);
  }
}

async function run() {
  console.log('=== Starting Phase 14 3D Observatory Validation ===');
  
  // Launch Chrome Headless
  const chromeProcess = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${CDP_PORT}`,
    '--remote-allow-origins=*',
    '--user-data-dir=/tmp/chrome_cdp_phase14',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--no-sandbox',
    '--window-size=1440,900',
    'about:blank'
  ]);

  let connected = false;
  let targetWs = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    await sleep(500);
    try {
      const targets = await httpGetJson(`http://localhost:${CDP_PORT}/json`);
      if (targets && targets.length > 0) {
        targetWs = targets[0].webSocketDebuggerUrl;
        connected = true;
        break;
      }
    } catch (e) {}
  }

  if (!connected || !targetWs) {
    chromeProcess.kill();
    throw new Error('Failed to connect to Chrome CDP');
  }

  console.log(`Connected to Chrome CDP at ${targetWs}`);
  const client = new CDPClient(targetWs);
  await client.connect();

  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('DOM.enable');
  await client.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false
  });

  console.log(`Navigating to ${TARGET_URL}...`);
  await client.send('Page.navigate', { url: TARGET_URL });

  // Wait for 3D Brain and Replay to initialize
  console.log('Waiting for 3D Brain & BrainLog to initialize...');
  let ready = false;
  for (let i = 0; i < 30; i++) {
    await sleep(400);
    try {
      const isReady = await client.eval(`
        Boolean(window.isReady && window.nodeMeshes && window.nodeMeshes.length > 0)
      `);
      if (isReady) {
        ready = true;
        break;
      }
    } catch (e) {}
  }

  if (!ready) {
    console.error('Observatory did not initialize within timeout.');
    console.error('Console errors:', client.consoleErrors);
    chromeProcess.kill();
    process.exit(1);
  }

  console.log('Observatory initialized successfully!');

  await client.eval(`
    window.dispatchEvent(new Event('resize'));
    if (window.renderFrame && window.replay) {
      window.renderFrame(window.replay.currentStep);
    }
  `);
  await sleep(600);

  // Check state verification
  const auditReport = await client.eval(`
    (() => {
      const r = window.replay;
      const meshCount = window.nodeMeshes.length;
      const step = r.currentStep;
      const totalSteps = r.totalSteps;
      const mode = document.getElementById('mode-latest-pill')?.innerText || '';
      const modelVal = document.getElementById('model-select')?.value || '';
      return {
        meshCount,
        step,
        totalSteps,
        mode,
        modelVal,
        hasScene: Boolean(window.scene),
        hasCamera: Boolean(window.camera)
      };
    })()
  `);

  console.log('Bootstrap State:', auditReport);

  // 1. Verify Default 3D View & Capture Screenshot
  const ss1 = path.join(VIS_SCREENSHOTS_DIR, 'phase14_default_3d_brain.png');
  const ss1_art = path.join(ARTIFACTS_DIR, 'phase14_default_3d_brain.png');
  await client.captureScreenshot(ss1);
  fs.copyFileSync(ss1, ss1_art);

  // 2. Test Camera Orbit, Zoom, Pan
  console.log('Testing Camera Orbit & Pan...');
  await client.eval(`
    (() => {
      window.camTheta += 0.4;
      window.camPhi = Math.PI / 2.5;
      window.camRadius = 35.0;
      window.updateCameraPosition();
    })()
  `);
  await sleep(300);

  // 3. Test Neuron Selection & Biophysical Inspector
  console.log('Testing Neuron Selection & Inspector...');
  await client.eval(`
    (() => {
      window.selectNode(2); // Select Node 2 (Recurrent neuron)
    })()
  `);
  await sleep(300);

  const inspState = await client.eval(`
    (() => {
      const elem = document.getElementById('insp-neuron-state');
      const v_soma = document.getElementById('n-vsoma')?.innerText;
      const v_dend = document.getElementById('n-vdend')?.innerText;
      const g_nmda = document.getElementById('n-gnmda')?.innerText;
      return {
        visible: elem.style.display !== 'none',
        v_soma,
        v_dend,
        g_nmda
      };
    })()
  `);
  console.log('Neuron Inspector State:', inspState);

  // 4. Test Temporal Coding Mode [T]
  console.log('Testing Temporal Coding Mode [T]...');
  await client.eval(`
    (() => {
      document.getElementById('temporal-coding-btn').click();
    })()
  `);
  await sleep(400);

  const tcOverlayState = await client.eval(`
    (() => {
      const o = document.getElementById('temporal-coding-overlay');
      const scheme = document.getElementById('tc-scheme')?.innerText;
      const windowSteps = document.getElementById('tc-window')?.innerText;
      const bursts = document.getElementById('tc-total-bursts')?.innerText;
      return {
        display: o.style.display,
        scheme,
        windowSteps,
        bursts
      };
    })()
  `);
  console.log('Temporal Coding Overlay:', tcOverlayState);

  const ss2 = path.join(VIS_SCREENSHOTS_DIR, 'phase14_temporal_coding_mode.png');
  const ss2_art = path.join(ARTIFACTS_DIR, 'phase14_temporal_coding_mode.png');
  await client.captureScreenshot(ss2);
  fs.copyFileSync(ss2, ss2_art);

  // 5. Test Population Resonance Mode [N]
  console.log('Testing Population Resonance Mode [N]...');
  await client.eval(`
    (() => {
      document.getElementById('resonance-btn').click();
    })()
  `);
  await sleep(400);

  const resOverlayState = await client.eval(`
    (() => {
      const o = document.getElementById('resonance-overlay');
      const rGlobal = document.getElementById('res-r-global')?.innerText;
      const rLocal = document.getElementById('res-r-local')?.innerText;
      const plv = document.getElementById('res-plv')?.innerText;
      const qfunc = document.getElementById('res-qfunc')?.innerText;
      return {
        display: o.style.display,
        rGlobal,
        rLocal,
        plv,
        qfunc
      };
    })()
  `);
  console.log('Resonance Overlay:', resOverlayState);

  const ss3 = path.join(VIS_SCREENSHOTS_DIR, 'phase14_resonance_mode.png');
  const ss3_art = path.join(ARTIFACTS_DIR, 'phase14_resonance_mode.png');
  await client.captureScreenshot(ss3);
  fs.copyFileSync(ss3, ss3_art);

  // 6. Test Information Flow Mode [I]
  console.log('Testing Information Flow Mode [I]...');
  await client.eval(`
    (() => {
      document.getElementById('info-flow-btn').click();
    })()
  `);
  await sleep(400);

  const infoOverlayState = await client.eval(`
    (() => {
      const o = document.getElementById('info-flow-overlay');
      return {
        display: o.style.display
      };
    })()
  `);
  console.log('Info Flow Overlay:', infoOverlayState);

  const ss4 = path.join(VIS_SCREENSHOTS_DIR, 'phase14_information_flow_mode.png');
  const ss4_art = path.join(ARTIFACTS_DIR, 'phase14_information_flow_mode.png');
  await client.captureScreenshot(ss4);
  fs.copyFileSync(ss4, ss4_art);

  // 7. Test Action Interface & Spike Encoding Overlays
  console.log('Testing Action Interface [A] and Spike Encoding [S] toggles...');
  await client.eval(`
    (() => {
      document.getElementById('action-interface-btn').click();
      document.getElementById('spike-encoding-btn').click();
    })()
  `);
  await sleep(300);

  // 8. Test Replay Step & Latest Restoration
  console.log('Testing Replay stepping & Latest-state restoration...');
  await client.eval(`
    (() => {
      window.replay.stepBackward();
      window.renderFrame(window.replay.currentStep);
    })()
  `);
  await sleep(200);

  await client.eval(`
    (() => {
      window.setMode('LATEST');
    })()
  `);
  await sleep(200);

  const finalMode = await client.eval(`window.mode`);
  console.log(`Current visualizer mode: ${finalMode}`);

  console.log('=== Console Logs & Errors Audit ===');
  console.log(`Total console logs: ${client.consoleLogs.length}`);
  console.log(`Critical console errors: ${client.consoleErrors.length}`);
  if (client.consoleErrors.length > 0) {
    console.error('Critical Errors Encountered:', client.consoleErrors);
  }

  // Cleanup
  chromeProcess.kill();
  console.log('Chrome process closed.');

  const passed = (
    auditReport.meshCount > 0 &&
    auditReport.totalSteps > 0 &&
    inspState.visible &&
    tcOverlayState.display === 'flex' &&
    resOverlayState.display === 'flex' &&
    infoOverlayState.display === 'flex' &&
    client.consoleErrors.length === 0
  );

  console.log(`\nObservatory Validation: ${passed ? 'PASSED 100%' : 'FAILED'}`);
  process.exit(passed ? 0 : 1);
}

run().catch(err => {
  console.error('Validation script failed:', err);
  process.exit(1);
});
