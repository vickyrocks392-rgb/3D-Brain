/**
 * Validate 3D Observatory interactions and modes via Chrome DevTools Protocol.
 */
const { spawn } = require('child_process');
const http = require('http');

async function run() {
  const chromeProcess = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
    '--headless=new',
    '--remote-debugging-port=9223',
    '--user-data-dir=/tmp/chrome_test_prof_' + Date.now(),
    '--window-size=1440,900',
    'http://localhost:8080/visualization/player.html'
  ]);

  // Wait for Chrome to initialize with retries
  let versionData = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      versionData = await new Promise((resolve, reject) => {
        const req = http.get('http://127.0.0.1:9223/json/version', res => {
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
        http.get('http://127.0.0.1:9223/json/list', res => {
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
  if (!pageTarget) throw new Error('No page target found on port 9223');
  console.log('Connected to target:', pageTarget.url);

  const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

  let msgId = 1;
  const callbacks = new Map();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && callbacks.has(msg.id)) {
      const cb = callbacks.get(msg.id);
      callbacks.delete(msg.id);
      cb(msg);
    }
    if (msg.method === 'Runtime.consoleAPICalled') {
      console.log('BROWSER CONSOLE:', msg.params.type, msg.params.args.map(a => a.value || a.description).join(' '));
    }
    if (msg.method === 'Runtime.exceptionThrown') {
      console.error('BROWSER EXCEPTION:', JSON.stringify(msg.params.exceptionDetails));
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
  await new Promise(r => setTimeout(r, 2500));

  // Evaluate state on startup
  const startupState = await sendCommand('Runtime.evaluate', {
    expression: `({
      appMode: window.appMode,
      totalFrames: window.logData ? window.logData.frames.length : 0,
      currentStep: window.replay ? window.replay.currentStep : -1,
      totalNodes: window.nodeMeshes ? window.nodeMeshes.length : 0,
      totalEdges: window.edgeLines ? window.edgeLines.length : 0
    })`,
    returnByValue: true
  });
  console.log('STARTUP STATE:', JSON.stringify(startupState.result.value));

  // 1. Test clicking a Neuron (Node 4)
  await sendCommand('Runtime.evaluate', {
    expression: `window.selectNeuron(4);`
  });
  await new Promise(r => setTimeout(r, 500));
  const neuronInspText = await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('n-label-title').innerText + ' | ' + document.getElementById('n-type-pill').innerText`,
    returnByValue: true
  });
  console.log('NEURON INSPECTION:', neuronInspText.result.value);

  // 2. Test Toggling Action Interface Mode [I]
  await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('action-interface-btn').click();`
  });
  await new Promise(r => setTimeout(r, 500));
  const actOverlayDisplay = await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('action-interface-overlay').style.display`,
    returnByValue: true
  });
  console.log('ACTION INTERFACE OVERLAY DISPLAY:', actOverlayDisplay.result.value);

  // Capture screenshot with Action Interface Mode active
  const shot1 = await sendCommand('Page.captureScreenshot');
  require('fs').writeFileSync('experiments/results/observatory_action_interface.png', Buffer.from(shot1.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_action_interface.png');

  // 3. Test Toggling Spike Encoding Mode [S]
  await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('action-interface-btn').click(); document.getElementById('spike-encoding-btn').click();`
  });
  await new Promise(r => setTimeout(r, 500));
  const spikeOverlayDisplay = await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('spike-encoding-overlay').style.display`,
    returnByValue: true
  });
  console.log('SPIKE ENCODING OVERLAY DISPLAY:', spikeOverlayDisplay.result.value);

  const shot2 = await sendCommand('Page.captureScreenshot');
  require('fs').writeFileSync('experiments/results/observatory_spike_encoding.png', Buffer.from(shot2.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_spike_encoding.png');

  // 4. Test Switching to REPLAY Mode
  await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('spike-encoding-btn').click(); window.setMode('REPLAY');`
  });
  await new Promise(r => setTimeout(r, 500));
  const replayBarDisplay = await sendCommand('Runtime.evaluate', {
    expression: `document.getElementById('replay-timeline-bar').style.display`,
    returnByValue: true
  });
  console.log('REPLAY TIMELINE BAR DISPLAY:', replayBarDisplay.result.value);

  const shot3 = await sendCommand('Page.captureScreenshot');
  require('fs').writeFileSync('experiments/results/observatory_replay_mode.png', Buffer.from(shot3.result.data, 'base64'));
  console.log('Saved experiments/results/observatory_replay_mode.png');

  // 5. Test Returning to LATEST Mode
  await sendCommand('Runtime.evaluate', {
    expression: `window.setMode('LATEST');`
  });
  await new Promise(r => setTimeout(r, 500));
  const returnMode = await sendCommand('Runtime.evaluate', {
    expression: `window.appMode + ' | step: ' + window.replay.currentStep`,
    returnByValue: true
  });
  console.log('RETURNED TO MODE:', returnMode.result.value);

  ws.close();
  chromeProcess.kill();
  console.log('VALIDATION COMPLETED SUCCESSFULLY!');
}

run().catch(err => {
  console.error('Test error:', err);
  process.exit(1);
});
