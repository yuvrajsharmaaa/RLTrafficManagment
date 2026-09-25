const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 8899;
const server = http.createServer((req, res) => {
  let reqPath = req.url.split('?')[0];
  if (reqPath === '/' || reqPath === '/index.html') reqPath = '/index.html';
  const filePath = path.join(__dirname, '..', reqPath.replace(/^\//, ''));
  
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath);
    const mime = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.json': 'application/json',
      '.css': 'text/css',
      '.png': 'image/png',
      '.svg': 'image/svg+xml'
    }[ext] || 'text/plain';
    res.writeHead(200, { 'Content-Type': mime, 'Access-Control-Allow-Origin': '*' });
    res.end(fs.readFileSync(filePath));
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, async () => {
  console.log(`Test server running on port ${PORT}`);
  
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const debugPort = 9227;
  const chrome = spawn(chromePath, [
    '--headless=new',
    `--remote-debugging-port=${debugPort}`,
    '--disable-gpu',
    '--no-sandbox',
    '--disable-extensions',
    `http://localhost:${PORT}/index.html`
  ]);

  await new Promise(r => setTimeout(r, 2500));

  try {
    const listRes = await fetch(`http://127.0.0.1:${debugPort}/json`);
    const list = await listRes.json();
    const target = list.find(t => t.type === 'page');
    if (!target) throw new Error('No page target found');

    const ws = new WebSocket(target.webSocketDebuggerUrl);

    let msgId = 1;
    function send(method, params = {}) {
      return new Promise((resolve, reject) => {
        const id = msgId++;
        const handler = (evt) => {
          const res = JSON.parse(evt.data);
          if (res.id === id) {
            ws.removeEventListener('message', handler);
            if (res.error) reject(res.error);
            else resolve(res.result);
          }
        };
        ws.addEventListener('message', handler);
        ws.send(JSON.stringify({ id, method, params }));
      });
    }

    ws.addEventListener('open', async () => {
      await send('Runtime.enable');
      await send('Console.enable');

      console.log('[CDP] Connected to page, evaluating motion orchestration...');

      // 1. Verify pickup-beacon is removed
      const beaconRes = await send('Runtime.evaluate', {
        expression: `
          (() => {
            let hasBeacon = false;
            for (let s of document.styleSheets) {
              try {
                for (let r of s.cssRules) {
                  if (r.name === 'pickup-beacon') hasBeacon = true;
                }
              } catch(e) {}
            }
            return { hasBeacon };
          })()
        `,
        returnByValue: true
      });
      const beaconCheck = beaconRes.result ? beaconRes.result.value : beaconRes;
      console.log('1. Pickup beacon animation removed:', !beaconCheck.hasBeacon);

      // 2. Verify functions exist
      const funcRes = await send('Runtime.evaluate', {
        expression: `
          ({
            hasRedrawFunc: typeof triggerRouteRedrawAnimation === 'function',
            hasTickFunc: typeof tickETANumber === 'function',
            hasMotionTrigger: typeof triggerOrchestratedMotionMoment === 'function'
          })
        `,
        returnByValue: true
      });
      const funcCheck = funcRes.result ? funcRes.result.value : funcRes;
      console.log('2. Motion functions present:', funcCheck);

      // 3. Test ETA ticking & route line redraw on replan event
      const triggerRes = await send('Runtime.evaluate', {
        expression: `
          new Promise(resolve => {
            const ev = { t: 55, type: 'replan', eta_before: 75, eta_after: 40 };
            triggerOrchestratedMotionMoment(ev, 55);
            
            setTimeout(() => {
              const poly = routePolyline;
              const pathEl = poly ? (poly.getElement ? poly.getElement() : poly._path) : null;
              const transitionApplied = pathEl ? pathEl.style.transition.includes('stroke-dashoffset') : false;
              const strokeDashApplied = pathEl ? pathEl.style.strokeDasharray.length > 0 : false;
              const tickingStarted = isETATicking;
              const currentETA = document.getElementById('metric-eta-large').innerText;

              resolve({
                transitionApplied,
                strokeDashApplied,
                tickingStarted,
                currentETA
              });
            }, 60);
          })
        `,
        awaitPromise: true,
        returnByValue: true
      });
      const triggerTest = triggerRes.result ? triggerRes.result.value : triggerRes;
      console.log('3. Replan motion in flight (ticking & drawing):', triggerTest);

      // 4. Wait 600ms for transition and ticking to complete
      await new Promise(r => setTimeout(r, 650));

      const settleRes = await send('Runtime.evaluate', {
        expression: `
          (() => {
            const finalETA = document.getElementById('metric-eta-large').innerText;
            const poly = routePolyline;
            const pathEl = poly ? (poly.getElement ? poly.getElement() : poly._path) : null;
            const stylesCleared = pathEl ? pathEl.style.strokeDasharray === '' : true;

            return {
              finalETA,
              isETATicking,
              stylesCleared
            };
          })()
        `,
        returnByValue: true
      });
      const settleTest = settleRes.result ? settleRes.result.value : settleRes;
      console.log('4. Replan motion settled after 400ms:', settleTest);

      // 5. Test natural playback loop crossing replan event at t = 55.0
      const playbackTestRes = await send('Runtime.evaluate', {
        expression: `
          new Promise(resolve => {
            // Jump to t = 54.0s right before the t = 55.0s replan event
            playbackTime = 54.0;
            updatePlayback(54.0);

            // Advance to t = 55.2s (triggering checkAndTriggerReplanMotion)
            updatePlayback(55.2);

            setTimeout(() => {
              const ticking = isETATicking;
              const eta = document.getElementById('metric-eta-large').innerText;
              const poly = routePolyline;
              const pathEl = poly ? (poly.getElement ? poly.getElement() : poly._path) : null;
              const drawing = pathEl ? pathEl.style.transition.includes('stroke-dashoffset') : false;

              resolve({
                playbackTriggeredTicking: ticking,
                playbackTriggeredDrawing: drawing,
                midTickETA: eta
              });
            }, 60);
          })
        `,
        awaitPromise: true,
        returnByValue: true
      });
      const playbackTest = playbackTestRes.result ? playbackTestRes.result.value : playbackTestRes;
      console.log('5. Natural playback crossing t=55.0s triggered replan moment:', playbackTest);

      ws.close();
      chrome.kill();
      server.close();
      process.exit(0);
    });
  } catch (e) {
    console.error('CDP test error:', e);
    chrome.kill();
    server.close();
    process.exit(1);
  }
});
