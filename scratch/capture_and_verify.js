const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 8898;
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
  const debugPort = 9226;
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
      await send('Page.enable');
      await send('DOM.enable');

      console.log('[CDP] Connected. Dismissing intro guide to show main view...');
      await send('Runtime.evaluate', {
        expression: `
          const modal = document.getElementById('intro-modal');
          if (modal) modal.hidden = true;
          if (typeof pause === 'function') pause();
        `
      });

      await new Promise(r => setTimeout(r, 1000));

      const artifactDir = 'C:\\Users\\Asus\\.gemini\\antigravity-ide\\brain\\afe228c8-bd1b-4904-950f-0e54721053dd';

      // 1. Desktop Viewport Screenshot (1280 x 800)
      console.log('Capturing Desktop Screenshot (1280x800)...');
      await send('Emulation.setDeviceMetricsOverride', {
        width: 1280,
        height: 800,
        deviceScaleFactor: 1,
        mobile: false
      });
      await send('Runtime.evaluate', { expression: 'if (typeof map !== "undefined" && map) map.invalidateSize();' });
      await new Promise(r => setTimeout(r, 1200));

      const desktopShot = await send('Page.captureScreenshot', { format: 'png' });
      const desktopPath = path.join(artifactDir, 'desktop_view.png');
      fs.writeFileSync(desktopPath, Buffer.from(desktopShot.data, 'base64'));
      console.log(`Saved desktop screenshot to ${desktopPath}`);

      // 2. Mobile Viewport Screenshot (390 x 844)
      console.log('Capturing Mobile Screenshot (390x844)...');
      await send('Emulation.setDeviceMetricsOverride', {
        width: 390,
        height: 844,
        deviceScaleFactor: 2,
        mobile: true
      });
      await send('Runtime.evaluate', { expression: 'if (typeof map !== "undefined" && map) map.invalidateSize();' });
      await new Promise(r => setTimeout(r, 1200));

      const mobileShot = await send('Page.captureScreenshot', { format: 'png' });
      const mobilePath = path.join(artifactDir, 'mobile_390_view.png');
      fs.writeFileSync(mobilePath, Buffer.from(mobileShot.data, 'base64'));
      console.log(`Saved mobile screenshot to ${mobilePath}`);

      // 3. Confirm Visible Focus States on Interactive Elements
      console.log('\n--- Verifying Keyboard Focus States ---');
      const focusCheck = await send('Runtime.evaluate', {
        expression: `
          (() => {
            const elements = [
              document.getElementById('scenario-selector'),
              document.getElementById('btn-toggle-compare'),
              document.getElementById('timeline'),
              document.getElementById('btn-play-pause'),
              document.getElementById('btn-restart'),
              document.querySelector('.info-btn'),
              document.getElementById('event-drawer-handle')
            ];
            const results = [];
            for (const el of elements) {
              if (!el) continue;
              el.focus();
              const style = window.getComputedStyle(el);
              results.push({
                tag: el.tagName,
                id: el.id || el.className,
                outlineStyle: style.outlineStyle,
                outlineColor: style.outlineColor,
                outlineWidth: style.outlineWidth,
                boxShadow: style.boxShadow,
                hasFocusVisible: el.matches(':focus-visible') || el.matches(':focus')
              });
            }
            return results;
          })()
        `,
        returnByValue: true
      });
      console.log('Focus verification results:', focusCheck.result.value);

      // 4. Confirm Prefers-Reduced-Motion
      console.log('\n--- Verifying prefers-reduced-motion support ---');
      // Emulate reduced motion
      await send('Emulation.setEmulatedMedia', {
        media: 'screen',
        features: [{ name: 'prefers-reduced-motion', value: 'reduce' }]
      });

      const reducedMotionCheck = await send('Runtime.evaluate', {
        expression: `
          (() => {
            const prefers = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            
            // Test tickETANumber with reduced motion
            const etaDisplay = document.getElementById('metric-eta-large');
            tickETANumber(100, 45, 400);
            const snappedImmediate = (etaDisplay.innerText === '00:45');
            const isTickingImmediate = isETATicking;

            return {
              prefersReducedMotionMatches: prefers,
              snappedImmediate,
              isTickingImmediate
            };
          })()
        `,
        returnByValue: true
      });
      console.log('Reduced motion check results:', reducedMotionCheck.result.value);

      // 5. Confirm Color Contrast
      console.log('\n--- Verifying Color Contrast ---');
      const contrastCheck = await send('Runtime.evaluate', {
        expression: `
          (() => {
            const rootStyle = getComputedStyle(document.documentElement);
            const bg = rootStyle.getPropertyValue('--bg').trim();
            const surface = rootStyle.getPropertyValue('--surface').trim();
            const textPri = rootStyle.getPropertyValue('--text-primary').trim();
            const textSec = rootStyle.getPropertyValue('--text-secondary').trim();

            function hexToRgb(hex) {
              hex = hex.replace('#', '');
              if (hex.length === 3) hex = hex.split('').map(c => c+c).join('');
              const num = parseInt(hex, 16);
              return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
            }

            function getLuminance(rgb) {
              const a = rgb.map(v => {
                v /= 255;
                return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
              });
              return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
            }

            function contrast(c1, c2) {
              const l1 = getLuminance(hexToRgb(c1));
              const l2 = getLuminance(hexToRgb(c2));
              const lighter = Math.max(l1, l2);
              const darker = Math.min(l1, l2);
              return (lighter + 0.05) / (darker + 0.05);
            }

            return {
              bg,
              surface,
              textPrimary: textPri,
              textSecondary: textSec,
              contrastPrimaryVsBg: contrast(textPri, bg).toFixed(2),
              contrastSecondaryVsBg: contrast(textSec, bg).toFixed(2),
              contrastSecondaryVsSurface: contrast(textSec, surface).toFixed(2)
            };
          })()
        `,
        returnByValue: true
      });
      console.log('Contrast calculation results:', contrastCheck.result.value);

      ws.close();
      chrome.kill();
      server.close();
      process.exit(0);
    });
  } catch (err) {
    console.error('Test execution error:', err);
    chrome.kill();
    server.close();
    process.exit(1);
  }
});
