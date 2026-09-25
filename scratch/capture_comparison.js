const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 8897;
const server = http.createServer((req, res) => {
  let reqPath = req.url.split('?')[0];
  if (reqPath === '/' || reqPath === '/index.html') reqPath = '/index.html';
  const filePath = path.join(__dirname, '..', reqPath.replace(/^\//, ''));
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath);
    const mime = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml' }[ext] || 'text/plain';
    res.writeHead(200, { 'Content-Type': mime, 'Access-Control-Allow-Origin': '*' });
    res.end(fs.readFileSync(filePath));
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, async () => {
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const debugPort = 9225;
  const chrome = spawn(chromePath, [
    '--headless=new',
    `--remote-debugging-port=${debugPort}`,
    '--disable-gpu',
    '--no-sandbox',
    '--disable-extensions',
    `http://localhost:${PORT}/index.html?compare=true`
  ]);

  await new Promise(r => setTimeout(r, 2500));

  try {
    const listRes = await fetch(`http://127.0.0.1:${debugPort}/json`);
    const list = await listRes.json();
    const target = list.find(t => t.type === 'page');
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

      await send('Runtime.evaluate', {
        expression: `
          const modal = document.getElementById('intro-modal');
          if (modal) modal.hidden = true;
          if (typeof pause === 'function') pause();
        `
      });

      await new Promise(r => setTimeout(r, 1500));

      const artifactDir = 'C:\\Users\\Asus\\.gemini\\antigravity-ide\\brain\\afe228c8-bd1b-4904-950f-0e54721053dd';

      // Desktop comparison
      await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false });
      await send('Runtime.evaluate', { expression: 'if (mapAdaptive) mapAdaptive.invalidateSize(); if (mapFixed) mapFixed.invalidateSize();' });
      await new Promise(r => setTimeout(r, 1000));
      const shotDesktop = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(path.join(artifactDir, 'desktop_comparison.png'), Buffer.from(shotDesktop.data, 'base64'));

      // Mobile comparison (390px)
      await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
      await send('Runtime.evaluate', { expression: 'if (mapAdaptive) mapAdaptive.invalidateSize(); if (mapFixed) mapFixed.invalidateSize();' });
      await new Promise(r => setTimeout(r, 1000));
      const shotMobile = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(path.join(artifactDir, 'mobile_390_comparison.png'), Buffer.from(shotMobile.data, 'base64'));

      console.log('Comparison screenshots saved!');
      ws.close();
      chrome.kill();
      server.close();
      process.exit(0);
    });
  } catch (e) {
    console.error(e);
    chrome.kill();
    server.close();
    process.exit(1);
  }
});
