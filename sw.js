/**
 * Retiring service worker.
 *
 * The previous single-page app registered a cache-first worker that stored
 * index.html and the old hero-run files. The current app (web/, served at /)
 * does not use a service worker. Browsers re-check this file on every visit,
 * so this version deletes every cache the old worker created, unregisters
 * itself and reloads open tabs, which then load the current app and data.
 */

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
      await self.registration.unregister();
      const clients = await self.clients.matchAll({ type: 'window' });
      clients.forEach((client) => client.navigate(client.url));
    })(),
  );
});

// No fetch handler: every request goes to the network.
