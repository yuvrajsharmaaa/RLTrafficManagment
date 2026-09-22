/**
 * QPSO Emergency Route Optimizer — Service Worker
 * Enables full offline functionality, fast cache-first loading, and standalone PWA experience.
 */

const CACHE_NAME = 'qpso-ems-v1';

const PRECACHE_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  './icon.svg',
  './hospitals.json',
  './frontend_data/index.json',
  './frontend_data/hospitals.json',
  './frontend_data/hero_medium_va_qpso.json',
  './frontend_data/hero_medium_fixed_beta_qpso.json',
  './frontend_data/hero_high_va_qpso.json',
  './frontend_data/hero_high_fixed_beta_qpso.json',
  './frontend_data/hero_low_va_qpso.json',
  './frontend_data/hero_low_fixed_beta_qpso.json',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css'
];

// Install: Cache critical assets and activate immediately
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Pre-caching offline assets');
      return cache.addAll(PRECACHE_ASSETS).catch((err) => {
        console.warn('[ServiceWorker] Some assets failed to precache:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

// Activate: Clean up old cache versions and claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing old cache version:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Cache-first strategy with dynamic caching and offline fallbacks
self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }

      // Fetch from network and dynamically cache valid responses
      return fetch(event.request).then((networkResponse) => {
        // Cache valid HTTP responses and opaque CDN responses (status 0)
        if (networkResponse && (networkResponse.status === 200 || networkResponse.type === 'opaque')) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch((fetchErr) => {
        console.warn('[ServiceWorker] Fetch failed; returning offline fallback:', event.request.url);

        // Offline Fallback for HTML navigations
        if (event.request.mode === 'navigate') {
          return caches.match('./index.html') || caches.match('/index.html');
        }

        // Offline Fallback for JSON scenario requests
        if (url.pathname.endsWith('.json')) {
          const filename = url.pathname.split('/').pop();
          return caches.match(`./frontend_data/${filename}`)
            .then(res => res || caches.match(`./hospitals.json`))
            .then(res => res || caches.match('./frontend_data/hero_medium_va_qpso.json'));
        }

        throw fetchErr;
      });
    })
  );
});
