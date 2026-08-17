/**
 * Keep the canonical Pages URL current.
 * GitHub's CDN caches HTML ~10 minutes. Navigations are fetched with a
 * one-time cache-busting query that never appears in the address bar.
 */
self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const isDocument =
    req.mode === 'navigate' ||
    req.destination === 'document' ||
    (req.headers.get('accept') || '').includes('text/html');

  if (!isDocument) return;

  event.respondWith(fresh(req));
});

async function fresh(req) {
  const u = new URL(req.url);
  u.searchParams.set('_sw', String(Date.now()));
  try {
    return await fetch(u.href, { cache: 'no-store' });
  } catch {
    return fetch(req);
  }
}
