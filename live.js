(function () {
  if (!('serviceWorker' in navigator)) return;

  var root = new URL('./', document.currentScript.src);

  navigator.serviceWorker.register(new URL('sw.js', root), { scope: root.pathname }).then(function () {
    if (navigator.serviceWorker.controller) return;
    navigator.serviceWorker.addEventListener('controllerchange', function () {
      if (sessionStorage.getItem('sw-claimed')) return;
      sessionStorage.setItem('sw-claimed', '1');
      location.reload();
    });
  });
})();
