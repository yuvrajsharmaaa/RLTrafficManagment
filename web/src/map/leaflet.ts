import L from 'leaflet';

// leaflet.markercluster is a UMD plugin that extends the global `L`. The dev
// server happens to provide one; the production bundle does not, so set it
// here. Import this module before 'leaflet.markercluster': ES modules run
// their imports in order, so the global exists when the plugin loads.
(window as unknown as { L: typeof L }).L = L;

export default L;
