// Phase 2 design tokens. CSS variables in src/styles/tokens.css are the
// source of truth; this file only maps Tailwind class names onto them.
const v = (name) => `var(--${name})`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    // Set at theme level (not extend) so Tailwind's defaults are replaced.
    screens: { tablet: '768px', desktop: '1280px', wide: '1440px', room: '1920px' },
    borderRadius: { none: '0', sm: '2px', DEFAULT: '4px', md: '4px', lg: '6px' },
    extend: {
      colors: {
        bg: v('color-bg'),
        panel: v('color-panel'),
        raised: v('color-raised'),
        border: {
          subtle: v('color-border-subtle'),
          DEFAULT: v('color-border'),
          strong: v('color-border-strong'),
          control: v('color-border-control'),
        },
        text: {
          1: v('color-text-1'),
          2: v('color-text-2'),
          3: v('color-text-3'),
          disabled: v('color-text-disabled'),
        },
        accent: { DEFAULT: v('color-accent'), weak: v('color-accent-weak'), on: v('color-on-accent') },
        success: { DEFAULT: v('color-success'), weak: v('color-success-weak') },
        warning: { DEFAULT: v('color-warning'), weak: v('color-warning-weak') },
        danger: { DEFAULT: v('color-danger'), weak: v('color-danger-weak') },
        info: { DEFAULT: v('color-info'), weak: v('color-info-weak') },
        traffic: {
          free: v('traffic-free'),
          moderate: v('traffic-moderate'),
          heavy: v('traffic-heavy'),
          blocked: v('traffic-blocked'),
        },
        vehicle: {
          standby: v('vehicle-standby'),
          'en-route': v('vehicle-en-route'),
          rerouting: v('vehicle-rerouting'),
          arrived: v('vehicle-arrived'),
          unavailable: v('vehicle-unavailable'),
        },
        hospital: {
          available: v('hospital-available'),
          busy: v('hospital-busy'),
          full: v('hospital-full'),
          unknown: v('hospital-unknown'),
        },
        route: {
          optimized: v('route-optimized'),
          baseline: v('route-baseline'),
          alternative: v('route-alternative'),
          blocked: v('route-blocked'),
        },
        heat: { 1: v('heat-1'), 2: v('heat-2'), 3: v('heat-3'), 4: v('heat-4'), 5: v('heat-5') },
        series: {
          'va-qpso': v('series-va-qpso'),
          'fixed-qpso': v('series-fixed-qpso'),
          pso: v('series-pso'),
          ga: v('series-ga'),
          sa: v('series-sa'),
          dijkstra: v('series-dijkstra'),
        },
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        'metric-xl': ['40px', { lineHeight: '44px', fontWeight: '600' }],
        'metric-lg': ['28px', { lineHeight: '32px', fontWeight: '600' }],
        'metric-md': ['20px', { lineHeight: '24px', fontWeight: '500' }],
        'title-lg': ['20px', { lineHeight: '28px', fontWeight: '600' }],
        title: ['16px', { lineHeight: '24px', fontWeight: '600' }],
        body: ['14px', { lineHeight: '20px' }],
        'body-sm': ['13px', { lineHeight: '18px' }],
        label: ['12px', { lineHeight: '16px', fontWeight: '500', letterSpacing: '0.01em' }],
        caption: ['12px', { lineHeight: '16px' }],
      },
      spacing: {
        0.5: '2px', 1: '4px', 2: '8px', 3: '12px', 4: '16px', 5: '20px',
        6: '24px', 8: '32px', 10: '40px', 12: '48px', 16: '64px',
      },
      boxShadow: { 1: v('elev-1'), 2: v('elev-2'), 3: v('elev-3'), none: 'none' },
      zIndex: {
        map: '0', 'map-overlay': '1000', panel: '1100', statusbar: '1200',
        popover: '1300', toast: '1400', dialog: '1500',
      },
      transitionDuration: { instant: '0ms', fast: '120ms', base: '200ms', slow: '320ms' },
      transitionTimingFunction: {
        standard: 'cubic-bezier(0.2, 0, 0, 1)',
        exit: 'cubic-bezier(0.4, 0, 1, 1)',
      },
      width: { panel: v('shell-panel-w'), nav: v('shell-nav-w') },
      height: { header: v('shell-header-h'), statusbar: v('shell-statusbar-h'), dock: v('shell-dock-h') },
    },
  },
  plugins: [],
};
