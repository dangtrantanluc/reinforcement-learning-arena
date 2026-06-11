/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Structural colours come from CSS variables so a `.dark` class can
        // override them globally without touching component class names.
        primary: 'rgb(var(--c-primary) / <alpha-value>)',
        secondary: 'rgb(var(--c-secondary) / <alpha-value>)',
        bg: 'rgb(var(--c-bg) / <alpha-value>)',
        surface: 'rgb(var(--c-surface) / <alpha-value>)',
        ink: 'rgb(var(--c-ink) / <alpha-value>)',
        sub: 'rgb(var(--c-sub) / <alpha-value>)',
        accent: 'rgb(var(--c-accent) / <alpha-value>)',
        line: 'rgb(var(--c-line) / <alpha-value>)',
        // Semantic + agent colours are theme-agnostic (work on light & dark).
        success: '#22c55e',
        danger: '#ef4444',
        warning: '#f59e0b',
        ppo: '#3b82f6',
        'ppo-soft': 'rgb(var(--c-ppo-soft) / <alpha-value>)',
        dynaq: '#a78bfa',
        'dynaq-soft': 'rgb(var(--c-dynaq-soft) / <alpha-value>)',
        'dqn-c': '#2dd4bf',
        'dqn-soft': 'rgb(var(--c-dqn-soft) / <alpha-value>)',
        box: '#d97706',
        gold: '#eab308',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06)',
        panel: '0 4px 16px rgba(47,58,85,0.08)',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(34,197,94,0.5)' },
          '50%': { boxShadow: '0 0 16px 4px rgba(34,197,94,0.7)' },
        },
      },
      animation: {
        'pulse-glow': 'pulse-glow 1s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
