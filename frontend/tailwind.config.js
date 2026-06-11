/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#2f3a55',
        secondary: '#5c6b8a',
        bg: '#f5f6f3',
        ink: '#1a1a1a',
        sub: '#6e6f73',
        accent: '#a29f76',
        line: '#dcdcdc',
        success: '#22c55e',
        danger: '#ef4444',
        warning: '#f59e0b',
        ppo: '#2563eb',
        'ppo-soft': '#dbeafe',
        dynaq: '#7c3aed',
        'dynaq-soft': '#ede9fe',
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
