import { Component, type ReactNode } from 'react';

interface State {
  error: Error | null;
}

/** Catches render errors so the page shows the cause instead of going blank. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: unknown) {
    // Surface to the console for debugging.
    console.error('[Arena] render error:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: 24, fontFamily: 'monospace', color: '#1a1a1a',
          background: '#f5f6f3', minHeight: '100vh',
        }}>
          <h1 style={{ color: '#ef4444', fontSize: 20, marginBottom: 12 }}>
            Frontend crashed — {this.state.error.name}
          </h1>
          <pre style={{
            whiteSpace: 'pre-wrap', background: '#fff', padding: 16,
            borderRadius: 8, border: '1px solid #dcdcdc', fontSize: 13,
          }}>
            {this.state.error.message}
            {'\n\n'}
            {this.state.error.stack}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: 16, padding: '8px 16px', borderRadius: 8,
              background: '#2f3a55', color: '#fff', border: 'none', cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
