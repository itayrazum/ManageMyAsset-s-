"""App theme.

Injects custom CSS so the app doesn't render in Streamlit's default look: a deep-navy dark
theme with a teal accent, the Inter font, a gradient title, rounded chat cards, and the default
Streamlit chrome hidden. The Streamlit config (.streamlit/config.toml) sets the matching dark base.
"""


def theme_css() -> str:
    """Return a <style> block styling the whole app (dark)."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:#0c1120; --bg2:#111a2e; --surface:#151f38;
  --text:#e7eef9; --muted:#94a3bd; --border:#233150;
  --accent:#2dd4bf; --accent2:#38bdf8; --glow:rgba(45,212,191,0.12);
}

/* Inter on text + form controls only. Do NOT target span/icon classes, or the
   Material icon font breaks and icons show as raw text (e.g. keyboard_double_arrow_left). */
html, body, .stApp, button, input, textarea { font-family:'Inter', system-ui, -apple-system, sans-serif !important; }
[data-testid="stIconMaterial"], span[class*="material"] {
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}

.stApp { background:var(--bg); color:var(--text); }
.stApp::before {
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background: radial-gradient(1200px 520px at 12% -12%, var(--glow), transparent 60%);
}
.block-container { padding-top:2.2rem; }

/* hide Streamlit's default chrome */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
[data-testid="stHeader"] { background:transparent; }

/* base text */
.stApp, .stMarkdown, p, li, label { color:var(--text); }
[data-testid="stCaptionContainer"], .subtitle { color:var(--muted) !important; }
h1, h2, h3, h4 { color:var(--text); font-weight:700; letter-spacing:-0.015em; }

/* custom header */
.hero { font-size:2.1rem; font-weight:800; letter-spacing:-0.02em; margin:0 0 .1rem; }
.hero .logo { margin-right:.45rem; }
.hero .grad {
  background:linear-gradient(92deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle { font-size:.95rem; margin:0 0 1.1rem; }

/* sidebar */
[data-testid="stSidebar"] { background:var(--bg2); border-right:1px solid var(--border); }

/* chat bubbles */
[data-testid="stChatMessage"] {
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:.35rem 1rem; margin-bottom:.5rem;
  box-shadow:0 2px 10px rgba(0,0,0,.10);
}

/* chat input */
[data-testid="stChatInput"] { border:1px solid var(--border); border-radius:14px; background:var(--bg2); }
[data-testid="stChatInput"] textarea { background:transparent !important; color:var(--text) !important; }

/* buttons */
.stButton button {
  background:var(--bg2); color:var(--text); border:1px solid var(--border);
  border-radius:11px; font-weight:500; transition:all .15s ease;
}
.stButton button:hover { border-color:var(--accent); color:var(--accent); box-shadow:0 0 0 3px var(--glow); }

/* expander + status cards (the agent trace) */
[data-testid="stExpander"] { border:1px solid var(--border); border-radius:12px; background:var(--surface); overflow:hidden; }
[data-testid="stExpander"] summary:hover { color:var(--accent); }
[data-testid="stStatus"] { border:1px solid var(--border); border-radius:12px; background:var(--surface); }

/* accents */
[data-baseweb="toggle"] div[aria-checked="true"] { background:var(--accent) !important; }
a { color:var(--accent); }
code, pre { background:var(--surface) !important; border:1px solid var(--border); border-radius:8px; }
</style>
"""
