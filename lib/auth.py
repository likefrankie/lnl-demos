"""Passcode gate for the hosted app.

The passcode lives in Streamlit secrets (`passcode = "..."`), set in the Streamlit Cloud
app settings or in `.streamlit/secrets.toml`. With no passcode set (a local run), the
gate is skipped. Every page calls `require_passcode()`, so a direct page URL is gated too.
"""

import hmac

import streamlit as st

from lib import theme


def _passcode():
    try:
        return st.secrets.get("passcode")
    except Exception:  # no secrets file at all
        return None


def require_passcode():
    code = _passcode()
    if not code or st.session_state.get("passcode_ok"):
        return

    theme.bar("LNL ▸ STATISTICAL MODELS ▸ DEMONSTRATIONS", "ACCESS")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.form("passcode_form", border=True):
        theme.side_header("ENTER THE CLASS PASSCODE")
        entered = st.text_input("passcode", type="password", label_visibility="collapsed")
        if st.form_submit_button("▶  ENTER", type="primary", width="stretch"):
            if hmac.compare_digest(entered.strip(), str(code)):
                st.session_state.passcode_ok = True
                st.rerun()
            st.error("Wrong passcode.")
    st.stop()
