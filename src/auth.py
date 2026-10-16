"""Google OAuth 2.0 authentication layer for the Mortgage Exception Workbench.

Usage
-----
Call ``require_auth()`` at the very top of app.py (before any other content).
It will:
  1. Redirect unauthenticated visitors to the Google OAuth consent screen.
  2. Exchange the auth-code for tokens and cache the id_token in Streamlit
     session state.
  3. Decode the id_token (JWT) to extract the user email and name without a
     network call (the signature is already verified by streamlit-oauth).
  4. Derive the workbench role from the email:
       - Email in REVIEWER_EMAILS env var → "reviewer"
       - Email domain in REVIEWER_DOMAINS env var → "reviewer"
       - Otherwise → "viewer"
  5. Return an ``Identity`` dict with ``email``, ``name``, and ``role``.

Required .env vars
------------------
  GOOGLE_CLIENT_ID       — OAuth 2.0 client ID from Google Cloud Console
  GOOGLE_CLIENT_SECRET   — OAuth 2.0 client secret
  OAUTH_REDIRECT_URI     — Must match the URI registered in Google Cloud Console
                           (e.g. http://localhost:8501/oauth2callback for local dev)

Optional .env vars
------------------
  REVIEWER_EMAILS        — Comma-separated list of emails granted reviewer role
                           (e.g. alice@example.com,bob@example.com)
  REVIEWER_DOMAINS       — Comma-separated list of domains whose users are
                           reviewers (e.g. mycompany.com)

If GOOGLE_CLIENT_ID is absent the module falls back to the legacy role-selector
sidebar so the app stays runnable without OAuth credentials (dev/demo mode).
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # picks up .env in the project root

# Lazy import so the app loads even when streamlit-oauth is not installed.
try:
    from streamlit_oauth import OAuth2Component  # type: ignore
    _OAUTH_AVAILABLE = True
except ImportError:
    _OAUTH_AVAILABLE = False


class Identity(TypedDict):
    email: str
    name: str
    role: str          # "reviewer" | "viewer"
    authenticated: bool


_SESSION_KEY = "workbench_identity"


def _reviewer_emails() -> set[str]:
    raw = os.environ.get("REVIEWER_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _reviewer_domains() -> set[str]:
    raw = os.environ.get("REVIEWER_DOMAINS", "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def _derive_role(email: str) -> str:
    """Map an authenticated email to a workbench role."""
    email_lower = email.lower()
    if email_lower in _reviewer_emails():
        return "reviewer"
    domain = email_lower.split("@")[-1] if "@" in email_lower else ""
    if domain and domain in _reviewer_domains():
        return "reviewer"
    return "viewer"


def _decode_id_token_payload(token: str) -> dict:
    """Decode the JWT payload (middle segment) without signature verification.

    streamlit-oauth already verified the token via the token endpoint; we just
    need the claims for display and role derivation.
    """
    try:
        segments = token.split(".")
        if len(segments) < 2:
            return {}
        # Add padding so base64 decodes cleanly.
        padded = segments[1] + "=" * (4 - len(segments[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        return {}


def _build_oauth_component() -> Optional["OAuth2Component"]:
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8501")
    if not client_id or not client_secret:
        return None
    return OAuth2Component(
        client_id=client_id,
        client_secret=client_secret,
        authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )


def require_auth() -> Identity:
    """Gate the app behind Google OAuth.  Returns the current Identity.

    If GOOGLE_CLIENT_ID is not set, falls back to the dev-mode role selector
    so the app remains runnable without credentials.
    """
    # ── Return cached identity if already authenticated ──────────────────
    if _SESSION_KEY in st.session_state:
        return st.session_state[_SESSION_KEY]  # type: ignore[return-value]

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")

    # ── Dev / demo fallback: no IdP configured ───────────────────────────
    if not client_id or not _OAUTH_AVAILABLE:
        from src.exception_workbench import ROLE_REVIEWER, ROLE_VIEWER  # avoid circular at module level
        st.sidebar.header("Access & Permissions")
        role = st.sidebar.selectbox(
            "Active Role",
            options=[ROLE_REVIEWER, ROLE_VIEWER],
            index=0,
            format_func=lambda r: (
                "Reviewer (can change cases)" if r == ROLE_REVIEWER else "Viewer (look only)"
            ),
            help="Set GOOGLE_CLIENT_ID in .env to enable real OAuth authentication.",
        )
        if not client_id:
            st.sidebar.caption(
                "⚠️ No OAuth credentials configured. "
                "Set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env to enable login."
            )
        identity: Identity = {
            "email": f"{role}@workbench.local",
            "name": role.capitalize(),
            "role": role,
            "authenticated": False,
        }
        st.session_state[_SESSION_KEY] = identity
        return identity

    # ── Real OAuth flow ───────────────────────────────────────────────────
    oauth = _build_oauth_component()
    if oauth is None:
        st.error("OAuth configuration is incomplete. Check GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env.")
        st.stop()

    redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8501")
    result = oauth.authorize_button(
        name="Sign in with Google",
        icon="https://www.google.com/favicon.ico",
        redirect_uri=redirect_uri,
        scope="openid email profile",
        key="google_oauth_btn",
        extras_params={"prompt": "consent", "access_type": "offline"},
        use_container_width=True,
        pkce="S256",
    )

    if result and result.get("token"):
        token_response = result["token"]
        id_token = token_response.get("id_token", "")
        claims = _decode_id_token_payload(id_token) if id_token else {}

        email = claims.get("email", token_response.get("email", "unknown@unknown"))
        name = claims.get("name", email.split("@")[0].title())
        role = _derive_role(email)

        identity = {
            "email": email,
            "name": name,
            "role": role,
            "authenticated": True,
        }
        st.session_state[_SESSION_KEY] = identity
        st.rerun()

    # Not yet authenticated — stop here so the rest of app.py doesn't render.
    st.stop()


def get_identity() -> Identity:
    """Return the cached Identity; raises if require_auth() was not called."""
    identity = st.session_state.get(_SESSION_KEY)
    if identity is None:
        raise RuntimeError("require_auth() must be called before get_identity()")
    return identity  # type: ignore[return-value]


def logout() -> None:
    """Clear the session identity (user will be redirected to OAuth on next render)."""
    st.session_state.pop(_SESSION_KEY, None)
