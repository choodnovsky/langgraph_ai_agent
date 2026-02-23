# modules/auth.py
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader


def get_authenticator():
    with open("streamlit_credentials.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def require_auth() -> str:
    """Проверяет авторизацию. Показывает имя и кнопку выхода в сайдбаре."""
    if st.session_state.get("authentication_status") is not True:
        st.warning("Необходима авторизация")
        st.page_link("app.py", label="Войти")
        st.stop()

    authenticator = get_authenticator()
    with st.sidebar:
        st.caption(f"**{st.session_state['name']}**")
        authenticator.logout("Выйти", location="sidebar")

    return st.session_state["username"]