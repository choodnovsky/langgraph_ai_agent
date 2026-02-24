# app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from modules.auth import get_authenticator, require_auth

st.set_page_config(page_title="ИИ агент", layout="wide")

authenticator = get_authenticator()
authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    st.error("Неверный логин или пароль")
    st.stop()

if st.session_state.get("authentication_status") is None:
    st.stop()

with st.sidebar:
    st.caption(f"**{st.session_state['name']}**")
    authenticator.logout("Выйти", location="sidebar")

st.title(f"Добро пожаловать, {st.session_state['name']}")
st.write("Выберите раздел в меню слева.")