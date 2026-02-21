# app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from streamlit_option_menu import option_menu

from src.graph_builder import build_graph
from chat import chat_page


# =============================
# INIT
# =============================
st.set_page_config(page_title="ИИ агент", layout="wide")


@st.cache_resource(show_spinner="Загружаю RAG систему...")
def get_graph():
    return build_graph(use_checkpointer=True)


try:
    graph = get_graph()
except Exception as e:
    st.error(f"Ошибка загрузки графа: {e}")
    st.stop()


# =============================
# АВТОРИЗАЦИЯ
# =============================
with open("streamlit_credentials.yaml") as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Неверный логин или пароль")
    st.stop()

if st.session_state.get("authentication_status") is None:
    st.warning("Введите логин и пароль")
    st.stop()

# Пользователь авторизован
thread_id = st.session_state["username"]


# =============================
# МЕНЮ
# =============================
with st.sidebar:
    st.caption(f"Привет, **{st.session_state['name']}**")
    authenticator.logout("Выйти", location="sidebar")
    st.divider()

    choice = option_menu(
        menu_title=None,
        options=["Чат", "О системе"],
        icons=["chat-dots", "info-circle"],
        default_index=0,
    )


# =============================
# РОУТИНГ
# =============================
if choice == "Чат":
    chat_page(graph, thread_id)

elif choice == "О системе":
    st.title("О системе")
    st.markdown("""
    **ИИ агент** — интеллектуальный ассистент с доступом к базе знаний.

    - 🔍 Ищет информацию в векторной базе ChromaDB
    - 🤔 Переформулирует вопросы при нерелевантных результатах
    - 🧠 Помнит историю беседы между сессиями
    - 📄 Показывает источник найденной информации
    """)