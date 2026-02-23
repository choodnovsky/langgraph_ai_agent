# pages/about.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from modules.auth import require_auth

st.set_page_config(page_title="О системе", layout="wide")

require_auth()

st.title("О системе")
st.markdown("""
Корпоративный ИИ-ассистент с доступом к базе знаний компании.

- Ищет информацию в векторной базе ChromaDB
- Переформулирует вопросы при нерелевантных результатах (до 2 попыток)
- Помнит историю беседы между сессиями (PostgreSQL)
- Автоматически сворачивает историю в сводку при росте контекста

Стек: LangGraph, ChromaDB, PostgreSQL, Streamlit
""")