# pages/chat.py
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from modules.auth import require_auth
from modules.feedback import init_feedback_table, render_feedback
from graph.builder import build_graph

st.set_page_config(page_title="Чат", layout="wide")

thread_id = require_auth()
init_feedback_table()


@st.cache_resource(show_spinner="Загружаю систему...")
def get_graph():
    return build_graph(use_checkpointer=True)


try:
    graph = get_graph()
except Exception as e:
    st.error(f"Ошибка загрузки графа: {e}")
    st.stop()


def stream_text(text: str, delay: float = 0.015):
    for ch in text:
        yield ch
        time.sleep(delay)


config = {"configurable": {"thread_id": thread_id}}


def get_interrupt(config):
    """Проверяет находится ли граф на паузе (interrupt)."""
    state = graph.get_state(config)
    if state and state.tasks:
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


# ── История ──────────────────────────────────────────────────────────────────
state = graph.get_state(config)
history = state.values.get("messages", []) if state and state.values else []

for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        with st.chat_message("assistant"):
            st.write(msg.content)
            render_feedback(msg.id, thread_id)

# ── Проверка interrupt ────────────────────────────────────────────────────────
interrupt_data = get_interrupt(config)

if interrupt_data:
    st.info("Найден документ. Использовать его для ответа?")
    st.caption(interrupt_data.get("doc_preview", ""))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Да, использовать", use_container_width=True):
            with st.spinner("Генерирую ответ..."):
                result = graph.invoke(Command(resume="yes"), config=config)
            ai_msg = result["messages"][-1]
            with st.chat_message("assistant"):
                st.write_stream(stream_text(ai_msg.content))
                render_feedback(ai_msg.id, thread_id)
            st.rerun()
    with col2:
        if st.button("Нет, искать другой", use_container_width=True):
            with st.spinner("Ищу другой источник..."):
                result = graph.invoke(Command(resume="no"), config=config)
            if result["messages"][-1].content:
                ai_msg = result["messages"][-1]
                with st.chat_message("assistant"):
                    st.write_stream(stream_text(ai_msg.content))
                    render_feedback(ai_msg.id, thread_id)
            st.rerun()

# ── Новый вопрос ──────────────────────────────────────────────────────────────
elif prompt := st.chat_input("Введите сообщение..."):
    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("Ищу информацию..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

        # Граф мог остановиться на interrupt
        interrupt_data = get_interrupt(config)
        if interrupt_data:
            st.rerun()
        else:
            ai_msg = result["messages"][-1]
            with st.chat_message("assistant"):
                st.write_stream(stream_text(ai_msg.content))
                render_feedback(ai_msg.id, thread_id)
            st.rerun()

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.exception(e)