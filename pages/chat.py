# pages/chat.py
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from modules.auth import require_auth
from graph.builder import build_graph

st.set_page_config(page_title="Чат", layout="wide")

thread_id = require_auth()


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

state = graph.get_state(config)
history = state.values.get("messages", []) if state and state.values else []

for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        with st.chat_message("assistant"):
            st.write(msg.content)

if prompt := st.chat_input("Введите сообщение..."):
    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("Ищу информацию..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

        ai_msg = result["messages"][-1]

        with st.chat_message("assistant"):
            st.write_stream(stream_text(ai_msg.content))

        st.rerun()

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.exception(e)