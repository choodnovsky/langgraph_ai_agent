# app_pro.py
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.graph_builder import build_graph


def stream_text(text: str, delay: float = 0.015):
    for ch in text:
        yield ch
        time.sleep(delay)


# =============================
# INIT
# =============================
st.set_page_config(page_title="ИИ агент", layout="wide")
st.title("🤖 ИИ агент")


@st.cache_resource(show_spinner="Загружаю RAG систему...")
def get_graph():
    return build_graph()


try:
    graph = get_graph()
except Exception as e:
    st.error(f"Ошибка загрузки графа: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "meta" not in st.session_state:
    st.session_state.meta = []


# =============================
# SIDEBAR — найденный документ
# =============================
with st.sidebar:
    st.header("📄 Найденный фрагмент")

    if st.session_state.meta:
        last = st.session_state.meta[-1]
        if last.get("tool"):
            st.caption(f"Инструмент: `{last['tool']}`")
            st.caption(f"Запрос: `{last['args']}`")
            st.divider()
            st.text_area(
                label="Текст из базы знаний",
                value=last["result"],
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("Поиск не использовался — ответ сгенерирован напрямую")
    else:
        st.caption("Здесь появится текст документа после первого запроса")

    st.divider()
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.messages = []
        st.session_state.meta = []
        st.rerun()


# =============================
# CHAT HISTORY
# =============================
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)


# =============================
# USER INPUT
# =============================
if prompt := st.chat_input("Введите сообщение..."):
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages = st.session_state.messages[-12:]
    st.session_state.meta = st.session_state.meta[-12:]

    prev_len = len(st.session_state.messages)

    try:
        with st.spinner("Ищу информацию..."):
            result = graph.invoke({"messages": st.session_state.messages})

        messages = result["messages"]
        ai_msg = messages[-1]

        with st.chat_message("assistant"):
            streamed_text = st.write_stream(stream_text(ai_msg.content))

        st.session_state.messages.append(AIMessage(content=streamed_text))

        # Извлекаем мета из текущего хода
        new_messages = messages[prev_len:]
        tool_meta = None

        for i, msg in enumerate(new_messages):
            if isinstance(msg, ToolMessage):
                tool_result = msg.content
                for prev in reversed(new_messages[:i]):
                    if isinstance(prev, AIMessage) and prev.tool_calls:
                        call = prev.tool_calls[0]
                        tool_meta = {
                            "tool": call["name"],
                            "args": call["args"],
                            "result": tool_result,
                        }
                        break
                break

        st.session_state.meta.append(tool_meta if tool_meta else {"tool": None})

    except Exception as e:
        st.error(f"❌ Ошибка при обработке запроса: {str(e)}")

    st.rerun()