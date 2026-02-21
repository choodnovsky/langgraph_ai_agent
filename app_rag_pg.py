# app.py
import time
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
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

# username → thread_id для памяти в Postgres
thread_id = st.session_state["username"]
config_graph = {"configurable": {"thread_id": thread_id}}

if "meta" not in st.session_state:
    st.session_state.meta = []


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.title("🤖 ИИ агент")
    st.caption(f"Привет, **{st.session_state['name']}**")
    authenticator.logout("Выйти", location="sidebar")

    st.divider()
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


# =============================
# CHAT HISTORY — из Postgres
# =============================
st.title("🤖 ИИ агент")

state = graph.get_state(config_graph)
history = state.values.get("messages", []) if state and state.values else []

for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)


# =============================
# USER INPUT
# =============================
if prompt := st.chat_input("Введите сообщение..."):
    with st.chat_message("user"):
        st.write(prompt)

    prev_len = len(history)

    try:
        with st.spinner("Ищу информацию..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config_graph,
            )

        messages = result["messages"]
        ai_msg = messages[-1]

        with st.chat_message("assistant"):
            st.write_stream(stream_text(ai_msg.content))

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