# app.py
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.graph_builder import build_graph


# =============================
# STREAM HELPERS
# =============================
def stream_text(text: str, delay: float = 0.02):
    for ch in text:
        yield ch
        time.sleep(delay)


# =============================
# INIT
# =============================
st.set_page_config(page_title="ИИ агент", layout="centered")
st.title("🤖 ИИ агент")


@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "meta" not in st.session_state:
    st.session_state.meta = []

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
if prompt := st.chat_input("Введите сообщение"):
    # ---- USER MESSAGE ----
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    MAX_HISTORY = 12
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    prev_len = len(st.session_state.messages)

    # ---- GRAPH CALL ----
    result = graph.invoke({"messages": st.session_state.messages})
    messages = result["messages"]

    ai_msg = messages[-1]

    # ---- ASSISTANT (STREAMING) ----
    with st.chat_message("assistant"):
        streamed_text = st.write_stream(
            stream_text(ai_msg.content, delay=0.015)
        )

    st.session_state.messages.append(
        AIMessage(content=streamed_text)
    )

    # =============================
    # META (ТОЛЬКО ТЕКУЩИЙ ХОД)
    # =============================
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

    if tool_meta:
        st.session_state.meta.append(tool_meta)
    else:
        st.session_state.meta.append({"tool": None})

# =============================
# META DISPLAY
# =============================
if st.session_state.meta:
    st.divider()
    st.subheader("Мета-информация")

    last = st.session_state.meta[-1]

    if last.get("tool") is None:
        st.markdown("**Инструменты не задействованы**")
    else:
        st.markdown(
            f"""
**Инструмент:** `{last['tool']}`  
**Аргументы:** `{last['args']}`  
**Результат:** `{last['result']}`
"""
        )