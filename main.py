import streamlit as st
from streamlit_chat import message

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from src.settings import settings


# =====================================================
# TOOLS
# =====================================================
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def divide(a: int, b: int) -> float:
    """Divide a by b"""
    return a / b


TOOLS = [add, multiply, divide]


# =====================================================
# GRAPH (CACHE)
# =====================================================
@st.cache_resource
def build_graph():
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_retries=2,
        base_url="https://openrouter.ai/api/v1",
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    system = SystemMessage(
        content=(
            "Ты арифметический помощник.\n"
            "Ты ОБЯЗАН помнить предыдущие результаты.\n"
            "Если требуется вычисление — всегда вызывай инструмент."
        )
    )

    def assistant(state: MessagesState):
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


graph = build_graph()


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Арифметический агент", layout="centered")
st.title("🤖 Арифметический агент")

# -----------------
# SESSION STATE
# -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "meta" not in st.session_state:
    st.session_state.meta = []

# -----------------
# CHAT DISPLAY
# -----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True)
    else:
        message(msg.content)

# -----------------
# INPUT
# -----------------
user_input = st.text_input("Введите сообщение", key="input")

if st.button("Отправить") and user_input:
    # добавляем сообщение пользователя
    st.session_state.messages.append(HumanMessage(content=user_input))

    # ограничиваем историю
    MAX_HISTORY = 12
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # 🔑 запоминаем длину истории ДО вызова
    prev_len = len(st.session_state.messages)

    # вызываем граф
    result = graph.invoke({"messages": st.session_state.messages})

    messages = result["messages"]

    # добавляем ответ ассистента
    ai_msg = messages[-1]
    st.session_state.messages.append(ai_msg)

    # -----------------
    # META (ТОЛЬКО ТЕКУЩИЙ ХОД)
    # -----------------
    new_messages = messages[prev_len:]
    tool_meta = None

    for i, msg in enumerate(new_messages):
        if isinstance(msg, ToolMessage):
            tool_result = msg.content

            # ищем AIMessage с tool_calls ПЕРЕД этим ToolMessage
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

    # если tool не использовался
    if tool_meta:
        st.session_state.meta.append(tool_meta)
    else:
        st.session_state.meta.append({"tool": None})

    st.rerun()


# =====================================================
# META DISPLAY
# =====================================================
if st.session_state.meta:
    st.divider()
    st.subheader("🛠️ Мета-информация")

    last = st.session_state.meta[-1]

    if last.get("tool") is None:
        st.markdown("🛠️ **Инструменты не задействованы**")
    else:
        st.markdown(
            f"""
**Инструмент:** `{last['tool']}`  
**Аргументы:** `{last['args']}`  
**Результат:** `{last['result']}`
"""
        )