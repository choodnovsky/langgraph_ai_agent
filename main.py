import streamlit as st
from streamlit_chat import message
import asyncio
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from src.settings import settings

# -----------------------------
# Арифметические инструменты
# -----------------------------
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]

# -----------------------------
# Инициализация LLM
# -----------------------------
llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.1,
    max_retries=2,
    base_url="https://openrouter.ai/api/v1"
)
llm_with_tools = llm.bind_tools(tools)
sys_msg = SystemMessage(content="Ты помощник для выполнения арифметических операций.")

def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

tools_node = ToolNode(tools=tools)

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tools_node)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

app_graph = builder.compile()

# -----------------------------
# Streamlit UI с Chat режимом
# -----------------------------
st.title("Корпоративный арифметический агент")

# История сообщений
if "history" not in st.session_state:
    st.session_state.history = []

# Состояние графа
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}

# Последний результат для операций
if "last_result" not in st.session_state:
    st.session_state.last_result = None

user_input = st.text_input("Ваше сообщение:")

if st.button("Отправить") and user_input:
    # Подставляем предыдущий результат
    if st.session_state.last_result is not None:
        user_input = user_input.replace("это число", str(st.session_state.last_result))
        user_input = user_input.replace("результат", str(st.session_state.last_result))

    # Добавляем сообщение пользователя в историю
    st.session_state.history.append(("user", user_input))

    # Асинхронно вызываем агента
    async def get_response(message_text: str):
        result = app_graph.invoke(
            input={"messages": [HumanMessage(content=message_text)]},
            state=st.session_state.graph_state
        )
        st.session_state.graph_state = result
        if "messages" in result and result["messages"]:
            content = result["messages"][-1].content
            # Сохраняем число для последующих операций
            numbers = re.findall(r"[-+]?\d*\.?\d+", content)
            if numbers:
                st.session_state.last_result = numbers[-1]
            return content
        return "[Агент не вернул сообщения]"

    response_text = asyncio.run(get_response(user_input))

    # Постепенный вывод ответа бота
    bot_msg_placeholder = st.empty()
    displayed_text = ""
    for char in response_text:
        displayed_text += char
        bot_msg_placeholder.markdown(f"**Агент:** {displayed_text}")
        time.sleep(0.03)  # пауза между символами

    st.session_state.history.append(("bot", response_text))

# -----------------------------
# Отображаем чат
# -----------------------------
for sender, msg in st.session_state.history:
    if sender == "user":
        message(msg, is_user=True)
    else:
        message(msg)