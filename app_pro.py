# app.py
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.graph_builder import build_graph


# =============================
# STREAM HELPERS
# =============================
def stream_text(text: str, delay: float = 0.02):
    """Потоковый вывод текста с задержкой."""
    for ch in text:
        yield ch
        time.sleep(delay)


# =============================
# INIT
# =============================
st.set_page_config(
    page_title="ИИ агент",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🤖 ИИ агент")
st.caption("Интеллектуальный поиск с автокоррекцией запросов")


@st.cache_resource(show_spinner="Загружаю RAG систему...")
def get_graph():
    """Кэшированная загрузка графа."""
    return build_graph()


# Загрузка графа
try:
    graph = get_graph()
except Exception as e:
    st.error(f"Ошибка загрузки графа: {e}")
    st.stop()

# Инициализация session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "meta" not in st.session_state:
    st.session_state.meta = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_queries": 0,
        "total_time": 0.0,
        "tool_uses": 0
    }

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("⚙️ Настройки")

    # Debug режим
    st.session_state.debug_mode = st.checkbox(
        "🐛 Режим отладки",
        value=st.session_state.debug_mode,
        help="Показывать детальную мета-информацию о каждом запросе"
    )

    # История сообщений
    MAX_HISTORY = st.slider(
        "📜 История сообщений",
        min_value=4,
        max_value=20,
        value=12,
        step=2,
        help="Количество последних сообщений для контекста"
    )

    # Скорость печати
    typing_speed = st.select_slider(
        "⚡ Скорость печати",
        options=["Медленно", "Нормально", "Быстро"],
        value="Нормально",
        help="Скорость потокового вывода ответа"
    )

    delay_map = {
        "Медленно": 0.03,
        "Нормально": 0.015,
        "Быстро": 0.005
    }
    typing_delay = delay_map[typing_speed]

    st.divider()

    # Статистика
    st.subheader("📊 Статистика")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Сообщений", len(st.session_state.messages))
    with col2:
        st.metric("Запросов", st.session_state.stats["total_queries"])

    if st.session_state.stats["total_queries"] > 0:
        avg_time = st.session_state.stats["total_time"] / st.session_state.stats["total_queries"]
        st.metric("⏱Среднее время", f"{avg_time:.2f}s")

    st.metric("🔧 Использований инструментов", st.session_state.stats["tool_uses"])

    st.divider()

    # Экспорт истории
    if st.session_state.messages:
        st.subheader(" Экспорт")

        # Генерация текста истории
        chat_history = []
        for msg, meta in zip(st.session_state.messages, st.session_state.meta):
            if isinstance(msg, HumanMessage):
                chat_history.append(f"Вы: {msg.content}")
            elif isinstance(msg, AIMessage):
                chat_history.append(f"Ассистент: {msg.content}")
                if meta.get("tool"):
                    chat_history.append(f"   [Использован инструмент: {meta['tool']}]")

        export_text = "\n\n".join(chat_history)

        st.download_button(
            label="Скачать историю",
            data=export_text,
            file_name=f"chat_history_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # Очистка истории
    if st.button("🗑️ Очистить историю", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.meta = []
        st.session_state.stats = {
            "total_queries": 0,
            "total_time": 0.0,
            "tool_uses": 0
        }
        st.rerun()

    # Footer
    st.divider()
    st.caption("Powered by LangGraph + ChromaDB")

# =============================
# WELCOME SCREEN
# =============================
if not st.session_state.messages:
    st.markdown("""
    ### 👋 Добро пожаловать!

    Я интеллектуальный ассистент с доступом к базе знаний. Умею:

    - 🔍 **Искать информацию** в векторной базе данных
    - 🤔 **Переформулировать вопросы** для лучших результатов
    - 🎯 **Самокорректироваться** при нерелевантных результатах

    ---

    #### 💡 Попробуйте спросить:
    """)

    # Примеры вопросов
    col1, col2, col3 = st.columns(3)

    examples = [
        ("🎯 Reward hacking", "Что такое reward hacking?"),
        ("🌀 Hallucination", "Объясни hallucination в LLM"),
        ("🎨 Diffusion models", "Как работают diffusion models?")
    ]

    for col, (label, question) in zip([col1, col2, col3], examples):
        with col:
            if st.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state.example_prompt = question
                st.rerun()

# =============================
# CHAT HISTORY
# =============================
for i, (msg, meta) in enumerate(zip(st.session_state.messages, st.session_state.meta)):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"

    with st.chat_message(role):
        st.write(msg.content)

        # Показать мета-информацию в debug режиме
        if st.session_state.debug_mode and role == "assistant" and meta.get("tool"):
            with st.expander("🔍 Детали запроса"):
                st.json({
                    "tool": meta["tool"],
                    "args": meta["args"],
                    "result_preview": meta["result"][:200] + "..." if len(meta["result"]) > 200 else meta["result"]
                })

# =============================
# USER INPUT
# =============================
# Обработка примера или реального ввода
prompt = None

if hasattr(st.session_state, 'example_prompt'):
    prompt = st.session_state.example_prompt
    del st.session_state.example_prompt
else:
    prompt = st.chat_input("Введите сообщение...")

if prompt:
    # ---- USER MESSAGE ----
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    # Ограничение истории
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
    st.session_state.meta = st.session_state.meta[-MAX_HISTORY:]

    prev_len = len(st.session_state.messages)

    # ---- GRAPH CALL ----
    start_time = time.time()

    try:
        with st.spinner("🔍 Анализирую вопрос и ищу информацию..."):
            result = graph.invoke({"messages": st.session_state.messages})

        elapsed_time = time.time() - start_time
        messages = result["messages"]
        ai_msg = messages[-1]

        # Обновление статистики
        st.session_state.stats["total_queries"] += 1
        st.session_state.stats["total_time"] += elapsed_time

        # ---- ASSISTANT (STREAMING) ----
        with st.chat_message("assistant"):
            streamed_text = st.write_stream(
                stream_text(ai_msg.content, delay=typing_delay)
            )

            # Показать метрики
            if st.session_state.debug_mode:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Время", f"{elapsed_time:.2f}s")
                with col2:
                    rewrites = result.get("rewrite_count", 0)
                    st.metric("🔄 Попыток", rewrites + 1)
                with col3:
                    st.metric("💬 Сообщений", len(messages))

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
                        st.session_state.stats["tool_uses"] += 1
                        break
                break

        if tool_meta:
            st.session_state.meta.append(tool_meta)
        else:
            st.session_state.meta.append({"tool": None})

        # =============================
        # META DISPLAY
        # =============================
        if not st.session_state.debug_mode:  # Показываем только если не debug режим
            st.divider()
            st.subheader("📋 Мета-информация последнего запроса")

            last = st.session_state.meta[-1]

            if last.get("tool") is None:
                st.info("ℹ️ Инструменты не задействованы - ответ сгенерирован напрямую")
            else:
                with st.expander("🔧 Детали использования инструментов", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**Инструмент:**")
                        st.code(last['tool'])

                        st.markdown("**Аргументы:**")
                        st.json(last['args'])

                    with col2:
                        st.markdown("**Результат поиска:**")
                        result_preview = last['result'][:300] + "..." if len(last['result']) > 300 else last['result']
                        st.text_area(
                            label="Найденные документы",
                            value=result_preview,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )

    except Exception as e:
        st.error(f"❌ Ошибка при обработке запроса: {str(e)}")

        if st.session_state.debug_mode:
            with st.expander("🐛 Техническая информация об ошибке"):
                st.exception(e)