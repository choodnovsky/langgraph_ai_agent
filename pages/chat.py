# pages/chat.py
import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def stream_text(text: str, delay: float = 0.015):
    """Добавляет эффект печатания текста"""
    for ch in text:
        yield ch
        time.sleep(delay)


def chat_page(graph, thread_id: str):
    """Страница чата. Принимает граф и thread_id от авторизованного пользователя."""

    config = {"configurable": {"thread_id": thread_id}}

    if "last_meta" not in st.session_state:
        st.session_state.last_meta = None

    # =============================
    # SIDEBAR — найденный документ
    # =============================
    with st.sidebar:
        st.header("📄 Найденный фрагмент")

        if st.session_state.last_meta:
            last = st.session_state.last_meta
            if last.get("tool"):
                st.success("Ответ на основании информации в хранилище знаний")
                st.caption(f"Инструмент: `{last['tool']}`")
                st.caption(f"Запрос: `{last['args']}`")
                st.divider()
                st.text_area(
                    label="Текст из базы знаний",
                    value=last["result"],
                    height=400,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.warning("Ответ сгенерирован моделью")
        else:
            st.caption("Здесь появится текст документа после первого запроса")

    # =============================
    # CHAT HISTORY — из Postgres
    # =============================
    state = graph.get_state(config)
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

        # Длина истории ДО invoke
        prev_len = len(history)

        try:
            with st.spinner("Ищу информацию..."):
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )

            messages = result["messages"]
            ai_msg = messages[-1]

            with st.chat_message("assistant"):
                st.write_stream(stream_text(ai_msg.content))

            # Новые сообщения = всё что появилось после invoke
            # graph.invoke возвращает полную историю, берём хвост начиная с prev_len
            new_messages = messages[prev_len:]
            tool_meta = None
            for i, msg in enumerate(new_messages):
                if isinstance(msg, ToolMessage):
                    tool_result = msg.content
                    for j, prev in enumerate(reversed(new_messages[:i])):
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
                # Новый поиск — показываем свежую мету
                st.session_state.last_meta = tool_meta
            else:
                # Поиска не было — прямой ответ модели
                st.session_state.last_meta = {"tool": None}
            st.rerun()

        except Exception as e:
            st.error(f"❌ Ошибка при обработке запроса: {str(e)}")
            # st.exception(e)