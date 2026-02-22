# graph/nodes/summarizer.py
"""
Узел суммаризации истории переписки.

Когда сообщений становится больше порога — сворачивает историю в короткую сводку.
Это экономит токены: вместо всей истории модель видит сводку + последние N сообщений.
"""

from typing import Literal
from langchain_core.messages import HumanMessage, RemoveMessage

# Сколько сообщений хранить после суммаризации
MESSAGES_TO_KEEP = 4

# Порог для запуска суммаризации
SUMMARIZE_AFTER = 10


def should_summarize(state) -> Literal["summarizer", "__end__"]:
    """Conditional edge: суммаризировать или завершить."""
    if len(state["messages"]) > SUMMARIZE_AFTER:
        return "summarizer"
    return "__end__"


def summarize_conversation(state):
    """Сворачивает историю в сводку, удаляет старые сообщения.

    Алгоритм:
    1. Если сводка уже есть — дополняем её
    2. Если нет — создаём с нуля
    3. Удаляем все сообщения кроме MESSAGES_TO_KEEP последних
    4. Сохраняем сводку в state["summary"]
    """
    from graph.nodes.query import get_response_model

    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"Это сводка разговора на данный момент: {summary}\n\n"
            "Дополни сводку, учитывая новые сообщения выше. "
            "Сохраняй все важные факты: имена, числа, решения, контекст:"
        )
    else:
        summary_message = (
            "Создай краткую сводку разговора выше. "
            "Включи все важные факты: имена, числа, решения, контекст:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]

    model = get_response_model()
    response = model.invoke(messages)

    # Удаляем старые сообщения, оставляем только последние MESSAGES_TO_KEEP
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][:-MESSAGES_TO_KEEP]
    ]

    print(f"[summarizer] История свёрнута. Удалено: {len(delete_messages)} сообщений, осталось: {MESSAGES_TO_KEEP}")

    return {
        "summary": response.content,
        "messages": delete_messages,
    }
