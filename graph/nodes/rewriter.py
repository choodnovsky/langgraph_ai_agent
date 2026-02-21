# src/components/rewriter.py

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

REWRITE_PROMPT = (
    "Посмотри на входные данные и попытайся проанализировать базовое семантическое намерение / значение.\n"
    "Предыдущий поисковый запрос не дал релевантных результатов. "
    "Переформулируй вопрос так, чтобы улучшить результаты поиска.\n"
    "Вот исходный вопрос:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Сформулируй улучшенный вопрос на том же языке, что и оригинал. "
    "Используй синонимы и альтернативные формулировки:"
)


def rewrite_question(state: MessagesState):
    """Переформулировать исходный вопрос пользователя для улучшения поиска.

    Этот узел вызывается когда найденные документы нерелевантны.
    Увеличивает счетчик попыток переформулирования.
    """
    # Импортируем модель только когда нужна
    from graph.nodes.query import get_response_model

    messages = state["messages"]

    # ВАЖНО: Берем ПОСЛЕДНИЙ вопрос пользователя (не первый!)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        print("Error: No HumanMessage found")
        question = "Unknown question"
    else:
        question = human_messages[-1].content  # Последний вопрос

    # Увеличиваем счетчик попыток
    rewrite_count = state.get("rewrite_count", 0) + 1


    # Генерируем переформулированный вопрос
    prompt = REWRITE_PROMPT.format(question=question)

    # Получаем модель (инициализируется только при первом вызове)
    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])

    rewritten_question = response.content

    # Возвращаем обновленный state:
    # - Заменяем первое сообщение на переформулированный вопрос
    # - Очищаем все промежуточные сообщения (tool calls, tool messages)
    # - Сохраняем счетчик попыток
    return {
        "messages": [HumanMessage(content=rewritten_question)],
        "rewrite_count": rewrite_count
    }