# src/components/rewrite_question.py

from langgraph.graph import MessagesState
from src.components.generate_query import response_model


REWRITE_PROMPT = """
Ты помогаешь улучшить поисковый запрос для системы поиска документов.

Переформулируй вопрос так, чтобы:
- он был максимально конкретным
- сохранился исходный смысл
- не добавлялись новые идеи
- не проводился анализ намерения
- не было пояснений

Верни только улучшенную формулировку вопроса.

Исходный вопрос:
{question}
"""


def rewrite_question(state: MessagesState):
    """Оптимизация вопроса для поиска."""

    question = None

    for msg in state["messages"]:
        if msg.type == "human":
            question = msg.content
            break

    if not question:
        raise ValueError("Вопрос не найден.")

    response = response_model.invoke(
        [
            {"role": "system", "content": "Ты переписываешь вопросы для поиска."},
            {"role": "user", "content": REWRITE_PROMPT.format(question=question)},
        ]
    )

    # Возвращаем как assistant message
    return {"messages": [response]}