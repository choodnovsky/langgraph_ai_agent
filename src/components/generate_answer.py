# src/components/generate_answer.py

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState
from src.components.generate_query import response_model


SYSTEM_PROMPT = """
Ты отвечаешь на вопросы сотрудников компании.

Используй ТОЛЬКО предоставленный контекст.
Если в контексте указан конкретный срок — укажи его точно.
Если ответа в контексте нет — напиши: "Ответ не найден в документах."

Не повторяй вопрос.
Начинай ответ сразу с сути.
Дай короткий конкретный ответ (1–3 предложения).
"""


def generate_answer(state: MessagesState):
    """Финальная генерация ответа."""

    # Берем последний вопрос
    question = next(
        (msg.content for msg in reversed(state["messages"]) if msg.type == "human"),
        None,
    )

    if not question:
        raise ValueError("Вопрос не найден в состоянии.")

    # Собираем только tool-контекст
    context = "\n\n".join(
        msg.content for msg in state["messages"] if msg.type == "tool"
    )

    response = response_model.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст:\n{context}",
            },
        ]
    )

    return {"messages": [AIMessage(content=response.content)]}