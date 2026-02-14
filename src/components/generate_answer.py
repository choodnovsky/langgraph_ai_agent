# src/components/generate_answer.py

from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

GENERATE_PROMPT = (
    "Ты — помощник по ответам на вопросы. "
    "Используй следующие фрагменты извлеченного контекста, чтобы ответить на вопрос. "
    "Если в контексте указан конкретный срок — укажи его точно. "
    "Если ты не знаешь ответа, просто скажи, что не знаешь. "
    "Используй максимум три предложения и старайся отвечать кратко.\n"
    "Вопрос: {question} \n"
    "Контекст: {context}"
)


def generate_answer(state: MessagesState):
    """Сгенерировать ответ на основе найденных документов.

    Ожидаемая структура state["messages"]:
    - Последний HumanMessage: текущий вопрос пользователя
    - Последний ToolMessage: результаты поиска
    """
    # Импортируем модель только когда нужна
    from src.components.generate_query import get_response_model
    from langchain_core.messages import HumanMessage

    messages = state["messages"]

    # ВАЖНО: Берем ПОСЛЕДНИЙ вопрос пользователя (не первый!)
    # В Streamlit может быть несколько вопросов в истории
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        print("Error: No HumanMessage found in state")
        question = "Unknown question"
    else:
        question = human_messages[-1].content  # Последний вопрос

    # Находим последнее ToolMessage с результатами поиска
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if not tool_messages:
        # Резервный вариант: если нет ToolMessage, используем последнее сообщение
        context = messages[-1].content if messages else ""
        # print("Warning: No ToolMessage found, using last message as context")
    else:
        context = tool_messages[-1].content

    # Генерируем ответ
    prompt = GENERATE_PROMPT.format(question=question, context=context)

    # Получаем модель (инициализируется только при первом вызове)
    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])

    # print(f"Generated answer based on question: '{question[:50]}...'")

    return {"messages": [response]}