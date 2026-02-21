# graph/nodes/answer.py

from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

GENERATE_PROMPT = (
    "Ты — помощник по ответам на вопросы на основе предоставленных документов. "
    "ВНИМАТЕЛЬНО прочитай контекст ниже и найди в нём прямой ответ на вопрос.\n\n"

    "ВАЖНО:\n"
    "- Если в контексте есть КОНКРЕТНАЯ информация (название инструмента, цифра, процесс) - используй её\n"
    "- Ищи ключевые слова из вопроса в контексте\n"
    "- НЕ говори 'в контексте нет информации', если она там есть - ищи внимательнее\n"
    "- Отвечай по существу (максимум 2-3 предложения)\n"
    "- Если ответ укладывается в 2-3 предложения — отвечай кратко\n"
    "- Если вопрос предполагает список, перечень, этапы или несколько элементов — выводи весь релевантный контекст полностью, ничего не обрезая\n\n"

    "Вопрос: {question}\n\n"
    "Контекст из документов:\n{context}\n\n"
    "Ответ:"
)


def generate_answer(state: MessagesState):
    """Сгенерировать ответ на основе найденных документов.

    Ожидаемая структура state["messages"]:
    - Последний HumanMessage: текущий вопрос пользователя
    - Последний ToolMessage: результаты поиска
    """
    # Импортируем модель только когда нужна
    from graph.nodes.query import get_response_model
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
    else:
        context = tool_messages[-1].content

    # Генерируем ответ
    prompt = GENERATE_PROMPT.format(question=question, context=context)

    # Получаем модель (инициализируется только при первом вызове)
    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])

    return {"messages": [response]}