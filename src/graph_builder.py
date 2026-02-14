# src/graph_builder.py

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


class GraphState(MessagesState):
    """Расширенный State для отслеживания дополнительной информации.

    Наследуется от MessagesState, который предоставляет:
    - messages: List[BaseMessage] - история сообщений

    Добавляем:
    - rewrite_count: int - количество попыток переформулирования вопроса
    """
    rewrite_count: int = 0


def build_graph():
    """Построить граф самокорректирующегося RAG с переформулированием вопросов.

    ВАЖНО: Все импорты компонентов происходят ВНУТРИ функции,
    чтобы не замедлять импорт модуля graph_builder.

    Структура графа:

    START
      ↓
    generate_query_or_respond (LLM решает: искать или ответить напрямую)
      ├─→ END (если может ответить без поиска)
      └─→ retrieve (поиск в векторной БД через ToolNode)
           ↓
         grade_documents (оценка релевантности найденных документов)
           ├─→ generate_answer (если документы релевантны)
           │    ↓
           │   END
           └─→ rewrite_question (если нерелевантны, переформулировать)
                ↓
              generate_query_or_respond (новая попытка с переформулированным вопросом)

    Особенности:
    - Максимум 2 попытки переформулирования (защита от бесконечных циклов)
    - Ленивая инициализация всех компонентов
    - Логирование на каждом шаге
    - Проверки типов сообщений
    """

    # Импортируем компоненты только при вызове build_graph()
    # Это критично для быстрого импорта модуля!
    from src.components.generate_answer import generate_answer
    from src.components.generate_query import generate_query_or_respond
    from src.components.grade_documents import grade_documents
    from src.components.rewrite_question import rewrite_question

    # retriever_tool импортируем отдельно для ToolNode
    # Сам инструмент легкий, тяжелые операции в get_retriever()
    from src.components.retriever_tool import retriever_tool

    # Создаем граф с расширенным State
    workflow = StateGraph(GraphState)

    # Определяем узлы графа
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Начинаем с узла generate_query_or_respond
    workflow.add_edge(START, "generate_query_or_respond")

    # Условное ребро: LLM решает вызвать инструмент или ответить напрямую
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # tools_condition проверяет наличие tool_calls в последнем AIMessage
        tools_condition,
        {
            # Если LLM вызвал инструмент → идем в retrieve
            "tools": "retrieve",
            # Если LLM ответил напрямую → завершаем
            END: END,
        },
    )

    # После retrieve оцениваем релевантность документов
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,  # Возвращает "generate_answer" или "rewrite_question"
    )

    # После генерации ответа завершаем
    workflow.add_edge("generate_answer", END)

    # После переформулирования возвращаемся к началу (новая попытка)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Компилируем граф
    return workflow.compile()