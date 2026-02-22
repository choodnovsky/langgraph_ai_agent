# graph/state.py

from langgraph.graph import MessagesState


class GraphState(MessagesState):
    """Расширенный State графа.

    Наследуется от MessagesState:
    - messages: List[BaseMessage] — история сообщений

    Добавляем:
    - rewrite_count: int — количество попыток переформулирования вопроса
    """
    rewrite_count: int = 0
    summary: str = ""