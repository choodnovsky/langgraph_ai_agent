# graph/state.py

from langgraph.graph import MessagesState
from typing import Optional



class GraphState(MessagesState):
    """Расширенный State графа.

    Наследуется от MessagesState:
    - messages: List[BaseMessage] — история сообщений

    Добавляем:
    - rewrite_count: int — количество попыток переформулирования вопроса
    """
    rewrite_count: int = 0
    summary: Optional[str]
    human_approved: Optional[bool]