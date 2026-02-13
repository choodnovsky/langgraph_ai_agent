# src/components/rewrite_question.py

from langchain.messages import HumanMessage
from langgraph.graph import MessagesState

from src.components.generate_query import response_model

REWRITE_PROMPT = (
    "Посмотри на входные данные и попытайся проанализировать базовое семантическое намерение / значение.\n"
    "Вот исходный вопрос:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Сформулируй улучшенный вопрос:"
)


def rewrite_question(state: MessagesState):
    """Переформулировать исходный вопрос пользователя."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}