# src/components/generate_answer.py

from langgraph.graph import MessagesState

from src.components.generate_query import response_model

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
    """Сгенерировать ответ."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}