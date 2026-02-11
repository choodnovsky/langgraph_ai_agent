# src/components/grade_documents.py

from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState

from src.settings import settings


GRADE_PROMPT = """
Ты — эксперт по оценке релевантности документа.

Ответь СТРОГО одним словом:
yes
или
no

Никаких объяснений.
Никакого дополнительного текста.
Только yes или no.

Документ:
{context}

Вопрос:
{question}

Если документ содержит фактическую информацию,
связанную с вопросом — ответь yes.

Если документ содержит только заголовки,
оглавление или нерелевантную информацию — ответь no.
"""


grader_model = init_chat_model(
    model=settings.OPENAI_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.BASE_URL,
    model_provider="openai",
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Определить, релевантны ли извлеченные документы вопросу."""

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)

    response = grader_model.invoke(
        [{"role": "user", "content": prompt}]
    )

    content = response.content.strip().lower()

    # защита от markdown и мусора
    content = content.replace("```", "").strip()

    if content.startswith("yes"):
        return "generate_answer"

    return "rewrite_question"