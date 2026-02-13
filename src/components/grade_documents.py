# src/components/grade_documents.py

from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.settings import settings

GRADE_PROMPT = (
    "Ты — эксперт, оценивающий релевантность извлеченного документа вопросу пользователя. \n "
    "Вот извлеченный документ: \n\n {context} \n\n"
    "Вот вопрос пользователя: {question} \n"
    "Если документ содержит ключевые слова или семантический смысл, связанный с вопросом пользователя, оцени его как релевантный. \n"
    "Дай бинарную оценку 'yes' (да) или 'no' (нет), чтобы указать, релевантен ли документ вопросу."
)


class GradeDocuments(BaseModel):
    """Оценка документов с использованием бинарного балла для проверки релевантности."""

    binary_score: str = Field(
        description="Оценка релевантности: 'yes', если документ релевантен, или 'no', если нет"
    )


grader_model = init_chat_model(
    model=settings.OPENAI_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.BASE_URL,
    model_provider="openai",
)


def grade_documents(state: MessagesState,) -> Literal["generate_answer", "rewrite_question"]:
    """Определить, релевантны ли извлеченные документы вопросу."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"

    return "rewrite_question"