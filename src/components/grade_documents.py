# src/components/grade_documents.py

from typing import Literal
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

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


@lru_cache(maxsize=1)
def get_grader_model():
    """Ленивая инициализация grader модели.

    Модель создается только при первом вызове и кэшируется.
    """
    from src.settings import settings

    model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    return model


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Определить, релевантны ли извлеченные документы вопросу.

    Возвращает:
    - "generate_answer": если документы релевантны
    - "rewrite_question": если документы нерелевантны и нужно переформулировать вопрос
    """
    messages = state["messages"]

    # Проверка на пустой state
    if not messages:
        print("Error: Empty messages in state")
        return "rewrite_question"

    # ВАЖНО: Берем ПОСЛЕДНИЙ вопрос пользователя (не первый!)
    from langchain_core.messages import HumanMessage
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        print("Error: No HumanMessage found in state")
        return "rewrite_question"

    question = human_messages[-1].content  # Последний вопрос

    # Находим последнее ToolMessage с результатами поиска
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if not tool_messages:
        print("Error: No ToolMessage found in state")
        return "rewrite_question"

    context = tool_messages[-1].content

    # Проверка количества попыток переформулирования
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= 2:
        print(f"Warning: Max rewrite attempts ({rewrite_count}) reached, generating answer anyway")
        return "generate_answer"

    # Оцениваем релевантность
    prompt = GRADE_PROMPT.format(question=question, context=context)

    try:
        # Получаем модель (инициализируется только при первом вызове)
        grader_model = get_grader_model()

        response = (
            grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score.lower()

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    except Exception as e:
        print(f"Error during document grading: {e}")
        # В случае ошибки, пытаемся сгенерировать ответ
        return "generate_answer"