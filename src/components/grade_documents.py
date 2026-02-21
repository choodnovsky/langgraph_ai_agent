# src/components/grade_documents.py

from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
from pydantic import BaseModel, Field

GRADE_PROMPT_STRICT = """Оцени релевантность документа для ответа на вопрос.

ВОПРОС: {question}

ДОКУМЕНТ: {context}

Документ содержит прямой ответ на вопрос?
Ответь ТОЛЬКО одним словом: yes или no"""


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="'yes' если документ релевантен, 'no' если нет"
    )


_grader_model = None


def get_grader_model():
    global _grader_model

    if _grader_model is not None:
        return _grader_model

    from src.settings import settings

    _grader_model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    return _grader_model


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Оценка документов - используется как conditional edge."""
    messages = state["messages"]

    if not messages:
        return "rewrite_question"

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        return "rewrite_question"

    question = human_messages[-1].content

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "rewrite_question"

    context = tool_messages[-1].content

    # Берём из GraphState — сбрасывается в 0 при каждом новом вопросе
    # Не зависит от длины истории в Postgres
    rewrite_count = state.get("rewrite_count", 0)
    print(f"[DEBUG grade] попытка={rewrite_count}, вопрос: {question[:60]}")

    if rewrite_count >= 2:
        print(f"[DEBUG grade] лимит попыток → generate_answer")
        return "generate_answer"

    if not context or len(context.strip()) < 50:
        print(f"[DEBUG grade] пустой контекст → rewrite_question")
        return "rewrite_question"

    prompt = GRADE_PROMPT_STRICT.format(question=question, context=context[:1500])

    try:
        grader_model = get_grader_model()

        # ✅ Простой вызов без with_structured_output - работает с любой моделью
        response = grader_model.invoke([{"role": "user", "content": prompt}])
        answer = response.content.strip().lower()
        print(f"[DEBUG grade] ответ модели: '{answer}'")

        if "yes" in answer:
            print(f"[DEBUG grade] релевантен → generate_answer")
            return "generate_answer"
        else:
            print(f"[DEBUG grade] нерелевантен → rewrite_question")
            return "rewrite_question"

    except Exception as e:
        print(f"[DEBUG grade] ошибка: {e} → generate_answer")
        return "generate_answer"