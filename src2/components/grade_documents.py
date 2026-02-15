# src/components/grade_documents.py
# Версия с более строгой оценкой релевантности

from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
from pydantic import BaseModel, Field

GRADE_PROMPT_STRICT = """Ты — строгий эксперт по оценке релевантности документов.

ЗАДАЧА: Определить, содержит ли документ КОНКРЕТНУЮ информацию для ответа на вопрос.

ДОКУМЕНТ:
{context}

ВОПРОС:
{question}

КРИТЕРИИ ОЦЕНКИ:

РЕЛЕВАНТЕН (yes) - только если:
- Документ ПРЯМО отвечает на вопрос
- Содержит КОНКРЕТНЫЕ факты по теме вопроса
- Информация ДОСТАТОЧНА для полного ответа

НЕ РЕЛЕВАНТЕН (no) - если:
- Документ лишь косвенно связан с темой
- Содержит общие рассуждения без конкретики
- Нет прямого ответа на вопрос
- Информация слишком поверхностная
- Вопрос о компании/организации, которой нет в документе
- Вопрос о специфичных деталях, которых нет в документе

ПРИМЕРЫ:

Вопрос: "Что такое reward hacking?"
Документ: "Reward hacking is when RL agents..."
→ yes (прямой ответ)

Вопрос: "Что такое reward hacking?"
Документ: "Machine learning algorithms can..."
→ no (косвенно связано, нет конкретики)

Вопрос: "Какие бонусы в компании XYZ?"
Документ: "Companies often provide benefits..."
→ no (нет информации о компании XYZ)

Вопрос: "Расскажи про политику отпусков"
Документ: "This article discusses diffusion models..."
→ no (совершенно другая тема)

ВАЖНО: Будь строгим! Лучше ответить 'no' и переформулировать вопрос, чем дать неточный ответ.

Дай ТОЛЬКО 'yes' или 'no'."""


class GradeDocuments(BaseModel):
    """Оценка документов с использованием бинарного балла для проверки релевантности."""

    binary_score: str = Field(
        description="Оценка релевантности: 'yes', если документ релевантен, или 'no', если нет"
    )


_grader_model = None


def get_grader_model():
    """Ленивая инициализация grader модели."""
    global _grader_model

    if _grader_model is not None:
        return _grader_model

    from src2.settings import settings

    _grader_model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,  # Детерминированная оценка
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    return _grader_model


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Оценка документов и возврат решения куда идти дальше.

    Используется как функция условного ребра (conditional edge).
    Возвращает строку: "generate_answer" или "rewrite_question"
    """
    messages = state["messages"]

    # Проверка на пустой state
    if not messages:
        return "rewrite_question"

    # Берем ПОСЛЕДНИЙ вопрос пользователя
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        return "rewrite_question"

    question = human_messages[-1].content

    # Находим последнее ToolMessage с результатами поиска
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if not tool_messages:
        return "rewrite_question"

    context = tool_messages[-1].content

    # Проверка количества попыток переформулирования
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= 2:
        return "generate_answer"

    # Проверка на пустой контекст
    if not context or len(context.strip()) < 50:
        return "rewrite_question"

    # Оцениваем релевантность
    prompt = GRADE_PROMPT_STRICT.format(question=question, context=context[:1000])

    try:
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
        return "generate_answer"