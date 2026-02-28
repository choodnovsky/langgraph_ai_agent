# graph/nodes/grader.py

import logging
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Логгер — пишет одновременно в консоль и в logs/debug.log в корне проекта
# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("grader")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    _fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)

    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

# ---------------------------------------------------------------------------

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

    from config.settings import settings

    _grader_model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    return _grader_model


def grade_documents(state) -> Literal["answer", "rewriter"]:
    """Оценка документов - используется как conditional edge."""
    messages = state["messages"]

    if not messages:
        return "rewriter"

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        return "rewriter"

    question = human_messages[-1].content

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "rewriter"

    context = tool_messages[-1].content

    # Берём из GraphState — сбрасывается в 0 при каждом новом вопросе
    # Не зависит от длины истории в Postgres
    rewrite_count = state.get("rewrite_count", 0)
    logger.debug(f"[grade] попытка={rewrite_count}, вопрос: {question[:60]}")

    if rewrite_count >= 2:
        logger.debug("[grade] лимит попыток → generate_answer")
        return "answer"

    if not context or len(context.strip()) < 50:
        logger.debug("[grade] пустой контекст → rewrite_question")
        return "rewriter"

    prompt = GRADE_PROMPT_STRICT.format(question=question, context=context[:1500])

    try:
        grader_model = get_grader_model()

        # Простой вызов без with_structured_output - работает с любой моделью
        response = grader_model.invoke([{"role": "user", "content": prompt}])
        answer = response.content.strip().lower()
        logger.debug(f"[grade] ответ модели: '{answer}'")

        if "yes" in answer:
            logger.debug("[grade] релевантен → generate_answer")
            return "answer"
        else:
            logger.debug("[grade] нерелевантен → rewrite_question")
            return "rewriter"

    except Exception as e:
        logger.debug(f"[grade] ошибка: {e} → generate_answer")
        return "answer"