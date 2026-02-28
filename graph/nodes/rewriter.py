# graph/nodes/rewriter.py

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rewriter")
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

REWRITE_PROMPT = (
    "Посмотри на входные данные и попытайся проанализировать базовое семантическое намерение / значение.\n"
    "Предыдущий поисковый запрос не дал релевантных результатов. "
    "Переформулируй вопрос так, чтобы улучшить результаты поиска.\n"
    "Вот исходный вопрос:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Сформулируй улучшенный вопрос на том же языке, что и оригинал. "
    "Используй синонимы и альтернативные формулировки:"
)


def rewrite_question(state: MessagesState):
    """Переформулировать вопрос для улучшения поиска."""
    from graph.nodes.query import get_response_model

    messages = state["messages"]

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        logger.debug("[rewriter] HumanMessage не найден")
        question = "Unknown question"
    else:
        question = human_messages[-1].content

    rewrite_count = state.get("rewrite_count", 0) + 1

    logger.debug(f"[rewriter] попытка №{rewrite_count}, переформулируем: {question[:60]}")

    prompt = REWRITE_PROMPT.format(question=question)
    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])

    rewritten = response.content
    logger.debug(f"[rewriter] новый вопрос: {rewritten[:80]}")

    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewrite_count": rewrite_count,
    }