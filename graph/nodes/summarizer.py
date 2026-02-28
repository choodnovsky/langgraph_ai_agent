# graph/nodes/summarizer.py

import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage

# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("summarizer")
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

MESSAGES_TO_KEEP = 4
SUMMARIZE_AFTER = 50


def should_summarize(state) -> Literal["summarizer", "__end__"]:
    """Conditional edge ДО summarizer: нужна ли суммаризация?"""
    count = len(state["messages"])
    if count > SUMMARIZE_AFTER:
        logger.debug(f"[summarizer] порог достигнут ({count} сообщений) → суммаризируем")
        return "summarizer"
    logger.debug(f"[summarizer] история короткая ({count} сообщений) → пропускаем")
    return "__end__"


def summarize_conversation(state):
    """Сворачивает историю в сводку, удаляет старые сообщения."""
    from graph.nodes.query import get_response_model

    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"Это сводка разговора на данный момент: {summary}\n\n"
            "Дополни сводку, учитывая новые сообщения выше. "
            "Сохраняй все важные факты: имена, числа, решения, контекст:"
        )
        logger.debug("[summarizer] дополняем существующую сводку")
    else:
        summary_message = (
            "Создай краткую сводку разговора выше. "
            "Включи все важные факты: имена, числа, решения, контекст:"
        )
        logger.debug("[summarizer] создаём сводку с нуля")

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    model = get_response_model()
    response = model.invoke(messages)

    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][:-MESSAGES_TO_KEEP]
    ]

    logger.debug(f"[summarizer] свёртка завершена. удалено: {len(delete_messages)}, осталось: {MESSAGES_TO_KEEP}")

    return {
        "summary": response.content,
        "messages": delete_messages,
    }