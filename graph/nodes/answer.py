# graph/nodes/answer.py

import logging
from pathlib import Path

from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("answer")
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

GENERATE_PROMPT = (
    "Ты — помощник по ответам на вопросы на основе предоставленных документов. "
    "ВНИМАТЕЛЬНО прочитай контекст ниже и найди в нём прямой ответ на вопрос.\n\n"

    "ВАЖНО:\n"
    "- Если в контексте есть КОНКРЕТНАЯ информация (название инструмента, цифра, процесс) - используй её\n"
    "- Ищи ключевые слова из вопроса в контексте\n"
    "- НЕ говори 'в контексте нет информации', если она там есть - ищи внимательнее\n"
    "- Отвечай по существу (максимум 2-3 предложения)\n"
    "- Если ответ укладывается в 2-3 предложения — отвечай кратко\n"
    "- Если вопрос предполагает список, перечень, этапы или несколько элементов — выводи весь релевантный контекст полностью, ничего не обрезая\n\n"

    "Вопрос: {question}\n\n"
    "Контекст из документов:\n{context}\n\n"
    "Ответ:"
)


def generate_answer(state: MessagesState):
    """Сгенерировать ответ на основе найденных документов."""
    from graph.nodes.query import get_response_model
    from langchain_core.messages import HumanMessage

    messages = state["messages"]

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        logger.debug("[answer] HumanMessage не найден — используем заглушку")
        question = "Unknown question"
    else:
        question = human_messages[-1].content

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        logger.debug("[answer] ToolMessage не найден — контекст пустой")
        context = messages[-1].content if messages else ""
    else:
        context = tool_messages[-1].content

    logger.debug(f"[answer] генерируем ответ на: {question[:60]}")

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])

    logger.debug(f"[answer] ответ сгенерирован ({len(response.content)} симв.)")

    return {"messages": [response]}