# graph/nodes/query.py
# Версия с few-shot примерами для лучшего роутинга

import logging
from functools import lru_cache
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

# ---------------------------------------------------------------------------
# Тот же лог-файл что и в grader.py — logs/debug.log в корне проекта
# ---------------------------------------------------------------------------
_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("generate_query")
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

SYSTEM_PROMPT_WITH_EXAMPLES = """Ты — умный ассистент с доступом к базе знаний через инструмент поиска документов.

КРИТЕРИИ ПРИНЯТИЯ РЕШЕНИЯ:

ОТВЕЧАЙ НАПРЯМУЮ (БЕЗ ПОИСКА) если:
- Общеизвестные факты: география, история, математика
- Определения базовых понятий: "Что такое Python?", "Что такое ИИ?"
- Приветствия и базовая коммуникация: "Привет", "Спасибо"
- Простые вычисления и логика

ИСПОЛЬЗУЙ ПОИСК если:
- Вопросы о процессах, регламентах, политиках компании
- Вопросы про конкретные инструменты, платформы, системы
- Технические детали из документации
- Любая информация, которой может не быть в общих знаниях

КРИТИЧЕСКИ ВАЖНО - КАК ФОРМУЛИРОВАТЬ ПОИСКОВЫЙ ЗАПРОС:

Когда ты вызываешь инструмент поиска retrieve_docs, в параметре query ты должен передать:
- ТОЛЬКО ключевые слова (5-8 максимум)
- КОНКРЕТНЫЕ термины из вопроса
- БЕЗ вопросительных слов (как, какой, что, где)
- БЕЗ глаголов и предлогов

ПОМНИ: Короткие запросы из ключевых слов находят релевантные документы лучше, чем длинные вопросы!

Отвечай кратко, четко и по существу."""


@lru_cache(maxsize=1)
def get_response_model():
    """Ленивая инициализация LLM модели.

    Модель создается только при первом вызове и кэшируется.
    """
    from config.settings import settings

    model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    return model


def generate_query_or_respond(state: MessagesState):
    """Вызвать модель для генерации ответа на основе текущего состояния.

    В зависимости от вопроса, модель примет решение:
    извлечь информацию с помощью инструмента поиска или просто ответить пользователю.
    """
    from graph.nodes.retriever import retriever_tool

    messages = [SystemMessage(content=SYSTEM_PROMPT_WITH_EXAMPLES)] + state["messages"]

    response_model = get_response_model()

    response = (
        response_model
        .bind_tools([retriever_tool])
        .invoke(messages)
    )

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") and response.tool_calls else None

    if tool_calls:
        logger.debug(f"[generate_query] Модель вызвала tool: {tool_calls}")
        for tc in tool_calls:
            logger.debug(f"[generate_query] Tool: {tc['name']}, Args: {tc['args']}")
    else:
        logger.debug("[generate_query] Tool не вызван — модель отвечает напрямую")

    return {"messages": [response]}