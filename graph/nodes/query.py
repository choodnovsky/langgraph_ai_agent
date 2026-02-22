# graph/nodes/query.py
# Версия с few-shot примерами для лучшего роутинга

from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

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
    # Импортируем инструмент только когда он нужен
    from graph.nodes.retriever import retriever_tool

    messages = [SystemMessage(content=SYSTEM_PROMPT_WITH_EXAMPLES)] + state["messages"]

    # Получаем модель (инициализируется только при первом вызове)
    response_model = get_response_model()

    response = (
        response_model
        .bind_tools([retriever_tool])
        .invoke(messages)
    )

    # DEBUG: Показываем что модель решила делать
    print(f"\n[DEBUG generate_query] Модель вызвала tool: {response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else 'НЕТ'}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"[DEBUG generate_query] Tool: {tool_call['name']}, Args: {tool_call['args']}")

    return {"messages": [response]}