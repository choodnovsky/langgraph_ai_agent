# README_GRAPH — Конспект по модулю графа

Полное описание RAG-графа с самокоррекцией, построенного на LangGraph.  
Охватывает все файлы: состояние, узлы, маршрутизацию, управление памятью.

---

## Оглавление

1. [Архитектура графа](#1-архитектура-графа)
2. [state.py — схема состояния](#2-statepy--схема-состояния)
3. [builder.py — сборка графа](#3-builderpy--сборка-графа)
4. [query.py — роутинг и генерация запроса](#4-querypy--роутинг-и-генерация-запроса)
5. [retriever.py — поиск по базе знаний](#5-retrieverpy--поиск-по-базе-знаний)
6. [grader.py — оценка релевантности](#6-graderpy--оценка-релевантности)
7. [rewriter.py — переформулировка вопроса](#7-rewriterpy--переформулировка-вопроса)
8. [answer.py — генерация ответа](#8-answerpy--генерация-ответа)
9. [summarizer.py — управление памятью](#9-summarizerpy--управление-памятью)
10. [Общий поток выполнения](#10-общий-поток-выполнения)
11. [Ключевые паттерны проекта](#11-ключевые-паттерны-проекта)

---

## 1. Архитектура графа

Граф реализует паттерн **Corrective RAG (CRAG)** — RAG с петлёй самокоррекции.  
Если найденные документы нерелевантны, граф переформулирует вопрос и повторяет поиск (до 2 раз).

### Визуальная схема

```
START
  │
  ▼
┌─────────────────────────────────────┐
│  query                              │
│  LLM: искать или ответить напрямую? │
└───────────────┬─────────────────────┘
                │
        ┌───────┴────────┐
        │                │
   (вызвал tool)    (ответил сам)
        │                │
        ▼                ▼
   retrieve          summarizer ──► END
   ToolNode              ▲
        │                │
        ▼                │
   grade_documents       │
        │                │
   ┌────┴────┐           │
   │         │           │
(релевантно) │     (нерелевантно)
   │         └──► rewriter ──► query (повтор)
   ▼
 answer
   │
   ▼
summarizer ──► END
```

### Режимы работы

| Режим | `use_checkpointer` | Кто управляет памятью |
|-------|-------------------|----------------------|
| LangGraph Studio | `False` | Studio сам через `POSTGRES_URI` |
| Streamlit | `True` | PostgreSQL через `PostgresSaver` |

---

## 2. `state.py` — схема состояния

**Путь:** `graph/state.py`

### Код

```python
from langgraph.graph import MessagesState

class GraphState(MessagesState):
    rewrite_count: int = 0
    summary: str = ""
```

### Описание

`GraphState` — единственный источник правды о данных, которые передаются между узлами.  
Наследуется от `MessagesState`, который предоставляет поле `messages` со специальным **редьюсером `add_messages`**.

### Поля

| Поле | Тип | Дефолт | Кто пишет | Кто читает |
|------|-----|--------|-----------|------------|
| `messages` | `list[BaseMessage]` | `[]` | Все узлы | Все узлы |
| `rewrite_count` | `int` | `0` | `rewriter.py` | Условные рёбра (защита от зацикливания) |
| `summary` | `str` | `""` | `summarizer.py` | `summarizer.py` |

### Как работает редьюсер `add_messages`

Когда узел возвращает `{"messages": [новое_сообщение]}`:
- обычное сообщение → **добавляется** в конец списка
- `RemoveMessage(id=...)` → **удаляет** сообщение с этим `id` из списка

Это объясняет, почему все узлы возвращают только новые сообщения, а не всю историю целиком.

### Типы сообщений в `messages`

| Тип | Кто создаёт | Содержимое |
|-----|------------|-----------|
| `HumanMessage` | Пользователь / `rewriter.py` | Вопрос пользователя или переформулировка |
| `AIMessage` | `query.py`, `answer.py` | Ответ LLM или tool_call |
| `ToolMessage` | `ToolNode` (LangGraph) | Результат поиска из ChromaDB |
| `SystemMessage` | `query.py` | Системный промпт (не сохраняется) |

---

## 3. `builder.py` — сборка графа

**Путь:** `graph/builder.py`

### Что делает

Функция-фабрика `build_graph()` создаёт, настраивает и компилирует граф.  
Принимает один параметр `use_checkpointer: bool`, который определяет режим работы.

### Регистрация узлов

```python
workflow = StateGraph(GraphState)

workflow.add_node("query",     generate_query_or_respond)
workflow.add_node("retrieve",  ToolNode([retriever_tool]))
workflow.add_node("answer",    generate_answer)
workflow.add_node("rewriter",  rewrite_question)
workflow.add_node("summarizer",summarize_conversation)
```

`ToolNode([retriever_tool])` — встроенный узел LangGraph. Он перехватывает `tool_calls` из `AIMessage`, вызывает соответствующий инструмент и упаковывает результат в `ToolMessage`.

### Рёбра и маршрутизация

```python
# Точка входа
workflow.add_edge(START, "query")

# После query: искать или ответить?
workflow.add_conditional_edges("query", tools_condition, {
    "tools": "retrieve",
    END:     "summarizer",
})

# После retrieve: документы релевантны?
workflow.add_conditional_edges("retrieve", grade_documents, {
    "answer":   "answer",
    "rewriter": "rewriter",
})

# Фиксированные переходы
workflow.add_edge("answer",   "summarizer")
workflow.add_edge("rewriter", "query")

# После summarizer: нужна ли ещё суммаризация?
workflow.add_conditional_edges("summarizer", should_summarize)
```

### Checkpointer (режим Streamlit)

```python
if use_checkpointer:
    conn = psycopg.connect(settings.POSTGRES_URI, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # создаёт таблицы в БД если их нет
    return workflow.compile(checkpointer=checkpointer)

return workflow.compile()
```

С checkpointer'ом состояние диалога сохраняется в PostgreSQL между HTTP-запросами Streamlit.  
Без него — живёт только в памяти процесса (для Studio это нормально, Studio управляет памятью сам).

### Экспорт для Studio

```python
graph = build_graph(use_checkpointer=False)
```

LangGraph Studio ищет переменную `graph` в модуле при запуске. Создаётся сразу при импорте.

---

## 4. `query.py` — роутинг и генерация запроса

**Путь:** `graph/nodes/query.py`

### Роль в графе

Первый узел после `START`. LLM решает: **лезть в базу знаний** (вызвать tool) или **ответить напрямую** из общих знаний.

### Системный промпт (логика роутинга)

**Отвечать напрямую:**
- Общеизвестные факты, определения базовых понятий
- Приветствия и общая коммуникация
- Простые вычисления

**Использовать поиск:**
- Корпоративные процессы, регламенты, политики
- Специфические инструменты и платформы
- Техническая документация

**Критически важно — формат поискового запроса:**  
LLM обязана передавать в tool только ключевые слова (5–8 слов), без вопросительных слов и глаголов.  
Пример: вместо `"Как настроить систему мониторинга?"` → `"настройка мониторинг система"`.  
Короткие запросы из ключевых слов дают лучшие результаты в векторном поиске.

### Инициализация модели

```python
@lru_cache(maxsize=1)
def get_response_model():
    model = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,           # детерминированность для роутинга
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )
    return model
```

`@lru_cache(maxsize=1)` — фактически синглтон. Модель создаётся **один раз** за всё время жизни процесса и переиспользуется всеми узлами (`rewriter`, `answer`, `summarizer` импортируют `get_response_model` из этого модуля).

`temperature=0` — важно для задачи маршрутизации: нужна предсказуемость, не творчество.

`get_secret_value()` — метод Pydantic `SecretStr`. Защищает ключ от случайного логирования: `print(settings.OPENAI_API_KEY)` выведет `"**********"`.

### Основная функция

```python
def generate_query_or_respond(state: MessagesState):
    from graph.nodes.retriever import retriever_tool   # ленивый импорт

    messages = [SystemMessage(content=SYSTEM_PROMPT_WITH_EXAMPLES)] + state["messages"]

    response = (
        get_response_model()
        .bind_tools([retriever_tool])   # сообщаем LLM о доступных инструментах
        .invoke(messages)
    )

    return {"messages": [response]}
```

`.bind_tools([retriever_tool])` — сериализует описание инструмента (имя, параметры, docstring) и добавляет в запрос к API в формате `tools`. LLM видит, что может вызвать `retrieve_docs(query: str)`.

Результат `response` — `AIMessage` с заполненным либо `content` (текст), либо `tool_calls` (запрос на поиск). Именно это проверяет `tools_condition` в следующем условном ребре.

---

## 5. `retriever.py` — поиск по базе знаний

**Путь:** `graph/nodes/retriever.py`

### Роль в графе

Не является узлом графа напрямую — является **инструментом (tool)**, который вызывает `ToolNode`.  
Выполняет векторный поиск по ChromaDB и возвращает текст найденных документов.

### Стек

```
ChromaDB (отдельный сервис, HTTP) ← chromadb.HttpClient
        ↕
HuggingFaceEmbeddings (локальная модель, та же что в indexer.py)
        ↕
LangChain Chroma (обёртка → стандартный интерфейс VectorStore)
        ↕
.as_retriever(search_kwargs={"k": 3}) → возвращает топ-3 Document
```

### Кэширование

```python
@lru_cache(maxsize=1)
def get_vectorstore():   # подключение к ChromaDB + загрузка модели эмбеддингов
    ...

@lru_cache(maxsize=1)
def get_retriever():     # vectorstore.as_retriever(k=3)
    ...
```

Оба кэшируются, потому что инициализация дорогостоящая:
- `chromadb.HttpClient` — сетевое соединение
- `HuggingFaceEmbeddings` — загрузка модели в память (секунды при первом запуске)

### Инструмент поиска

```python
@tool
def retrieve_docs(query: str) -> str:
    """Поиск и получение информации из документов.
    ...
    """
    docs = get_retriever().invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])
```

Декоратор `@tool` автоматически формирует JSON-схему инструмента из имени функции, docstring и аннотаций типов. Эту схему LangGraph отправляет в LLM вместе с запросом.

Разделитель `---` между документами помогает LLM понять границы источников.

```python
retriever_tool = retrieve_docs   # псевдоним для импорта в builder.py и query.py
```

### Полный путь запроса

```
LLM передала строку query
    ↓
retrieve_docs(query)
    ↓
HuggingFaceEmbeddings: "запрос" → вектор [0.23, -0.11, ...]
    ↓
ChromaDB: cosine similarity search → топ-3 Document
    ↓
"текст doc1\n\n---\n\nтекст doc2\n\n---\n\nтекст doc3"
    ↓
ToolMessage (создаётся LangGraph автоматически)
```

---

## 6. `grader.py` — оценка релевантности

**Путь:** `graph/nodes/grader.py`  
*(файл не был разобран отдельно, описание по роли в графе)*

### Роль в графе

Условное ребро после `retrieve`. Оценивает, насколько найденные документы отвечают на вопрос.  
Возвращает строку `"answer"` или `"rewriter"` — это определяет следующий узел.

### Место в потоке

```python
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,          # функция-условие
    {
        "answer":   "answer",    # документы релевантны → генерировать ответ
        "rewriter": "rewriter",  # документы нерелевантны → переформулировать
    }
)
```

---

## 7. `rewriter.py` — переформулировка вопроса

**Путь:** `graph/nodes/rewriter.py`

### Роль в графе

Вызывается когда `grade_documents` признал документы нерелевантными.  
Переформулирует вопрос, увеличивает счётчик попыток, возвращает управление в `query`.

### Промпт

Ключевые инструкции для LLM:
- Проанализировать **семантическое намерение** (не просто перефразировать слова)
- Учесть, что предыдущий поиск **не дал результатов** (нужно сменить угол)
- Сохранить **язык оригинала** (без этого модель может переключиться на другой язык)
- Использовать **синонимы и альтернативные формулировки**

### Логика функции

```python
def rewrite_question(state: MessagesState):
    # Берём ПОСЛЕДНИЙ HumanMessage (не первый — может быть уже переформулированный)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    question = human_messages[-1].content

    # Увеличиваем счётчик попыток
    rewrite_count = state.get("rewrite_count", 0) + 1

    # LLM переформулирует — БЕЗ .bind_tools() (только текст)
    response = get_response_model().invoke([{"role": "user", "content": prompt}])

    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": rewrite_count
    }
```

Возврат `[HumanMessage(...)]` — редьюсер добавит его в конец истории.  
Граф пойдёт в `query`, LLM увидит всю историю и новый вопрос последним — и сделает новый поиск.

### Состояние после rewriter

```
HumanMessage("исходный вопрос")
AIMessage(tool_calls=[retrieve_docs])
ToolMessage("нерелевантные документы")
HumanMessage("переформулированный вопрос")   ← добавлен rewriter
```

---

## 8. `answer.py` — генерация ответа

**Путь:** `graph/nodes/answer.py`

### Роль в графе

Вызывается когда `grade_documents` признал документы релевантными.  
Генерирует финальный ответ пользователю на основе найденных документов.

### Промпт (ключевые правила)

| Правило | Зачем |
|---------|-------|
| "ВНИМАТЕЛЬНО прочитай контекст" | LLM должна отвечать из документов, не из общих знаний |
| "НЕ говори 'нет информации', если она есть" | Защита от ленивого поведения модели |
| "Максимум 2–3 предложения" | Краткость для обычных вопросов |
| "Если список/этапы — выводи полностью" | Исключение для структурированных ответов |

### Логика функции

```python
def generate_answer(state: MessagesState):
    # Последний вопрос (учитываем возможную переформулировку)
    question = [m for m in messages if isinstance(m, HumanMessage)][-1].content

    # Последний ToolMessage (если была переформулировка — берём результат последнего поиска)
    context = [m for m in messages if isinstance(m, ToolMessage)][-1].content

    prompt = GENERATE_PROMPT.format(question=question, context=context)

    # LLM отвечает — БЕЗ .bind_tools() (только текст)
    response = get_response_model().invoke([{"role": "user", "content": prompt}])

    return {"messages": [response]}
```

Важно: модель получает не всю историю, а только чистый промпт `вопрос + документы`. Это намеренно — LLM не должна отвлекаться на техническую историю tool_calls.

Если была переформулировка и два поиска, берётся последний `ToolMessage` — он соответствует актуальному (улучшенному) запросу.

---

## 9. `summarizer.py` — управление памятью

**Путь:** `graph/nodes/summarizer.py`

### Роль в графе

Вызывается после каждого ответа (и при прямом ответе без поиска).  
Сжимает историю диалога когда она становится слишком длинной.

### Константы

```python
MESSAGES_TO_KEEP = 4    # сколько сообщений оставить после сжатия
SUMMARIZE_AFTER  = 10   # порог запуска суммаризации
```

Стратегия: история растёт → достигает 10 → сжимается до 4 → снова растёт → снова сжимается.

### Условное ребро `should_summarize`

```python
def should_summarize(state) -> Literal["summarizer", "__end__"]:
    if len(state["messages"]) > SUMMARIZE_AFTER:
        return "summarizer"
    return "__end__"
```

Обратите внимание: эта функция используется **после** узла `summarizer`. То есть `summarizer` вызывается всегда (проверяет условие внутри), а `should_summarize` решает — нужна ли ещё одна итерация.

### Логика суммаризации

```python
def summarize_conversation(state):
    summary = state.get("summary", "")

    if summary:
        # Дополняем существующую сводку
        summary_message = f"Это сводка на данный момент: {summary}\n\nДополни сводку..."
    else:
        # Создаём с нуля
        summary_message = "Создай краткую сводку разговора..."

    # Добавляем инструкцию в конец истории как HumanMessage
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = get_response_model().invoke(messages)

    # Формируем команды на удаление старых сообщений
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][:-MESSAGES_TO_KEEP]  # все кроме последних 4
    ]

    return {
        "summary": response.content,
        "messages": delete_messages,   # редьюсер удалит эти сообщения по id
    }
```

### Хитрость с промптом

Инструкция для суммаризации добавляется в историю как `HumanMessage` — это не сохраняется в состоянии, это одноразовый контекст только для данного вызова LLM. Модель видит всю историю и задание в конце.

### Повторная суммаризация

Если сводка уже была (`summary != ""`), LLM получает её как контекст и **дополняет**, а не перезаписывает с нуля. Старые факты не теряются, новые добавляются.

### Результат после суммаризации

```
state["summary"]  = "краткая сводка всего важного..."
state["messages"] = [последние 4 сообщения]
```

---

## 10. Общий поток выполнения

### Сценарий 1: Прямой ответ (без поиска)

```
START → query (LLM решает: ответить напрямую)
      → summarizer (should_summarize: история < 10)
      → END
```

### Сценарий 2: Поиск с первой попытки

```
START → query (LLM решает: вызвать retrieve_docs)
      → retrieve (ToolNode выполняет поиск)
      → grade_documents → "answer"
      → answer (LLM генерирует ответ по документам)
      → summarizer (should_summarize: история < 10)
      → END
```

### Сценарий 3: Поиск с переформулировкой

```
START → query → retrieve → grade_documents → "rewriter"
      → rewriter (LLM переформулирует, rewrite_count = 1)
      → query → retrieve → grade_documents → "answer"
      → answer → summarizer → END
```

### Сценарий 4: Длинный диалог (с суммаризацией)

```
... (после 10 сообщений) ...
→ summarizer (should_summarize: история > 10)
→ summarizer (суммаризация выполняется: история сжата до 4 + сводка)
→ END
```

### Состояние messages на каждом этапе

```
После query (tool_call):
  [HumanMessage("вопрос"), AIMessage(tool_calls=[...])]

После retrieve:
  [..., ToolMessage("текст документов")]

После answer:
  [..., AIMessage("финальный ответ")]

После summarizer (если история > 10):
  [последние 4 сообщения]  +  summary в отдельном поле
```

---

## 11. Ключевые паттерны проекта

### Ленивые импорты

Все тяжёлые зависимости импортируются **внутри функций**, а не на уровне модуля:

```python
def some_node(state):
    from graph.nodes.query import get_response_model   # импорт здесь
    from config.settings import settings               # а не вверху файла
```

Это решает две проблемы: избегает циклических импортов между модулями и ускоряет запуск (зависимости загружаются только когда реально нужны).

### Одна модель на весь граф

`get_response_model()` определена в `query.py` и импортируется в `rewriter.py`, `answer.py`, `summarizer.py`. Все четыре узла используют **один кэшированный экземпляр LLM**.

```
query.py    → get_response_model() [определение + @lru_cache]
rewriter.py → from graph.nodes.query import get_response_model
answer.py   → from graph.nodes.query import get_response_model
summarizer.py → from graph.nodes.query import get_response_model
```

### `bind_tools` только там где нужно

| Узел | `bind_tools` | Причина |
|------|-------------|---------|
| `query.py` | ✅ Да | LLM должна иметь возможность вызвать поиск |
| `rewriter.py` | ❌ Нет | Только переформулировать текст |
| `answer.py` | ❌ Нет | Только сгенерировать текстовый ответ |
| `summarizer.py` | ❌ Нет | Только создать текстовую сводку |

### Всегда берём ПОСЛЕДНЕЕ сообщение нужного типа

```python
# Правильно во всех узлах:
question = [m for m in messages if isinstance(m, HumanMessage)][-1].content
context  = [m for m in messages if isinstance(m, ToolMessage)][-1].content
```

Это учитывает возможную переформулировку: в истории может быть несколько `HumanMessage` и несколько `ToolMessage`, нужен всегда последний актуальный.

### Возврат только изменений

Все узлы возвращают только **дельту** состояния, а не всё состояние целиком:

```python
return {"messages": [response]}           # добавить сообщение
return {"messages": [HumanMessage(...)], "rewrite_count": n}  # добавить + обновить счётчик
return {"summary": "...", "messages": delete_messages}        # обновить сводку + удалить старые
```

LangGraph сам применяет эти изменения к текущему состоянию через редьюсеры.

---

*Далее: отдельные блоки по конфигурации и Streamlit-интерфейсу.*
