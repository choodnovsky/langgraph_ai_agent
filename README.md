# Корпоративный ИИ-агент

> RAG-система с долгосрочной памятью, самокоррекцией и корпоративной базой знаний

---

## Что это такое

Интеллектуальный ассистент, который знает всё о внутренних регламентах и процессах компании. Задаёшь вопрос — он находит ответ в документах, показывает источник и помнит контекст разговора между сессиями.

**Не просто чат-бот.** Агент сам решает — искать ли информацию в базе знаний или ответить напрямую. Если найденный документ нерелевантен — переформулирует запрос и ищет снова. Если история переписки растёт — сворачивает её в сводку, экономя токены.

---

## Как это выглядит

```
Пользователь: Какие бонусы применяются в компании?

Агент:  В компании применяются четыре вида бонусов:
        1. Квартальный — 10–30% от оклада за выполнение KPI
        2. Годовой — 20–100% от годового оклада
        3. Проектный — за успешную реализацию проектов
        4. Разовый — за особые заслуги и инновации
```

---

## Архитектура

### Граф обработки запроса

![RAG Agent Graph](img/schema_graph.png)

### Стек технологий

| Слой | Технология |
|------|-----------|
| Граф агента | LangGraph |
| LLM | OpenRouter `z-ai/glm-4.5-air:free` |
| Векторная БД | ChromaDB |
| Эмбеддинги | HuggingFace `intfloat/multilingual-e5-base` |
| Долгосрочная память | PostgreSQL (LangGraph Checkpointer) |
| Фидбек | PostgreSQL (таблица `feedback`) |
| UI | Streamlit |
| Авторизация | streamlit-authenticator |

---

## Ключевые решения

### 1. Самокоррекция запроса

Агент не просто ищет — он оценивает что нашёл. Если документ нерелевантен, переформулирует вопрос и пробует снова (до 2 раз).

```python
# graph/nodes/grader.py
def grade_documents(state) -> Literal["answer", "rewriter"]:
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= 2:
        return "answer"  # лимит исчерпан — отвечаем как есть

    response = grader_model.invoke(prompt)

    if "yes" in response.content.lower():
        return "answer"
    else:
        return "rewriter"  # переформулируем и ищем снова
```

### 2. Долгосрочная память

Каждый пользователь имеет свой `thread_id`. История сохраняется в PostgreSQL и восстанавливается между сессиями — агент помнит что ты говорил неделю назад.

```python
# graph/builder.py
conn = psycopg.connect(POSTGRES_URI, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

graph = workflow.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"configurable": {"thread_id": "victor"}}
)
```

### 3. Суммаризация истории

Когда сообщений становится больше 10 — агент сворачивает историю в краткую сводку. Модель видит сводку + последние 4 сообщения вместо всей истории. Токены не растут бесконечно.

```python
# graph/nodes/summarizer.py
def summarize_conversation(state):
    summary = state.get("summary", "")

    if summary:
        prompt = f"Текущая сводка: {summary}\n\nДополни с учётом новых сообщений:"
    else:
        prompt = "Создай краткую сводку разговора:"

    response = model.invoke(state["messages"] + [HumanMessage(content=prompt)])
    delete = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]

    return {"summary": response.content, "messages": delete}
```

### 4. Инкрементальная индексация

Индексатор отслеживает изменения файлов через MD5-хэши. При запуске обновляет только изменившиеся документы — не переиндексирует всё с нуля.

```python
# services/indexer.py
for filepath_str, filepath in current_files.items():
    current_hash = md5_file(filepath)
    saved_hash   = state.get(filepath_str)

    if current_hash != saved_hash:
        delete_file_chunks(collection, filepath_str)
        upsert_file(collection, embeddings_model, filepath)
        state[filepath_str] = current_hash
```

### 5. Система фидбека

Под каждым ответом агента пользователь может поставить 👍 или 👎. Оценка сохраняется в PostgreSQL вместе с оригинальными текстами вопроса и ответа — для последующей аналитики и дообучения.

```python
# modules/feedback.py
def save_feedback(thread_id, message_id, rating, question, answer):
    # rating: 1 = лайк, -1 = дизлайк
    # question/answer — оригинальные тексты без лемматизации
    conn.execute(
        "INSERT INTO feedback (thread_id, message_id, rating, question, answer) VALUES (%s, %s, %s, %s, %s)",
        (thread_id, message_id, rating, question, answer),
    )
```

### 6. Аналитика качества

На основе накопленных фидбеков работает модуль кластеризации. Все вопросы векторизуются той же моделью что используется в индексаторе, кластеризуются через KMeans, а рейтинг используется как сигнал качества внутри каждого кластера. LLM автоматически даёт название каждой теме.

```python
# analytics/cluster_questions.py
clusters = get_question_clusters(min_questions=10, max_k=8)

# Каждый кластер имеет health-индикатор:
# ✅ Отлично   — дизлайков нет
# 🟠 Аномалия  — 90%+ лайков, но дизлайк вдруг появился
# 🟡 Зона риска — 60–90% лайков
# 🔴 Проблема  — ниже 60% лайков
```

---

## Структура проекта

```
project/
├── app.py                      # точка входа: авторизация + меню
├── langgraph.json              # конфиг LangGraph Studio
├── streamlit_credentials.yaml  # логины пользователей
│
├── graph/
│   ├── builder.py              # сборка графа
│   ├── state.py                # GraphState (messages, rewrite_count, summary)
│   └── nodes/
│       ├── query.py            # роутинг: искать или ответить напрямую
│       ├── retriever.py        # поиск в ChromaDB
│       ├── grader.py           # оценка релевантности
│       ├── answer.py           # генерация ответа
│       ├── rewriter.py         # переформулирование вопроса
│       └── summarizer.py       # суммаризация истории
│
├── analytics/
│   └── cluster_questions.py    # кластеризация вопросов с рейтингом как сигналом
│
├── models/
│   └── schemas.py              # доменные модели: Question, ClusterStats
│
├── modules/
│   └── feedback.py             # сохранение и рендер фидбека
│
├── pages/
│   └── chat.py                 # UI страницы чата
│
├── config/
│   └── settings.py             # все настройки из .env
│
└── services/
    ├── indexer.py              # индексация wiki/ → ChromaDB
    ├── clear_collection.py     # сброс коллекции
    └── index_state.json        # MD5-хэши файлов (авто)
```

---

## Запуск

### Зависимости
```bash
pip install -r requirements.txt
```

### Инфраструктура
```bash
docker compose up -d
```

### Индексация документов
```bash
python services/indexer.py

# Cron — автообновление каждые 10 минут
*/10 * * * * python /path/to/services/indexer.py
```

### LangGraph Studio
```bash
langgraph dev
```

### Streamlit
```bash
streamlit run app.py
```

---

## Переменные окружения

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=z-ai/glm-4.5-air:free
BASE_URL=https://openrouter.ai/api/v1

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000
COLLECTION_NAME=wiki_docs

# PostgreSQL
POSTGRES_URI=postgresql://langgraph:langgraph@localhost:5432/langgraph

# Индексатор
FOLDER_PATH=wiki
INDEX_STATE_FILE=services/index_state.json
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

---

## Что планируется

- **Telegram авторизация** — вход через Telegram вместо логин/пароль
- **Мультимодальность** — поддержка PDF и изображений в базе знаний
- **Механизм смены модели** — возможность переключения моделей в зависимости от задач
- **Очистка чата и создание нового чата** — в пределах пользователя
- **Страница аналитики** — дашборд в Streamlit с кластерами вопросов, графиком качества по времени, топом лучших ответов и детектором галлюцинаций