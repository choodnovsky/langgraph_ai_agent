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

    # LLM оценивает: документ содержит ответ?
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

# При каждом вызове — история восстанавливается по thread_id
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

    # Дополняем существующую сводку или создаём новую
    if summary:
        prompt = f"Текущая сводка: {summary}\n\nДополни с учётом новых сообщений:"
    else:
        prompt = "Создай краткую сводку разговора:"

    response = model.invoke(state["messages"] + [HumanMessage(content=prompt)])

    # Удаляем старые сообщения, оставляем 4 последних
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
        delete_file_chunks(collection, filepath_str)  # удаляем старые чанки
        upsert_file(collection, embeddings_model, filepath)  # загружаем новые
        state[filepath_str] = current_hash
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
│   └── cluster_questions.py    # кластеризатор вопросов и ответов
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
# ChromaDB
docker compose up -d
```

### Индексация документов
```bash
# Положи .txt файлы в wiki/ и запусти
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
- **Механизм смены модели** - возможность переключения моделей в зависимости от задач
- **Очистка чата и создание нового чата** - в пределах пользователя
- **Аналитика** — Имея три поля question, answer, rating, можно развернуть довольно богатую аналитику:  
    * Качество по времени. Процент дизлайков по дням/неделям — анализировать деградацию после обновления модели, индекса или промпта. 
    * Кластеризация плохих вопросов. Векторизуем все дизлайкнутые вопросы через ChromaDB, кластеризуем и получаешь темы где модель стабильно проваливается. Например "вопросы про даты" или "вопросы с отрицанием". 
    * Сравнение ответов на похожие вопросы. Находим семантически близкие вопросы с разным рейтингом — анализируем чем отличаются ответы. Это даёт понимание что именно в формулировке ответа работает. 
    * Длина vs качество. Коррелируем длину ответа с рейтингом — часто есть паттерн: слишком короткие или слишком длинные ответы получают дизлайки. 
    * Выявление галлюцинаций. Дизлайкнутые ответы можно прогнать через LLM с промптом "найди фактические ошибки" — получаешь автоматический детектор галлюцинаций на реальных данных. 
    * Топ лучших ответов. Залайканные пары вопрос→ответ — это готовый датасет few-shot примеров, можно добавлять их в системный промпт динамически. 
    * Всё это будет размещено в том же приложении на странице Analytics — графики через st.line_chart, таблицы провальных тем, топ хороших ответов.