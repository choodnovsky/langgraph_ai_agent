# RAG Граф с LangGraph + Streamlit

Самокорректирующаяся RAG система с переформулированием вопросов и интерфейсом на Streamlit.

## 🎯 Что это?

**Intelligent RAG** с автоматической оценкой релевантности и переформулированием запросов:

```
Вопрос пользователя
    ↓
LLM решает: искать или ответить?
    ↓
Поиск в векторной БД (ChromaDB)
    ↓
Оценка релевантности
    ├─→ Релевантно → Генерация ответа
    └─→ НЕ релевантно → Переформулирование → Повторный поиск
```

**Особенности:**
- ✅ Самокорректирующийся (до 2 попыток переформулирования)
- ✅ ChromaDB для персистентного хранения векторов
- ✅ Streamlit UI для удобного взаимодействия
- ✅ Оптимизирован для быстрого старта (< 5s импорт)
- ✅ Поддержка русского и английского языков

---

## 📦 Установка

### 1. Клонируйте репозиторий

```bash
git clone <your-repo>
cd <your-repo>
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

### 4. Настройте переменные окружения

Создайте файл `.env`:

```bash
# OpenAI API (или другой провайдер)
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
BASE_URL=https://api.openai.com/v1

# Опционально: для других провайдеров
# ANTHROPIC_API_KEY=...
# GROQ_API_KEY=...
```

### 5. Подготовьте документы

Поместите ваши документы в директорию `wiki/`:

```
wiki/
├── document1.txt
├── document2.txt
└── ...
```

Или измените URL веб-документов в `src/components/process_web_documents.py`.

---

## 🚀 Запуск

### Запуск Streamlit приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере: `http://localhost:8501`

### Первый запуск

При первом запросе система создаст ChromaDB индекс (15-20 секунд):

```
🔄 Initializing ChromaDB vectorstore...
📚 Total documents to index: 245
✅ ChromaDB created and persisted to /path/to/chroma_db
```

Последующие запуски будут мгновенными (загрузка с диска ~1-2 секунды).

---

## 🏗️ Структура проекта

```
.
├── app.py                          # Streamlit интерфейс
├── src/
│   ├── graph_builder.py           # Главный граф LangGraph
│   ├── settings.py                # Настройки (API keys, модели)
│   └── components/
│       ├── generate_query.py      # LLM роутинг (искать/ответить)
│       ├── retriever_tool.py      # ChromaDB retriever
│       ├── grade_documents.py     # Оценка релевантности
│       ├── generate_answer.py     # Генерация финального ответа
│       ├── rewrite_question.py    # Переформулирование вопроса
│       ├── process_web_documents.py   # Загрузка web документов
│       └── process_txt_documents.py   # Загрузка локальных .txt
├── wiki/                          # Ваши локальные документы
├── chroma_db/                     # ChromaDB векторная БД (auto-created)
├── requirements.txt
├── .env                           # API ключи (создайте сами)
├── .gitignore
└── README.md
```

---

## 🔧 Конфигурация

### settings.py

Основные настройки в `src/settings.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Параметры retriever

В `src/components/retriever_tool.py`:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Количество документов для retrieval
)
```

### Параметры chunking

В `process_web_documents.py` и `process_txt_documents.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Размер чанка
    chunk_overlap=150,   # Перекрытие между чанками
)
```

---

## 💾 ChromaDB

### Управление векторной БД

**Пересоздать индекс:**

```bash
# Удалить существующую БД
rm -rf chroma_db/

# Или использовать утилиту
python -c "from src.components.retriever_tool import clear_chroma_db; clear_chroma_db()"
```

**Pre-build индекса (перед деплоем):**

```bash
python build_chroma_index.py --verify
```

**Проверить размер БД:**

```python
from pathlib import Path
import os

chroma_path = Path("chroma_db")
total_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(chroma_path)
    for filename in filenames
)
print(f"ChromaDB size: {total_size / (1024*1024):.2f} MB")
```

---

## 🎨 Streamlit интерфейс

### Основные возможности

- **Chat interface** - история сообщений
- **Streaming responses** - потоковый вывод ответов
- **Debug mode** - просмотр промежуточных шагов
- **Settings sidebar** - настройка параметров
- **Clear history** - очистка истории чата

### Пример app.py (базовая версия)

```python
import streamlit as st
from langchain_core.messages import HumanMessage
from src.graph_builder import graph

st.title("🤖 RAG Assistant")

# Инициализация истории
if "messages" not in st.session_state:
    st.session_state.messages = []

# Показать историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Инпут пользователя
if prompt := st.chat_input("Задайте вопрос"):
    # Добавить сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Получить ответ от графа
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            result = graph.invoke({
                "messages": [HumanMessage(content=prompt)]
            })
            response = result["messages"][-1].content
            st.markdown(response)
    
    # Добавить ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 🧪 Тестирование

### Проверка времени импорта

```bash
python check_import_time.py
```

Должно быть **< 5 секунд** для graph_builder.

### Тестирование retriever

```python
from src.components.retriever_tool import retrieve_docs

# Тест поиска
result = retrieve_docs.invoke("What is reward hacking?")
print(result[:200])
```

### Тестирование графа

```python
from src.graph_builder import graph
from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage(content="Что такое hallucination в LLM?")]
})

print(result["messages"][-1].content)
print(f"Rewrites: {result.get('rewrite_count', 0)}")
```

---

## 🚀 Деплой

### Streamlit Cloud

1. Создайте файл `secrets.toml` в `.streamlit/`:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4o-mini"
```

2. Деплойте на Streamlit Cloud:
   - Подключите GitHub репозиторий
   - Добавьте secrets через UI
   - Streamlit автоматически установит requirements.txt

### Docker

```bash
# Build image
docker build -t rag-app .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  rag-app
```

### Heroku / Railway / Render

Все поддерживают Streamlit из коробки. Добавьте в репозиторий:

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📊 Производительность

### Метрики

| Операция | Время |
|----------|-------|
| Импорт графа | < 3s ✅ |
| Первый запуск (создание индекса) | 15-20s |
| Загрузка индекса (перезапуск) | 1-2s ✅ |
| Поиск в ChromaDB | 0.1-0.5s |
| Генерация ответа (LLM) | 2-5s |
| Полный запрос (с retrieval) | 3-8s |

### Оптимизация для production

- ✅ Pre-build ChromaDB индекса перед деплоем
- ✅ Используйте кэширование в Streamlit (`@st.cache_resource`)
- ✅ Настройте connection pooling для LLM API
- ✅ Мониторьте rate limits

---

## 🐛 Troubleshooting

### "Import timeout" ошибка

**Проблема:** Граф долго инициализируется.

**Решение:**
1. Проверьте что все модели/embeddings в `@lru_cache` функциях
2. Запустите `python check_import_time.py`
3. Убедитесь что нет глобальных инициализаций

### ChromaDB не создается

**Проблема:** Ошибка при создании vectorstore.

**Решение:**
1. Проверьте права на запись в директорию
2. Убедитесь что установлены: `chromadb`, `langchain-chroma`
3. Попробуйте удалить `chroma_db/` и пересоздать

### Медленные ответы

**Проблема:** LLM долго отвечает.

**Решение:**
1. Используйте более быстрые модели (gpt-4o-mini, claude-haiku)
2. Уменьшите `k` в retriever (меньше документов)
3. Включите streaming в Streamlit

### Out of memory

**Проблема:** Приложение использует много памяти.

**Решение:**
1. Уменьшите `chunk_size` при сплиттинге
2. Используйте более легкую embeddings модель
3. Ограничьте количество документов в индексе

---

## 📚 Документация

- **CHROMADB_MIGRATION.md** - Подробно о ChromaDB и миграции
- **РЕШЕНИЕ_IMPORT_TIMEOUT.md** - Как исправлять проблемы с импортом
- **check_import_time.py** - Диагностика времени импорта
- **build_chroma_index.py** - Pre-build векторного индекса

---

## 🤝 Contributing

1. Fork репозиторий
2. Создайте feature branch: `git checkout -b feature/amazing-feature`
3. Commit изменения: `git commit -m 'Add amazing feature'`
4. Push в branch: `git push origin feature/amazing-feature`
5. Откройте Pull Request

---

## 📝 License

MIT License - используйте как хотите!

---

## 🎯 Roadmap

- [ ] Поддержка загрузки PDF файлов через UI
- [ ] Multiple collection support в ChromaDB
- [ ] Интеграция с другими LLM провайдерами (Anthropic, Groq)
- [ ] Метрики и аналитика запросов
- [ ] A/B тестирование промптов
- [ ] Feedback loop от пользователей
- [ ] API endpoint (FastAPI) помимо Streamlit
- [ ] Мультиязычная поддержка документов

---

## 💬 Support

Нашли баг или есть вопрос? Откройте issue в GitHub!