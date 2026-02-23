# README_CHROMA — Конспект по индексации и управлению ChromaDB

Полное описание скриптов для наполнения и обслуживания векторной базы знаний.  
Охватывает: `indexer.py` (инкрементальная индексация), `clear_collection.py` (сброс базы).

---

## Оглавление

1. [Общая концепция](#1-общая-концепция)
2. [Конфигурация](#2-конфигурация)
3. [indexer.py — инкрементальная индексация](#3-indexerpy--инкрементальная-индексация)
4. [clear_collection.py — сброс базы](#4-clear_collectionpy--сброс-базы)
5. [Взаимосвязь компонентов](#5-взаимосвязь-компонентов)
6. [Типичные сценарии использования](#6-типичные-сценарии-использования)

---

## 1. Общая концепция

Система состоит из двух независимых частей: **индексации** (наполнение базы) и **поиска** (использование базы в графе RAG). Эти скрипты отвечают только за индексацию.

### Что такое ChromaDB и зачем она нужна

ChromaDB — векторная база данных. В отличие от обычных БД, которые хранят текст и ищут по точному совпадению, ChromaDB хранит **векторы** (массивы чисел) и ищет по **смысловому сходству**.

Процесс: текст → модель эмбеддингов → вектор из N чисел → сохранить в ChromaDB.  
При поиске: запрос → тот же вектор → найти ближайшие векторы → вернуть их тексты.

Чтобы поиск работал корректно, **модель эмбеддингов при индексации и при поиске должна быть одной и той же**. Это связывает `indexer.py` и `retriever.py` из модуля графа.

### Инкрементальность

Индексер не переиндексирует всё при каждом запуске. Он отслеживает изменения через MD5-хэши файлов и обрабатывает только то, что реально изменилось. Это позволяет запускать его по cron часто (например, каждые 10 минут) без заметной нагрузки.

### INDEX_STATE_FILE

JSON-файл на диске — "память" индексера. Хранит `{путь_к_файлу: md5_хэш}` для всех проиндексированных файлов. Является зеркалом состояния ChromaDB в терминах файлов. Пока ChromaDB и `INDEX_STATE_FILE` синхронны — индексер работает инкрементально. Если они рассинхронизировались (например, контейнер с ChromaDB пересоздали) — нужен `clear_collection.py`.

---

## 2. Конфигурация

Все параметры читаются из `config.settings`:

| Параметр | Тип | Назначение |
|----------|-----|-----------|
| `FOLDER_PATH` | `Path` | Директория с `.txt` файлами для индексации |
| `CHROMA_HOST` | `str` | Хост ChromaDB-сервера |
| `CHROMA_PORT` | `int` | Порт ChromaDB-сервера |
| `COLLECTION_NAME` | `str` | Имя коллекции внутри ChromaDB (аналог таблицы) |
| `EMBEDDINGS_MODEL` | `str` | Название модели HuggingFace для эмбеддингов |
| `INDEX_STATE_FILE` | `Path` | Путь к JSON-файлу с хэшами |
| `CHUNK_SIZE` | `int` | Максимальный размер одного чанка в символах |
| `CHUNK_OVERLAP` | `int` | Перекрытие соседних чанков в символах |

### Параметры нарезки

`CHUNK_SIZE` и `CHUNK_OVERLAP` — ключевые параметры, влияющие на качество поиска.

`CHUNK_SIZE` определяет сколько текста попадёт в один фрагмент. Слишком маленький (100–200 символов) — контекст теряется, модель не получит достаточно информации. Слишком большой (2000+ символов) — вектор размывается, поиск становится менее точным. Оптимум обычно 400–800 символов.

`CHUNK_OVERLAP` — перекрытие соседних чанков. Если нужная информация оказалась ровно на границе разбивки, перекрытие гарантирует что она попадёт хотя бы в один чанк целиком. Обычно 10–15% от `CHUNK_SIZE`.

---

## 3. `indexer.py` — инкрементальная индексация

**Путь:** `indexer.py`  
**Запуск:** вручную или по cron

### Вспомогательные функции

#### `log(msg)`

```python
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
```

Простой логгер с временной меткой. `flush=True` критично для cron — без него вывод буферизуется и в лог-файл попадает не сразу, а только после завершения процесса.

#### `md5_file(path)`

```python
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

Считает MD5-хэш файла — уникальный "отпечаток" содержимого. Если файл изменится хоть на один символ, хэш изменится полностью.

Файл читается кусками по 64 КБ (`65536` байт) — защита от загрузки больших файлов целиком в память. `iter(lambda: f.read(65536), b"")` — Python-паттерн для итерации до конца файла: вызывает `f.read()` снова и снова до получения пустых байт.

Возвращает 32-символьную строку вида `"d41d8cd98f00b204e9800998ecf8427e"`.

#### `load_state()` и `save_state(state)`

```python
def load_state() -> dict:
    if INDEX_STATE_FILE.exists():
        with open(INDEX_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    INDEX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
```

Загрузка и сохранение словаря `{путь_к_файлу: md5}` на диск. При первом запуске файла нет — `load_state()` возвращает пустой словарь, что эквивалентно "всё новое, всё нужно индексировать".

`parent.mkdir(parents=True, exist_ok=True)` — создаёт все промежуточные директории без ошибки если они уже существуют.

#### `scan_txt_files()`

```python
def scan_txt_files() -> dict[str, Path]:
    if not FOLDER_PATH.exists():
        log(f"ERROR: FOLDER_PATH не существует: {FOLDER_PATH}")
        sys.exit(1)
    return {str(p): p for p in FOLDER_PATH.rglob("*.txt")}
```

Рекурсивно находит все `.txt` в `FOLDER_PATH` и вложенных папках. `rglob("*.txt")` аналогичен `find . -name "*.txt"`.

Возвращает словарь с двумя представлениями одного пути: строка нужна как ключ в JSON, объект `Path` нужен для операций с файлом.

#### `doc_ids_for_file(filepath, n_chunks)`

```python
def doc_ids_for_file(filepath: str, n_chunks: int) -> list[str]:
    base = hashlib.md5(filepath.encode()).hexdigest()
    return [f"{base}_chunk_{i}" for i in range(n_chunks)]
```

Генерирует стабильные IDs чанков для ChromaDB. Для файла с тремя чанками:
```
["a1b2c3d4_chunk_0", "a1b2c3d4_chunk_1", "a1b2c3d4_chunk_2"]
```

ID строится из MD5 **пути к файлу** (не содержимого) — при повторной индексации того же файла ID будут теми же самыми. Это позволяет использовать `upsert` (обновить существующее) вместо delete + insert.

---

### Функции работы с ChromaDB

#### `get_chroma_collection()`

```python
def get_chroma_collection():
    import chromadb
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection
```

`get_or_create_collection` — атомарная и идемпотентная операция: если коллекция есть — вернёт её, нет — создаст. Безопасно вызывать при каждом запуске.

`"hnsw:space": "cosine"` — метрика расстояния для поиска ближайших векторов. Косинусное сходство измеряет угол между векторами, не их длину — стандарт для семантического поиска. Должна совпадать с метрикой при создании в `clear_collection.py`.

#### `get_embeddings_model()`

```python
def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    log(f"Загружаем модель эмбеддингов: {EMBEDDINGS_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
```

Загружает локальную модель с HuggingFace. При первом запуске скачивает (~500 МБ для типичной sentence-transformer), при последующих — из локального кэша.

Эта функция вызывается **только если есть что индексировать** — ленивая инициализация, которая экономит несколько секунд при запуске без изменений (актуально для cron).

#### `split_file(filepath)`

```python
def split_file(filepath: Path) -> list:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
```

`TextLoader` — читает `.txt` и оборачивает в объект `Document(page_content=текст, metadata={"source": путь})`.

`RecursiveCharacterTextSplitter` — умный сплиттер с иерархией разделителей. Сначала пытается разбить по `"\n\n"` (абзацы). Если чанк всё ещё больше `CHUNK_SIZE` — по `"\n"` (строки), потом по пробелам, в крайнем случае посимвольно. Это сохраняет смысловые границы: абзац не будет разрезан пополам если это не абсолютно необходимо.

#### `delete_file_chunks(collection, filepath)`

```python
def delete_file_chunks(collection, filepath: str):
    results = collection.get(where={"source": filepath})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
```

Находит все чанки файла по метаданному полю `source` и удаляет их. Нужно перед повторной индексацией изменённого файла — количество и содержание чанков могло измениться, простой `upsert` не удалит лишние старые чанки.

#### `upsert_file(collection, embeddings_model, filepath)`

```python
def upsert_file(collection, embeddings_model, filepath: Path):
    chunks = split_file(filepath)
    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)   # пакетная векторизация
    ids = doc_ids_for_file(str(filepath), len(chunks))
    metadatas = [
        {
            "source": str(filepath),
            "filename": filepath.name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(chunks)
```

Полный цикл индексации одного файла:

`embed_documents(texts)` — **пакетная** векторизация всех чанков за один вызов. Значительно быстрее чем векторизовать по одному: модель обрабатывает батч за один проход через нейросеть.

`metadatas` — для каждого чанка сохраняется: полный путь файла (`source`, нужен для поиска и удаления), имя файла (`filename`, для читаемых ответов), порядковый номер (`chunk_index`, для отладки и восстановления порядка).

`collection.upsert()` — вставить если нет, обновить если есть (по `id`). Поскольку ID строятся из пути, при повторной индексации того же файла данные обновятся, а не задублируются.

---

### Основная функция `run()`

Пошаговая логика:

**Шаг 1 — Подключение и диагностика**
```python
_, collection = get_chroma_collection()
log(f"Документов в коллекции до старта: {collection.count()}")
```
Сразу логируем начальное состояние — удобно для контроля.

**Шаг 2 — Загрузка состояния и сканирование диска**
```python
state = load_state()           # хэши из прошлого запуска
current_files = scan_txt_files()  # что есть на диске сейчас
```

**Шаг 3 — Сравнение: что изменилось?**
```python
files_to_update = []
for filepath_str, filepath in current_files.items():
    current_hash = md5_file(filepath)
    saved_hash = state.get(filepath_str)
    if current_hash != saved_hash:      # новый или изменённый файл
        files_to_update.append((filepath_str, filepath, current_hash))

deleted_files = [fp for fp in state if fp not in current_files]
```

Логика определения статуса каждого файла:

| Условие | Статус | Действие |
|---------|--------|---------|
| Файл есть, хэши совпадают | Не изменился | Пропустить |
| Файл есть, хэши не совпадают | Изменился | Переиндексировать |
| Файл есть, в `state` нет | Новый | Проиндексировать |
| В `state` есть, файла нет | Удалён | Удалить из ChromaDB |

**Шаг 4 — Ранний выход**
```python
if not files_to_update and not deleted_files:
    log("Изменений нет. Выход.")
    return
```
Модель эмбеддингов не загружается, ChromaDB не трогается. Важно для cron: запуск каждые 10 минут без изменений занимает миллисекунды.

**Шаг 5 — Обработка изменений**
```python
embeddings_model = get_embeddings_model()   # загружаем только теперь

for filepath_str, filepath, new_hash in files_to_update:
    action = "Обновление" if filepath_str in state else "Добавление"
    delete_file_chunks(collection, filepath_str)   # удалить старые чанки
    n = upsert_file(collection, embeddings_model, filepath)  # загрузить новые
    state[filepath_str] = new_hash   # обновить хэш в памяти (не на диске!)

for filepath_str in deleted_files:
    delete_file_chunks(collection, filepath_str)
    del state[filepath_str]
```

Хэши обновляются в памяти после каждого файла, но `save_state` вызывается один раз в конце — это атомарнее (если упадёт посередине, старое состояние не повредится).

**Шаг 6 — Сохранение и итог**
```python
save_state(state)
log(f"Документов в коллекции после: {collection.count()}")
```

---

## 4. `clear_collection.py` — сброс базы

**Путь:** `clear_collection.py`

### Режимы запуска

```bash
python clear_collection.py              # удалить ChromaDB + сбросить хэши (с подтверждением)
python clear_collection.py --force      # то же без подтверждения (для скриптов/CI)
python clear_collection.py --state-only # только сбросить хэши, ChromaDB не трогать
```

`--state-only` нужен когда ChromaDB уже пуста (пересоздали контейнер), но `INDEX_STATE_FILE` остался. Без сброса хэшей индексер будет считать всё актуальным и ничего не переиндексирует.

### Функция `clear_chroma(force)`

```python
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME not in existing:
    log("Коллекция не найдена — ничего удалять не нужно.")
else:
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()

    if not force:
        answer = input(f"Удалить безвозвратно? [yes/N]: ").strip().lower()
        if answer != "yes":
            sys.exit(0)

    client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
```

Проверка существования защищает от ошибки при удалении несуществующей коллекции.

Защита `[yes/N]` требует точно ввести `"yes"` — `"y"`, `"да"`, `"ок"` не сработают. `sys.exit(0)` — выход с кодом 0 (успешное завершение, не ошибка).

После удаления коллекция создаётся заново с той же метрикой `cosine`. Это критично: если просто удалить и не создать, `get_or_create_collection` в `indexer.py` создаст коллекцию с дефолтной метрикой `l2` (евклидово расстояние), что изменит поведение поиска.

### Функция `clear_state(force)`

```python
with open(INDEX_STATE_FILE, "r", encoding="utf-8") as f:
    state = json.load(f)

count = len(state)   # показываем пользователю что будет сброшено

# ... подтверждение ...

with open(INDEX_STATE_FILE, "w", encoding="utf-8") as f:
    json.dump({}, f)
```

Файл не удаляется — перезаписывается пустым словарём `{}`. После этого `load_state()` в индексере вернёт `{}`, и при следующем запуске все файлы будут считаться новыми — полная переиндексация.

### Функция `main()`

```python
if args.state_only:
    clear_state(force=args.force)
else:
    clear_chroma(force=args.force)
    clear_state(force=args.force)
```

Порядок важен: сначала физическое удаление данных из ChromaDB, потом сброс учётных записей о них. Наоборот было бы опасно — если упасть после сброса хэшей но до удаления ChromaDB, состояние будет несинхронным.

---

## 5. Взаимосвязь компонентов

```
Файловая система (.txt)              ChromaDB               INDEX_STATE_FILE
         │                               │                         │
         ├── scan_txt_files()            │                    load_state()
         │                               │                         │
         ├── md5_file() ─────────────────┼─────────────────► сравниваем хэши
         │                               │                         │
         │   [файл новый / изменился]    │                         │
         ├── split_file()                │                         │
         ├── embed_documents()           │                         │
         └── upsert() ─────────────────►│                    save_state()
                                    новые чанки               новые хэши
         │
         │   [файл удалён]              │                         │
         └── delete_file_chunks() ─────►│                    del state[fp]
                                   удалены чанки


retriever.py (граф RAG):
    поисковый запрос → embed_query() → similarity_search() ──► ChromaDB
                       та же модель!                       ◄── топ-3 Document


clear_collection.py:
    delete_collection() ─────────────► ChromaDB очищена
    json.dump({}) ────────────────────────────────────────► INDEX_STATE_FILE сброшен
```

### Связь с retriever.py

`indexer.py` и `retriever.py` связаны через одну константу — `EMBEDDINGS_MODEL`. Оба используют одну и ту же модель:

- `indexer.py`: `HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)` при индексации
- `retriever.py`: `HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)` при поиске

Если поменять модель и переиндексировать — нужно перезапустить и сервис с графом, иначе поиск сломается: старые векторы в ChromaDB и новые запросы будут в разных математических пространствах.

---

## 6. Типичные сценарии использования

### Первый запуск (база пуста)

```bash
python indexer.py
```

`load_state()` возвращает `{}` → все файлы новые → загружается модель → все файлы индексируются → `save_state()` сохраняет хэши всех файлов.

### Плановое обновление по cron

```
*/10 * * * * /usr/bin/python3 /path/to/indexer.py >> /var/log/indexer.log 2>&1
```

Если файлы не менялись — выход за миллисекунды, модель не загружается. Если добавился новый файл — индексируется только он.

### Изменился один документ

```bash
# редактируем файл knowledge/policy.txt
python indexer.py
```

Индексер обнаружит изменение MD5 только у `policy.txt`, удалит его старые чанки, загрузит новые. Остальные файлы не трогаются.

### Сменили модель эмбеддингов

```bash
# 1. Обновить EMBEDDINGS_MODEL в .env
# 2. Полный сброс (старые векторы несовместимы с новой моделью)
python clear_collection.py --force
# 3. Переиндексация всего с новой моделью
python indexer.py
# 4. Перезапустить сервис с графом RAG (retriever.py использует ту же модель)
```

### Пересоздали контейнер с ChromaDB

```bash
# ChromaDB пуста, но INDEX_STATE_FILE остался со старыми хэшами
# Просто сброс хэшей без удаления ChromaDB
python clear_collection.py --state-only --force
# Переиндексация
python indexer.py
```

### Ручная полная очистка без подтверждений (CI/CD)

```bash
python clear_collection.py --force
python indexer.py
```

---

*Далее: отдельный блок по конфигурации и Streamlit-интерфейсу.*