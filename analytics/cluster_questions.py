# analytics/cluster_questions.py
"""
Кластеризация всех вопросов с рейтингом как сигналом качества.

Алгоритм:
1. Загружаем все вопросы с рейтингами из PostgreSQL
2. Векторизуем той же HuggingFace-моделью что используется в indexer.py
3. Кластеризуем через KMeans (sklearn)
4. Считаем метрики по каждому кластеру (лайки/дизлайки/процент)
5. Для каждого кластера просим LLM дать название темы
6. Возвращаем структуру для отображения в Streamlit

Запуск как скрипт:
    python -m analytics.cluster_questions
"""

from __future__ import annotations

import numpy as np
import psycopg
from typing import Optional

from config.settings import settings
from models.schemas import Question, ClusterStats


# ── Загрузка данных из PostgreSQL ────────────────────────────────────────────

def load_all_questions(min_count: int = 5) -> list[Question]:
    """
    Загружает все вопросы с рейтингами из таблицы feedback.
    Возвращает пустой список если вопросов меньше min_count.
    """
    with psycopg.connect(settings.POSTGRES_URI) as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, message_id, question, answer, rating, created_at
            FROM feedback
            WHERE question IS NOT NULL
              AND question != ''
            ORDER BY created_at DESC
            """,
        ).fetchall()

    if len(rows) < min_count:
        return []

    return [
        Question(
            id=r[0],
            thread_id=r[1],
            message_id=r[2],
            question=r[3],
            answer=r[4] or "",
            rating=r[5],
            created_at=str(r[6]),
        )
        for r in rows
    ]


# ── Векторизация ─────────────────────────────────────────────────────────────

def embed_questions(questions: list[str]) -> np.ndarray:
    """
    Векторизует вопросы той же моделью что используется в indexer.py.
    Возвращает numpy array shape (n, dim).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL)
    vectors = model.embed_documents(questions)
    return np.array(vectors, dtype=np.float32)


# ── Кластеризация ─────────────────────────────────────────────────────────────

def optimal_k(n_samples: int, max_k: int = 8) -> int:
    """Эвристика: sqrt(n/2), но не больше max_k и не меньше 2."""
    return max(2, min(max_k, int(np.sqrt(n_samples / 2))))


def cluster_vectors(vectors: np.ndarray, k: int) -> np.ndarray:
    """KMeans кластеризация. Возвращает массив меток длиной n_samples."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(vectors)
    return km.labels_


# ── LLM — названия кластеров ─────────────────────────────────────────────────

def label_cluster_with_llm(questions_sample: list[str]) -> tuple[str, str]:
    """
    Просит LLM дать короткое название темы и описание для кластера.
    questions_sample — до 10 вопросов из кластера.
    Возвращает (label, description).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(
        model=settings.OPENAI_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        base_url=settings.BASE_URL,
        model_provider="openai",
    )

    sample_text = "\n".join(f"- {q}" for q in questions_sample[:10])

    messages = [
        SystemMessage(content=(
            "Ты аналитик RAG-системы. "
            "Тебе дают список похожих вопросов пользователей из одного тематического кластера. "
            "Определи общую тему и дай: "
            "1) короткое название темы (до 5 слов), "
            "2) одно предложение — о чём спрашивают пользователи в этом кластере. "
            "Ответ строго в формате:\n"
            "LABEL: <название>\n"
            "DESC: <описание>"
        )),
        HumanMessage(content=f"Вопросы:\n{sample_text}"),
    ]

    response = llm.invoke(messages)
    text = response.content.strip()

    label, description = "Без названия", "Описание не получено"
    for line in text.splitlines():
        if line.startswith("LABEL:"):
            label = line.removeprefix("LABEL:").strip()
        elif line.startswith("DESC:"):
            description = line.removeprefix("DESC:").strip()

    return label, description


# ── Основная функция ──────────────────────────────────────────────────────────

def get_question_clusters(
    min_questions: int = 10,
    max_k: int = 8,
    use_llm_labels: bool = True,
) -> Optional[list[ClusterStats]]:
    """
    Полный пайплайн: загрузка → векторизация → кластеризация → метрики → LLM-названия.

    Args:
        min_questions:  минимум вопросов для запуска (иначе None)
        max_k:          максимальное число кластеров
        use_llm_labels: False = просто номера кластеров (быстро, без LLM)

    Returns:
        list[ClusterStats] отсортированный по like_rate (проблемные сверху),
        или None если мало данных.
    """
    all_questions = load_all_questions(min_count=min_questions)
    if not all_questions:
        return None

    texts = [q.question for q in all_questions]
    vectors = embed_questions(texts)

    k = optimal_k(len(all_questions), max_k=max_k)
    labels = cluster_vectors(vectors, k=k)

    # Группируем по кластерам и считаем метрики
    clusters: dict[int, ClusterStats] = {}
    for question, cluster_id in zip(all_questions, labels):
        cid = int(cluster_id)
        if cid not in clusters:
            clusters[cid] = ClusterStats(
                cluster_id=cid,
                label=f"Кластер {cid}",
                description="",
            )
        c = clusters[cid]
        c.questions.append(question)
        c.total += 1
        if question.rating == 1:
            c.likes += 1
        else:
            c.dislikes += 1

    for c in clusters.values():
        c.like_rate = c.likes / c.total if c.total > 0 else 0.0
        if use_llm_labels:
            sample = [q.question for q in c.questions]
            c.label, c.description = label_cluster_with_llm(sample)

    # Сортируем по alert_score — проблемные и аномалии сверху
    return sorted(
        clusters.values(),
        key=lambda c: c.alert_score,
        reverse=True,
    )


# ── CLI для ручного запуска ───────────────────────────────────────────────────

if __name__ == "__main__":
    clusters = get_question_clusters(min_questions=5, use_llm_labels=True)
    if clusters is None:
        print("Недостаточно вопросов для кластеризации.")
    else:
        for c in clusters:
            print(f"\n{'='*55}")
            print(f"{c.health}  [{c.total} вопросов | 👍 {c.likes} / 👎 {c.dislikes}]  {c.label}")
            print(f"  {c.description}")
            for q in c.questions[:3]:
                print(f"  • {q.question[:100]}")