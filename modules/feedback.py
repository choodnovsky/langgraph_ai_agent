# modules/feedback.py
import psycopg
import streamlit as st
from config.settings import settings


def init_feedback_table():
    """Создаёт таблицу feedback если не существует."""
    with psycopg.connect(settings.POSTGRES_URI) as conn:
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS feedback
                     (
                         id SERIAL PRIMARY KEY,
                         thread_id TEXT NOT NULL,
                         message_id TEXT NOT NULL,
                         rating SMALLINT NOT NULL CHECK (rating IN (1, -1)),
                         question TEXT,
                         answer TEXT,
                         created_at TIMESTAMPTZ DEFAULT now()
                         )
                     """)
        conn.commit()


def save_feedback(thread_id: str, message_id: str, rating: int,
                  question: str = None, answer: str = None):
    """Сохраняет оценку. rating: 1 = лайк, -1 = дизлайк.
    question/answer — оригинальные тексты без лемматизации.
    """
    with psycopg.connect(settings.POSTGRES_URI) as conn:
        conn.execute(
            """INSERT INTO feedback (thread_id, message_id, rating, question, answer)
               VALUES (%s, %s, %s, %s, %s)""",
            (thread_id, message_id, rating, question, answer),
        )
        conn.commit()


def render_feedback(message_id: str, thread_id: str,
                    question: str = None, answer: str = None):
    """Рендерит кнопки лайк/дизлайк под сообщением.

    Args:
        message_id: LangChain run ID сообщения
        thread_id:  ID треда/сессии
        question:   оригинальный текст вопроса пользователя
        answer:     оригинальный текст ответа модели
    """
    key_like = f"like_{message_id}"
    key_dislike = f"dislike_{message_id}"
    key_done = f"feedback_done_{message_id}"

    if st.session_state.get(key_done):
        st.caption("Оценка сохранена")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    with col1:
        if st.button("👍", key=key_like):
            save_feedback(thread_id, message_id, 1, question, answer)
            st.session_state[key_done] = True
            st.rerun()
    with col2:
        if st.button("👎", key=key_dislike):
            save_feedback(thread_id, message_id, -1, question, answer)
            st.session_state[key_done] = True
            st.rerun()