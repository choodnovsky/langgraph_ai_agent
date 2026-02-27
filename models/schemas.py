# models/schemas.py
"""
Доменные модели проекта.
Все датаклассы хранятся здесь — не в settings.py и не в модулях аналитики.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Feedback ─────────────────────────────────────────────────────────────────

@dataclass
class Question:
    id: int
    thread_id: str
    message_id: str
    question: str
    answer: str
    rating: int          # 1 = лайк, -1 = дизлайк
    created_at: str


# ── Аналитика кластеров ───────────────────────────────────────────────────────

@dataclass
class ClusterStats:
    cluster_id: int
    label: str           # название темы от LLM
    description: str     # описание темы от LLM
    total: int = 0
    likes: int = 0
    dislikes: int = 0
    like_rate: float = 0.0
    questions: list[Question] = field(default_factory=list)

    @property
    def health(self) -> str:
        """Визуальный индикатор здоровья кластера."""
        if self.dislikes == 0:
            return "✅ Отлично"
        if self.like_rate >= 0.9:
            return "🟠 Аномалия"
        if self.like_rate >= 0.6:
            return "🟡 Зона риска"
        return "🔴 Проблема"

    @property
    def alert_score(self) -> float:
        """
        Score тревоги для сортировки — выше = важнее смотреть.

        Логика:
        - Кластер без дизлайков → 0.0
        - Хроническая проблема (низкий like_rate, много дизлайков) → высокий score
        - Аномалия (высокий like_rate, но дизлайк вдруг появился) → средний score,
          но выше чем просто зона риска с теми же дизлайками,
          потому что это неожиданное падение качества
        """
        if self.dislikes == 0:
            return 0.0
        dislike_rate = 1.0 - self.like_rate
        anomaly_bonus = self.like_rate * self.dislikes * 0.5
        return dislike_rate * self.total + anomaly_bonus