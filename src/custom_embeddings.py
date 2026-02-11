import torch
import torch.nn as nn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List

from src.settings import settings


class CustomEmbeddings(Embeddings):
    """
    Обёртка для эмбеддингов:
    - модель задаётся в settings.EMBEDDINGS_MODEL
    - базовая размерность определяется автоматически
    - target_dim — размерность проекции (например, OpenAI-compatible 1536)
    - нормализация L2
    """

    def __init__(
        self,
        target_dim: int = 1536,
        device: str = "cpu",
    ):
        self.device = device
        self.target_dim = target_dim

        # 1️⃣ Базовая модель HuggingFace
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL
        )

        # 2️⃣ Автоопределение размерности
        probe = self.base_embeddings.embed_query("dimension probe")
        self.base_dim = len(probe)

        # 3️⃣ Линейная проекция
        self.proj = nn.Linear(self.base_dim, self.target_dim, bias=True).to(self.device)
        self._init_projection()

    def _init_projection(self):
        """
        Identity + zero padding для корректной проекции меньшей размерности
        """
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

            dim = min(self.base_dim, self.target_dim)
            # ставим единичную матрицу для первых dim элементов
            self.proj.weight[:dim, :dim] = torch.eye(dim)

    def _project_and_normalize(self, vecs: torch.Tensor):
        """
        Применяет линейную проекцию и L2-нормализацию
        """
        vecs = vecs.to(self.device)
        out = self.proj(vecs)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        return out.detach().cpu().numpy()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинг списка документов"""
        base = self.base_embeddings.embed_documents(texts)
        base = torch.tensor(base, dtype=torch.float32)
        result = self._project_and_normalize(base)
        return result.tolist()  # list[list[float]]

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг одного запроса"""
        base = self.base_embeddings.embed_query(text)
        base = torch.tensor(base, dtype=torch.float32).unsqueeze(0)
        result = self._project_and_normalize(base)[0]
        return result.tolist()  # list[float]