# src/custom_embeddings.py

import torch
import torch.nn as nn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List

from src.settings import settings


class CustomEmbeddings(Embeddings):
    """
    Embeddings wrapper:
    - модель берётся из settings.EMBEDDINGS_MODEL
    - base_dim определяется автоматически
    - target_dim принудительно (OpenAI-compatible)
    - L2-нормализация
    """

    def __init__(
        self,
        target_dim: int = 1536,
        device: str = "cpu",
    ):
        self.device = device
        self.target_dim = target_dim

        # 1️⃣ Базовая модель из settings
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
        Identity + zero padding
        """
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

            dim = min(self.base_dim, self.target_dim)
            self.proj.weight[:dim, :dim] = torch.eye(dim)

    def _project_and_normalize(self, vecs: torch.Tensor):
        vecs = vecs.to(self.device)
        out = self.proj(vecs)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        return out.detach().cpu().numpy()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинг списка документов"""
        base = self.base_embeddings.embed_documents(texts)
        base = torch.tensor(base, dtype=torch.float32)
        result = self._project_and_normalize(base)
        return result.tolist()  # Возвращаем list[list[float]]

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг одного запроса"""
        base = self.base_embeddings.embed_query(text)
        base = torch.tensor(base, dtype=torch.float32).unsqueeze(0)
        result = self._project_and_normalize(base)[0]
        return result.tolist()  # Возвращаем list[float]