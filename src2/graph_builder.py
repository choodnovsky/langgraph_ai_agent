# src/graph_builder.py
# !/usr/bin/env python3
"""
Пример использования RAG системы с LangGraph
"""
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

from src2.components.generate_query import generate_query_or_respond
from src2.components.grade_documents import grade_documents
from src2.components.generate_answer import generate_answer
from src2.components.rewrite_question import rewrite_question
from src2.components.retriever_tool import retriever_tool


def build_graph():
    """Создание RAG графа с ChromaDB"""

    # Создаем граф
    workflow = StateGraph(MessagesState)

    # Добавляем узлы
    workflow.add_node("generate_query", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("rewrite_question", rewrite_question)

    # Рёбра
    workflow.set_entry_point("generate_query")

    # После generate_query: либо tool, либо END
    workflow.add_conditional_edges(
        "generate_query",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        }
    )

    # После retrieve: оцениваем документы и идём либо в generate_answer, либо в rewrite_question
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,  # Напрямую используем функцию - она возвращает строку
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        }
    )

    # После rewrite_question: возвращаемся к generate_query
    workflow.add_edge("rewrite_question", "generate_query")

    # После generate_answer: END
    workflow.add_edge("generate_answer", END)

    # Компиляция
    return workflow.compile()