# graph/builder.py
# !/usr/bin/env python3
"""
RAG система с LangGraph + персистентная память через PostgreSQL.

Checkpointer сохраняет состояние графа (историю messages) в Postgres после
каждого шага. При перезапуске граф восстанавливает историю по thread_id.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from graph.state import GraphState


class GraphState(MessagesState):
    """Расширенный State для отслеживания дополнительной информации.

    Наследуется от MessagesState, который предоставляет:
    - messages: List[BaseMessage] - история сообщений

    Добавляем:
    - rewrite_count: int - количество попыток переформулирования вопроса
    """
    rewrite_count: int = 0


def build_graph(use_checkpointer: bool = False):
    """Построить граф с персистентной памятью через PostgreSQL.

    Структура графа не изменилась. Добавлен PostgresSaver как checkpointer —
    он автоматически сохраняет state после каждого узла и восстанавливает
    историю по thread_id при следующем вызове.

    Таблицы в Postgres создаются автоматически при первом запуске (setup()).
    """
    from graph.nodes.query import generate_query_or_respond
    from graph.nodes.grader import grade_documents
    from graph.nodes.answer import generate_answer
    from graph.nodes.rewriter import rewrite_question
    from graph.nodes.retriever import retriever_tool

    # ── Граф (структура не изменилась) ───────────────────────────────────────
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        }
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # ── Checkpointer (только для Streamlit, Studio управляет сам) ───────────
    if use_checkpointer:
        import psycopg
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver
        from config.settings import settings

        conn = psycopg.connect(
            settings.POSTGRES_URI,
            autocommit=True,
            row_factory=dict_row,
        )
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()

# Экспорт для LangGraph Studio (без checkpointer — Studio управляет сам)
graph = build_graph(use_checkpointer=False)