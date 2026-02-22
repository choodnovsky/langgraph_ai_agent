# graph/builder.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import GraphState


def build_graph(use_checkpointer: bool = False):
    """Построить RAG граф с самокоррекцией.

    Параметры:
    - use_checkpointer=True  — для Streamlit, память через PostgreSQL
    - use_checkpointer=False — для LangGraph Studio, Studio управляет памятью сам

    Структура графа:
        START
          ↓
        query        (LLM: искать или ответить напрямую)
          ├─→ summarizer    (прямой ответ → проверка нужна ли сводка)
          └─→ retrieve
               ↓
             grader
               ├─→ answer → summarizer
               └─→ rewriter → query (новая попытка, макс 2)
          summarizer
               ├─→ END      (история короткая)
               └─→ END      (история свёрнута в сводку)
    """
    from graph.nodes.query import generate_query_or_respond
    from graph.nodes.grader import grade_documents
    from graph.nodes.answer import generate_answer
    from graph.nodes.rewriter import rewrite_question
    from graph.nodes.retriever import retriever_tool
    from graph.nodes.summarizer import summarize_conversation, should_summarize

    workflow = StateGraph(GraphState)

    workflow.add_node("query", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("answer", generate_answer)
    workflow.add_node("rewriter", rewrite_question)
    workflow.add_node("summarizer", summarize_conversation)

    workflow.add_edge(START, "query")

    workflow.add_conditional_edges(
        "query",
        tools_condition,
        {
            "tools": "retrieve",
            END: "summarizer",
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "answer": "answer",
            "rewriter": "rewriter",
        }
    )

    workflow.add_edge("answer", "summarizer")
    workflow.add_edge("rewriter", "query")
    workflow.add_conditional_edges("summarizer", should_summarize)

    # ── Checkpointer ─────────────────────────────────────────────────────────
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


# Экспорт для LangGraph Studio (Studio управляет памятью сам через POSTGRES_URI)
graph = build_graph(use_checkpointer=False)