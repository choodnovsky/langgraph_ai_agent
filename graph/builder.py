# graph/builder.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import GraphState

def route_after_review(state: GraphState):
    """После human_review — если одобрено идём в grader, иначе rewriter."""
    if state.get("human_approved") is False:
        return "rewriter"
    return "grader"

def build_graph(use_checkpointer: bool = False):
    """Построить RAG граф с самокоррекцией и human-in-the-loop.

    Структура графа:
        START → query
          ├─→ retrieve → reviewer (interrupt)
          │     ├─→ ДА  → answer → should_summarize → summarizer → END
          │     │                                   └─→ END
          │     └─→ НЕТ → rewriter → query
          └─→ END
    """
    from graph.nodes.query import generate_query_or_respond
    from graph.nodes.grader import grade_documents
    from graph.nodes.answer import generate_answer
    from graph.nodes.rewriter import rewrite_question
    from graph.nodes.retriever import retriever_tool
    from graph.nodes.summarizer import summarize_conversation, should_summarize
    from graph.nodes.reviewer import human_review

    workflow = StateGraph(GraphState)

    workflow.add_node("query", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("reviewer", human_review)
    workflow.add_node("answer", generate_answer)
    workflow.add_node("rewriter", rewrite_question)
    workflow.add_node("summarizer", summarize_conversation)

    workflow.add_edge(START, "query")

    # query: tool call → retrieve, прямой ответ → проверка суммаризации
    workflow.add_conditional_edges(
        "query",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    # retrieve → reviewer (interrupt)
    workflow.add_edge("retrieve", "reviewer")

    # reviewer → grader или rewriter
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "grader": "answer",
            "rewriter": "rewriter",
        }
    )

    workflow.add_conditional_edges(
        "answer",
        should_summarize,
        {
            "summarizer": "summarizer",
            "__end__": END,
        }
    )

    workflow.add_edge("summarizer", END)
    workflow.add_edge("rewriter", "query")

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


# Экспорт для LangGraph Studio
graph = build_graph(use_checkpointer=False)