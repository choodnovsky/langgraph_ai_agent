# src/graph_builder.py

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.components.generate_answer import generate_answer
from src.components.generate_query import generate_query_or_respond
from src.components.grade_documents import grade_documents
from src.components.retriever_tool import retriever_tool
from src.components.rewrite_question import rewrite_question
from src.components.analysis_step import analysis_step


def build_graph():
    workflow = StateGraph(MessagesState)

    # Узлы
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("analysis", analysis_step)
    workflow.add_node("generate_answer", generate_answer)

    # Старт
    workflow.add_edge(START, "generate_query_or_respond")

    # Решаем — нужен ли retriever
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # После retrieve оцениваем документы
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "rewrite": "rewrite_question",
            "generate": "analysis",
        },
    )

    # После анализа — финальный ответ
    workflow.add_edge("analysis", "generate_answer")

    # После rewrite — снова в цикл
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Завершение
    workflow.add_edge("generate_answer", END)

    return workflow.compile()