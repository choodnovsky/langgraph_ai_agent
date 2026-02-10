from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.components.generate_query import generate_query_or_respond
from src.components.generate_answer import generate_answer
from src.components.grade_documents import grade_documents
from src.components.rewrite_question import rewrite_question
from src.components.retriever_tool import retriever_tool

def build_graph():
    workflow = StateGraph(MessagesState)

    # Узлы
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Рёбра
    workflow.add_edge(START, "generate_query_or_respond")

    # Решаем, нужно ли извлекать документы
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    # После retrieval — оцениваем документы и решаем, что делать
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"},
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()