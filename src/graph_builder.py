# src/graph_builder.py
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.settings import settings
from src.tools import TOOLS

def build_graph():
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_retries=2,
        base_url=settings.BASE_URL,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    system = SystemMessage(
        content=(
            "Ты арифметический помощник.\n"
            "Ты ОБЯЗАН помнить предыдущие результаты.\n"
            "Если требуется вычисление — всегда вызывай инструмент."
        )
    )

    def assistant(state: MessagesState):
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()