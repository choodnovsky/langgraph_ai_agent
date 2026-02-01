# src/graph_builder.py
import streamlit as st
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.settings import settings
from src.tools import TOOLS

@st.cache_resource
def build_graph():
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_retries=2,
        base_url="https://openrouter.ai/api/v1",
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