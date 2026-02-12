# src/components/analysis_step.py

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState


def analysis_step(state: MessagesState):
    return {
        "messages": [
            AIMessage(content="Анализирую имеющуюся информацию...")
        ]
    }