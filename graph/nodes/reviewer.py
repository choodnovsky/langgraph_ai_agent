# graph/nodes/reviewer.py

from langgraph.types import interrupt
from langchain_core.messages import ToolMessage
from graph.state import GraphState


def human_review(state: GraphState) -> GraphState:
    """Показывает найденный документ пользователю и ждёт подтверждения."""
    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return state

    doc = tool_messages[-1].content[:500]

    # Граф останавливается здесь и ждёт resume
    decision = interrupt({
        "doc_preview": doc,
        "question": "Использовать этот источник для ответа?",
    })

    # decision приходит от пользователя: "yes" или "no"
    return {**state, "human_approved": decision == "yes"}