from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.main import ask_llm


class ReceptionistState(TypedDict):
    message: str
    session_id: str
    answer: str
    error: str


def validate_input(state: ReceptionistState) -> dict:
    message = state.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return {"error": "Message cannot be empty."}

    if len(message) > 2000:
        return {"error": "Message exceeds the 2000-character limit."}

    return {"error": ""}


def route_after_validation(state: ReceptionistState) -> str:
    if state.get("error"):
        return "reject"
    return "receptionist"


def receptionist_node(state: ReceptionistState) -> dict:
    answer = ask_llm(
        message=state["message"],
        session_id=state["session_id"],
    )
    return {"answer": answer}


def reject_node(state: ReceptionistState) -> dict:
    return {"answer": state["error"]}


workflow = StateGraph(ReceptionistState)

workflow.add_node("validate_input", validate_input)
workflow.add_node("receptionist", receptionist_node)
workflow.add_node("reject", reject_node)

workflow.add_edge(START, "validate_input")

workflow.add_conditional_edges(
    "validate_input",
    route_after_validation,
    {
        "receptionist": "receptionist",
        "reject": "reject",
    },
)

workflow.add_edge("receptionist", END)
workflow.add_edge("reject", END)

graph = workflow.compile()


def run_receptionist_graph(
    message: str,
    session_id: str = "default",
) -> dict:
    initial_state: ReceptionistState = {
        "message": message,
        "session_id": session_id,
        "answer": "",
        "error": "",
    }
    return graph.invoke(initial_state)
