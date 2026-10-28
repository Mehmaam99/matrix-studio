from collections import defaultdict

SESSIONS: dict[str, list[dict[str, str]]] = defaultdict(list)
MAX_HISTORY_MESSAGES = 12


def get_history(session_id: str) -> list[dict[str, str]]:
    return SESSIONS[session_id]


def save_message(session_id: str, role: str, content: str) -> None:
    history = SESSIONS[session_id]
    history.append({"role": role, "content": content})

    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]
