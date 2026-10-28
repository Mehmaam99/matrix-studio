import os
from app.memory import get_history, save_message
from dotenv import load_dotenv
from openai import OpenAI
from app.tools import TOOL_REGISTRY
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

if not api_key:
    raise RuntimeError("GROQ_API_KEY .env mein set nahi hai")

if not model:
    raise RuntimeError("GROQ_MODEL .env mein set nahi hai")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


import json


def ask_llm(message: str, session_id: str = "default") -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_appointment_availability",
                "description": "Check demo appointment availability for a date.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                    "required": ["date"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_clinic_hours",
                "description": "Get clinic opening hours and working days.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a dental clinic receptionist. Use tools when appropriate. "
                "Never claim a real appointment was booked. Use conversation history "
                "to answer follow-up questions."
            ),
        },
        *get_history(session_id),
        {"role": "user", "content": message},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.tool_calls:
        print("TOOL CALLS:", assistant_message.tool_calls)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)

                if tool_name not in TOOL_REGISTRY:
                    result = {"error": "Unknown tool"}
                elif not isinstance(arguments, dict):
                    result = {"error": "Tool arguments must be an object"}
                else:
                    result = TOOL_REGISTRY[tool_name](**arguments)

            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                result = {"error": f"Tool execution failed: {exc}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        answer = final_response.choices[0].message.content or ""
    else:
        answer = assistant_message.content or ""

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", answer)

    return answer


if __name__ == "__main__":
    print(ask_llm("What are the clinic's opening hours?"))
    print("-" * 40)
    print(ask_llm("Is an appointment available on 2026-10-17?"))