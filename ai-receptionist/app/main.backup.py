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


def ask_llm(message: str) -> str:
    tools = [
    {
        "type": "function",
        "function": {
            "name": "check_appointment_availability",
            "description": "Check demo appointment availability for a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                },
                "required": ["date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clinic_hours",
            "description": "Get the clinic's opening hours and working days.",
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
            "content": "You are a dental clinic receptionist. Use the availability tool when asked about appointment slots. Never claim a real booking was made.",
        },
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
            arguments = json.loads(tool_call.function.arguments)

            if tool_name not in TOOL_REGISTRY:
                result = {"error": "Unknown tool"}
            else:
                try:
                    result = TOOL_REGISTRY[tool_name](**arguments)
                except (TypeError, ValueError) as exc:
                    result = {"error": f"Invalid tool arguments: {exc}"}

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

        return final_response.choices[0].message.content or ""

    return assistant_message.content or ""



if __name__ == "__main__":
    print(ask_llm("What are the clinic's opening hours?"))
    print("-" * 40)
    print(ask_llm("Is an appointment available on 2026-10-17?"))