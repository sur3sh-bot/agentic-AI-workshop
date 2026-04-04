import json
import os
from openai import OpenAI

# Set Parameters:
model_id = "minimax/minimax-m2.5:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)


def schedule_meeting(title, time, participants):
    return f"✅ SUCCESS: Scheduled '{title}' at {time} with {participants}"

# ---------------------------------------------------------
# 1. TRADITIONAL SCRIPT APPROACH (Rigid & Brittle)
# ---------------------------------------------------------
def traditional_assistant(user_input):
    """
    Relies on explicit keywords, exact order, and rigid parsing.
    Requires the user to type like a robot.
    """
    print("\n[Traditional Script Processing...]")
    user_input = user_input.lower()

    # Needs perfect syntax: "schedule meeting: [title] | [time] | [participants]"
    if user_input.startswith("schedule meeting:"):
        try:
            # Strip the command prefix
            args_str = user_input.replace("schedule meeting:", "").strip()

            # Split by a rigid delimiter
            parts = args_str.split("|")

            title = parts[0].strip()
            time = parts[1].strip()
            participants = [p.strip() for p in parts[2].split(",")]

            return schedule_meeting(title, time, participants)

        except IndexError:
            return "❌ ERROR: Invalid format. Must be 'schedule meeting: title | time | person1, person2'"
    else:
        return "❌ ERROR: Unknown command. Type 'help' for a list of commands."

# ---------------------------------------------------------
# 2. AI AGENT APPROACH (Flexible & Intent-Based)
# ---------------------------------------------------------
def agentic_assistant(user_input):
    """
    Relies on LLM reasoning and the native Tool schema.
    Understands messy, human language and automatically extracts structure.
    """
    print("\n[AI Agent Processing...]")

    tools = [{
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting or calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Subject of the meeting"},
                    "time": {"type": "string", "description": "When the meeting happens"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of people attending"
                    }
                },
                "required": ["title", "time", "participants"]
            }
        }
    }]

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": user_input}],
        tools=tools,
        max_tokens=1024
    )

    message = response.choices[0].message

    if message.tool_calls:
        # The LLM automatically parsed the chaos into perfect JSON
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Execute the tool with the beautifully structured arguments
        return schedule_meeting(
            title=args.get("title"),
            time=args.get("time"),
            participants=args.get("participants")
        )
    else:
        return f"Agent replied normally: {message.content}"

# ---------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("TEST 1: The 'Perfect' Robotic-User")
    print("="*60)
    perfect_input = "schedule meeting: Project Sync | Tomorrow 3PM | Alice, Bob"
    print(f"User typed: '{perfect_input}'")
    print(traditional_assistant(perfect_input))
    print(agentic_assistant(perfect_input))

    print("\n" + "="*60)
    print("TEST 2: The 'Messy' Human User")
    print("="*60)
    messy_input = "Hey, can you book a quick sync with Alice and Bob for tomorrow at 3pm to talk about the project?"
    print(f"User typed: '{messy_input}'")
    print(traditional_assistant(messy_input))
    print(agentic_assistant(messy_input))