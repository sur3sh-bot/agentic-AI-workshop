import json
import os
import re
from datetime import datetime
from openai import OpenAI



model_id = "nvidia/nemotron-3-super-120b-a12b:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

# OpenRouter Native Tool Definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression, e.g. '5 * 4'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# Define Tools
def calculate_expression(expression):
    """Calculator: Evaluate a mathematical expression"""
    safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
    if safe_expr.strip() == "":
        return None
    try:
        return eval(safe_expr)
    except:
        return None

def get_weather(location):
    """Weather: Get weather information for a location"""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 58°F",
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F"
    }
    location_lower = location.lower()
    for city, weather in weather_data.items():
        if city in location_lower:
            return f"Weather in {city.title()}: {weather}"
    return f"Weather information for {location} is not available in simulation."

def get_date():
    """Get Date: Get the current date"""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    return f"Today's date is: {date_str}"

def get_time():
    """Get Time: Get the current time"""
    now = datetime.now()
    time_str = now.strftime("%I:%M:%S %p")
    return f"The current time is: {time_str}"

def call_tool(tool_name, tool_input_str):
    """Execute a tool function based on tool name"""
    try:
        args = json.loads(tool_input_str)
    except:
        args = {}

    if tool_name == "calculator":
        result = calculate_expression(args.get("expression", ""))
        return f"The result is: {result}" if result is not None else "I couldn't compute that."
    elif tool_name == "get_weather":
        return get_weather(args.get("location", ""))
    elif tool_name == "get_date":
        return get_date()
    elif tool_name == "get_time":
        return get_time()
    else:
        return f"Unknown tool: {tool_name}"

print("Welcome! I'm your personal assistant. I can tell you the current date, time, and weather. I can also calculate mathematical expressions. Type 'quit' to stop.")

system_message = input("enter the role: ")
conversation_history = [{"role": "system", "content": system_message}]

while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "quit":
        print("Agent: Goodbye!")
        break

    # Append user's new message to the actual memory
    conversation_history.append({"role": "user", "content": user_input})

    # Loop until the LLM returns a final natural language response (no tools)
    while True:
        response = client.chat.completions.create(
            model=model_id,
            messages=conversation_history,
            tools=TOOLS,
            max_tokens=1024
        )

        # Append the LLM's raw message to memory exactly as it was returned
        message = response.choices[0].message
        conversation_history.append(message.model_dump(exclude_unset=True))

        # Check if the model chose to use a tool
        if message.tool_calls:
            print("\n🔄 LLM Response (Tool Call):")
            for tool_call in message.tool_calls:
                print(f"[{tool_call.function.name}] => {tool_call.function.arguments}")

                # Execute tool
                tool_result = call_tool(tool_call.function.name, tool_call.function.arguments)

                # Append tool result directly back into memory (which will be fed right back to the LLM the next loop)
                print("\n🧠 Feeding tool result back to LLM for final response...")
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            # The model replied without tools
            print("Agent:", message.content)
            break