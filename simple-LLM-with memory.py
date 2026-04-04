import os
from openai import OpenAI

# Set Parameters:
model_id = "nvidia/nemotron-3-super-120b-a12b:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

print("Welcome! I'm your personal assistant with memory. I can remember our conversation. Type 'quit' to stop.")

system_message = "You're a helpful personal assistant."
conversation_history = [{"role": "system", "content": system_message}]

while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "quit":
        print("Agent: Goodbye!")
        break

    # Append user message to memory
    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=conversation_history,
            max_tokens=1024
        )

        # Append the LLM's raw message to memory exactly as it was returned
        message = response.choices[0].message
        conversation_history.append(message.model_dump(exclude_unset=True))

        print("💬 LLM provided final response.\n")
        print("Agent:", message.content)

    except Exception as e:
        print(f"Error calling OpenRouter: {e}")