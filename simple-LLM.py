import os
from openai import OpenAI


model_id = "nvidia/nemotron-3-super-120b-a12b:free"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Loop until user enters "quit"
while True:
    # Query to send to llm
    query = input("👤 Enter your query (or 'quit' to exit): ")

    # Check if user wants to quit
    if query.lower() == "quit":
        print("Goodbye!")
        break

    # Make the API call using OpenRouter
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=1024
        )

        # Extract and print the response
        output = response.choices[0].message.content
        print(f"👤 Query: {query}")
        print(f"\nResponse:\n{output}\n")

    except Exception as e:
        print(f"Error calling OpenRouter: {e}\n")