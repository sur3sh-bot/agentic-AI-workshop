from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent

# Set Parameters:
model_id = ""

# Initialize OpenRouter LLM
model = ChatOpenAI(
    model=model_id,
    openai_api_key="",
)


# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


add_agent = create_agent(
    model=model,
    tools=[add],
    name="add_agent",
    system_prompt="You are an addition expert."
)

multiply_agent = create_agent(
    model=model,
    tools=[multiply],
    name="multiply_agent",
    system_prompt="You are a multiplication expert."
)

divide_agent = create_agent(
    model=model,
    tools=[divide],
    name="divide_agent",
    system_prompt="You are a division expert."
)

# Create supervisor workflow
workflow = create_supervisor(
    [add_agent, multiply_agent, divide_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing math experts."
        "For addition problems, use add_agent."
        "For multiplication problems, use multiply_agent."
        "For division problems, use divide_agent."
    )
)

# Compile and run
app = workflow.compile()
result_supervisor = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's (34531235 + 73453412312) * 31231335345 / 2353413123?"
        }
    ]
})

# show the answer
print(result_supervisor["messages"][-1].content)

# Extract system handoff:
visualize_agent_flow(result_supervisor)