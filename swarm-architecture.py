from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langgraph_swarm import create_swarm, create_handoff_tool

# Set Parameters:
model_id = ""

# Initialize OpenRouter LLM
model = ChatOpenAI(
    model=model_id,
    openai_api_key="",
)


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
    tools=[
        add,
        create_handoff_tool(agent_name="multiply_agent", description="Transfer to multiply_agent, it can help with multiplication"),
        create_handoff_tool(agent_name="divide_agent", description="Transfer to divide_agent, it can help with division")
    ],
    name="add_agent",
    system_prompt="You are an addition expert."
)

multiply_agent = create_agent(
    model=model,
    tools=[
        multiply,
        create_handoff_tool(agent_name="add_agent", description="Transfer to add_agent, it can help with addition"),
        create_handoff_tool(agent_name="divide_agent", description="Transfer to divide_agent, it can help with division")
    ],
    name="multiply_agent",
    system_prompt="You are a multiplication expert."
)

divide_agent = create_agent(
    model=model,
    tools=[
        divide,
        create_handoff_tool(agent_name="multiply_agent", description="Transfer to multiply_agent, it can help with multiplication"),
        create_handoff_tool(agent_name="add_agent", description="Transfer to add_agent, it can help with addition")
    ],
    name="divide_agent",
    system_prompt="You are a division expert."
)


checkpointer = InMemorySaver()
workflow = create_swarm(
    [add_agent, multiply_agent, divide_agent],
    default_active_agent="add_agent"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

result_swarm = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's (34531235 + 73453412312) * 31231335345 / 2353413123?"
        },
    ],
    },
    config,
)

# show the answer
print(result_swarm["messages"][-1].content)

# visualize flow:
visualize_agent_flow(result_swarm)