from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Define math tools. Note: Each tool function should have DocString or else compilation will fail
def add (a, b):
    """
    Add two numbers
    """
    return a + b

def subtract (a, b):
    """
    Subtract two numbers
    """
    return a - b

def multiply (a, b):
    """
    Multiply two numbers
    """
    return a * b

def divide (a, b):
    """
    Divide two numbers
    """
    return a / b

def power (a, b):
    """
    Raise a to the power of b
    """
    return a ** b

def square_root (a):
    """
    Return the square root of a
    """
    return a ** 0.5

tools = [add, subtract, multiply, divide, power, square_root]

# Define LLM with access to tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
system_message = SystemMessage(content="You are a Math Agent. That can help you with math operations.")

# Define the math agent node
def math_assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

# Build Graph
builder = StateGraph(MessagesState)
builder.add_node("math_assistant", math_assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "math_assistant")
builder.add_conditional_edges(
    "math_assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "math_assistant")

# Compile the graph
graph = builder.compile()