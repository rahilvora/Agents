from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

# Define Agent State
class GraphState(MessagesState):
  summary: str

# Define the LLM model
llm = ChatOpenAI(model="gpt-4o")

# Define the call to modal node logic
def conversation(state: GraphState):
  summary = state.get("summary", "")
  if (summary):
    system_message = f"Here is the summary: {summary}. Please use conversation summary to has questions more accurately."
    messages = [SystemMessage(content=system_message) + state["messages"]]
  else:
    messages = state["messages"]
  response = llm.invoke(messages)
  return {"messages": response}

# Define the summarization node logic
def summarization(state: GraphState):
  summary = state.get("summary", "")
  if summary:
    summary_message_prompt = (
      f"Here is the summary: {summary} \n\n."
      "Extend the summary by taking into account the new messages above:"
      )
  else:
    summary_message_prompt = "create summary of the conversation above:"
  
  messages = state["messages"] + [HumanMessage(content=summary_message_prompt)]
  response = llm.invoke(messages)

  # Delete all but the last couple of messages
  latest_messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]
  return {"messages": latest_messages, "summary": response.content}

def summarization_condition_node(state: GraphState):
  messages = state["messages"]
  if (len(messages) > 3):
    return "summarization"
  return END

# create the graph

builder = StateGraph(GraphState)
builder.add_node("conversation", conversation)
builder.add_node("summarization", summarization)
builder.add_edge(START, "conversation")
builder.add_conditional_edges("conversation", summarization_condition_node)
builder.add_edge("summarization", END)

graph = builder.compile()

# compile the graph