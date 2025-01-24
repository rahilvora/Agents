from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import operator
from typing import List
import operator
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.constants import Send

# Define search tool
tavily_search = TavilySearchResults(max_results=3)

# Define LLM with access to tools
llm = ChatOpenAI(model="gpt-4o")

# Prompts
subtopic_query_prompt = """
 You are a research assistant. Your assigned a task to create sub-topic if possible for the given {topic}.
 if it is possible to create sub-topics create at most 4 sub-topics
 Please make sure to only consider sub topics that are relevant to the given topic. 
 Do not make any assumptions. If required to do so, perform web search to get the sub-topics."""

section_prompt = """
You are a research assistant. You are assigned a write a section for an article on a given {subtopic}.
please write a consise and informative section on the given subtopic. Do not assume anything.
Please add code examples if required."""

article_prompt = """
You are an expert technical writer. write an article on the given {topic} based on a these sections.
<section>
{sections}
</section>
Please make sure to follow the instructions given below:
1. Add title of the article
2. Add content from each of the section into seperate paragraph with the given name, description and content
3. Add code examples if required
4. Add conclusion
"""

# Schema
class Section(BaseModel):
  name: str = Field(
    description="Name for this section of the article.",
  )
  description: str = Field(
    description="Brief overview of the main topics and concepts to be covered in this section.",
  )
  research: bool = Field(
    description="Whether to perform web research for this section of the article."
  )
  content: str = Field(
    description="The content of the section."
  )   

class SearchQuery(BaseModel):
  search_query: str = Field(None, description="Query for web search.")

class Subtopics(BaseModel):
  names: List[str] = Field(description="List of subtopics for the article.")

class Article(BaseModel):
  title: str = Field(
    description="Title of the article.",
  )
  sections: List[Section] = Field(
    description="Sections of the article.",
  )

class SubtopicState(TypedDict):
  subtopic: str

class ResearchAssistantState(MessagesState):
  topic: str = Field(description="Topic for the article.")
  subtopics: list = Field(description="Subtopics for the article.")
  sections: Annotated[List[Section], operator.add] = Field(description="Sections of the article.")

# Functions
def generate_subtopics(state: ResearchAssistantState):
  prompt = subtopic_query_prompt.format(topic=state["topic"])
  response = llm.with_structured_output(Subtopics).invoke(prompt)
  return {"subtopics": response.names}

def continue_to_generate_sections(state: ResearchAssistantState):
  return [Send("generate_section", {"subtopic": s}) for s in state["subtopics"]]

def generate_section(state: SubtopicState):
  prompt = section_prompt.format(subtopic=state["subtopic"])
  response = llm.with_structured_output(Section).invoke(prompt)
  return {"sections": [response]}

def write_article(state: ResearchAssistantState):
  prompt = article_prompt.format(topic=state["topic"], sections=state["sections"])
  response = llm.with_structured_output(Article).invoke(prompt)
  
  # Format each section into a string
  formatted_sections = []
  for section in response.sections:
    section_text = f"\n## {section.name}\n\n{section.description}\n\n{section.content}"
    formatted_sections.append(section_text)
  
  # Combine title and sections
  full_article = f"# {response.title}\n" + '\n'.join(formatted_sections)
  
  return {"messages": full_article}

article_assistant_builder = StateGraph(ResearchAssistantState)
article_assistant_builder.add_node("generate_subtopics", generate_subtopics)
article_assistant_builder.add_node("generate_section", generate_section)
article_assistant_builder.add_node("write_article", write_article)
article_assistant_builder.add_edge(START, "generate_subtopics")
article_assistant_builder.add_conditional_edges("generate_subtopics", continue_to_generate_sections, ["generate_section"])
article_assistant_builder.add_edge("generate_section", "write_article")
article_assistant_builder.add_edge("write_article", END)

graph = article_assistant_builder.compile()