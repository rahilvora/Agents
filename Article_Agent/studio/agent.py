from langgraph.graph import START, StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict
import operator
from typing import  Annotated
from langchain_community.tools.tavily_search import TavilySearchResults

# Define search tool
tavily_search = TavilySearchResults(max_results=3)

# Define LLM with access to tools
llm = ChatOpenAI(model="gpt-4o")

# Prompt for the LLM to generate an article
article_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible article for {topic} based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Content (### header)
c. Code examples (### header)
d. Conclusion (### header)
e. Sources (### header)

4. Make your title engaging for the user. Make it more metaphorical and engaging with the real world anolody: 

5. For the content section:
- Set up content with general background / context related to the title of the article
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

class Article(BaseModel):
  """Article data structure."""
  title: str
  content: str
  code_examples: list[str]
  conclusion: str
  sources: list[str]

# Define the state graph
class GraphState(TypedDict):
  topic: str
  context: Annotated[list, operator.add]
  article: Article

# Method to search the web
def search_web(state: GraphState):
  # message to search the web about the topic
  search_message = f'Search the web specifically about the information on {state.get("topic", "")} topic.'
  # invoke the search tool
  search_docs = tavily_search.invoke(search_message)
  
  # Format
  formatted_search_docs = "\n\n---\n\n".join(
    [
      f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
      for doc in search_docs
    ]
  )

  return {"context": [formatted_search_docs]}

# Method to generate an article
def generate_article(state: GraphState):
	# message to generate an article 
	llm_with_structured_output = llm.with_structured_output(Article)
	# invoke the LLM tool
	article = llm_with_structured_output.invoke(article_writer_instructions.format(topic=state.get("topic","")))
	return {"article": article}

# create a tool to write article to a file write
def write_article_to_file(state: GraphState):
	"""Write the article to a file."""
	article = state.get("article")
	if article is not None:
		with open("article.txt", "w") as f:
			f.write(getattr(article, "title", "") + "\n")
			f.write(getattr(article, "content", "") + "\n")
			f.write("\n".join(getattr(article, "code_examples", [])) + "\n")
			f.write(getattr(article, "conclusion", "") + "\n")
			f.write("\n".join(getattr(article, "sources", [])) + "\n")

def write_article_condition(state: GraphState):
	"""Write the article to a file."""
	if state.get("article") is not None:
		return "write_article_to_file"
	return END

# Add the nodes to the graph
article_builder = StateGraph(GraphState)
article_builder.add_node("search_web", search_web)
article_builder.add_node("generate_article", generate_article)
article_builder.add_node("write_article_to_file", write_article_to_file)

# Add the edges to the graph
article_builder.add_edge(START, "search_web")
article_builder.add_edge("search_web", "generate_article")
article_builder.add_conditional_edges("generate_article", write_article_condition)
article_builder.add_edge("write_article_to_file", END)

# Define the agent
graph = article_builder.compile()




