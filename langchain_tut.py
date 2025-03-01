import os 
from langchain_openai import ChatOpenAI
from langchain_community.tools.brave_search.tool import BraveSearch
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent # ReAct agent: Reasoning and Acting; Thinks -> Acts -> Loops Until Done
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
brave_api_key = os.getenv("BRAVE_API_KEY")

memory = MemorySaver()

model = ChatOpenAI(model = "gpt-4o-mini", api_key=api_key) #type: ignore
search = BraveSearch.from_api_key(api_key = brave_api_key, search_kwargs = {"max_results": 4}) #type: ignore
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer = memory)

config = {"configurable": {"thread_id": "abc123"}}

for step in agent_executor.stream(
    {"messages": [HumanMessage(content= "Who am I ?")]},
    config = config, #type: ignore
    stream_mode = "values",
):
    step["messages"][-1].pretty_print()
