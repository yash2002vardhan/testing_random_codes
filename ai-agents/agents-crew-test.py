from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv
from crewai_tools import BraveSearchTool
from pydantic import BaseModel

load_dotenv()



llm = LLM(model = "openai/gpt-4o-mini")

class football_event(BaseModel):
    date: str
    time: str

api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv("OPENAI_MODEL_NAME")
brave_api_key = os.getenv("BRAVE_API_KEY")

search_tool = BraveSearchTool()

football_agent = Agent(
    role = "Football Agent",
    goal  = "You can get me the event details for a football match for a particular team",  
    backstory = "YOU ARE A FOOTBALL AGENT",
    verbose = True,
    tools = [search_tool],
    llm = llm  # Pass the LLM instance to the agent to specify which model to use
)

task1 = Task(
    description  = "Tell me about the next premiere league football match of LIVERPOOL, formatted in a nice manner in IST",
    expected_output = "Date: 01/01/2025, Time: 7:30 pm IST",
    agent = football_agent
)

crew = Crew(
    agents = [football_agent],
    tasks =  [task1],
    verbose = True
)

result = crew.kickoff()

print("###########################")
print("Here is the result ")
print(result)
