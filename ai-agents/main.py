from phi.agent.agent import Agent
from phi.model.ollama import Ollama
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.crawl4ai_tools import Crawl4aiTools



# agent = Agent(provider=OpenAI(model="gpt-4o-mini"), tools = [YFinanceTools(stock_fundamentals=True, income_statements=True, key_financial_ratios=True, analyst_recommendations=True)], show_tool_calls=True, markdown=True, instructions=["Use tables to display the data"]) #type: ignore



# web_agent  = Agent(provider=Ollama(model="llama3.1:latest"), tools = [DuckDuckGo()], show_tool_calls=True, markdown=True, instructions=["always include sources"]) #type: ignore


# team_agent = Agent(team = [web_agent, agent], show_tool_calls=True, markdown=True, instructions=["always include sources", "always use tables to display the data"]) #type: ignore


# team_agent.print_response("Summarise and compare the analyst recommendations and stock fundamentals of Apple and Google stocks with the latest news")


# football_agent = Agent(provider = OpenAIChat(model = "gpt-4o-mini"), tools = [GoogleSearch()], show_tool_calls=True, markdown=True, instructions=["only return the date and time of the event in indian standard time in a well formatted manner", "do not return any other information"]) #type: ignore

football_agent = Agent(provider = OpenAIChat(model = "gpt-4o-mini"), tools = [Crawl4aiTools()], show_tool_calls=True, markdown=True, instructions=["only return the date and time of the event in indian standard time in a well formatted manner", "do not return any other information"]) #type: ignore

football_agent.print_response("Tell me when is the next premier league football game of Manchester City")
