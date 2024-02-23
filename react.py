from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY=os.environ.get("OPENAI_API_KEY")

# tools = [TavilySearchResults(max_results=1)]
tools = [DuckDuckGoSearchRun()]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = OpenAI() # default model "gpt-3.5-turbo-instruct" up to Sep 2021

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "what is LangChain?"})) # Initial release October 2022