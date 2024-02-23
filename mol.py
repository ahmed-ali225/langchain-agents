from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import random

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY=os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=API_KEY)

search = DuckDuckGoSearchRun()
# defining a single tool
# tools = [
#     Tool(
#         name = "search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events. You should ask targeted questions"
#     )
# ]

def meaning_of_life(input=""):
    return 'The meaning of life is 42 if rounded but is actually 42.17658'

life_tool = Tool(
    name='Meaning of Life',
    func= meaning_of_life,
    description="Useful for when you need to answer questions about the meaning of life. input should be MOL "
)


def random_num(input=""):
    return random.randint(0,5)

random_tool = Tool(
    name='Random number',
    func= random_num,
    description="Useful for when you need to get a random number. input should be 'random'"
)


# tools = [search, random_tool]
tools = [search, random_tool, life_tool]

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)


# create our agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)


# conversational_agent("What time is it in London?")
# conversational_agent("Can you give me a random number?")
conversational_agent("What is the meaning of life?")

# MOL LLM's answer: "The meaning of life is a philosophical question that has been debated for centuries. Some believe it is to seek happiness, others think it is to fulfill a purpose or destiny. Ultimately, the answer may vary depending on individual beliefs and perspectives."


#################################################
# fixed_prompt = '''Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

# conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

# conversational_agent("What is the meaning of life?")