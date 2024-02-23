import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import Tool
import matplotlib
import numpy
# from io import StringIO
from dotenv import load_dotenv
import os

matplotlib.use('TkAgg')

load_dotenv()
API_KEY=os.environ.get("OPENAI_API_KEY")


def main():
    st.set_page_config(page_title="Ask your CSV ")
    st.header("Ask your CSV ")

    user_csv = st.file_uploader("Upload your CSV file", type=["csv"])

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your CSV")
        # stringio = StringIO(user_csv.getvalue().decode("utf-8"))
        # st.write(stringio)

        pandas_kwargs={'on_bad_lines': 'skip'}
        # Create a csv generating pandas dataframe agent with the GPT-3.5-turbo API model
        agent = create_csv_agent(
            ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0125"),
            # Ollama(base_url='http://localhost:11434', model="llama2"),
            path=user_csv,
            verbose=True,
            pandas_kwargs=pandas_kwargs,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        if user_question is not None and user_question != "":
            with st.spinner("Loading answer(s) from the LLM..."):
                response = agent.invoke(user_question)
                st.write(response)


if __name__ == "__main__":
    main()


# Can you group the records by the size run id attribute and give me the summation of each group and plot a graph based on the result?