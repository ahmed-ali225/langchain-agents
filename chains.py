from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY=os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# chain = prompt | llm 

# print(chain.invoke({"input": "how can langsmith help with testing?"}))

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"input": "how can langsmith help with testing?"}))


# Output chain 1
# content='Langsmith can help with testing in several ways:\n\n1. Automated Testing: Langsmith can be used to generate test cases automatically, saving time and effort in writing test scripts manually. This can help in increasing test coverage and identifying potential issues early in the development cycle.\n\n2. Test Data Generation: Langsmith can be used to generate realistic and diverse test data, which can be used to validate the functionality of the software under different scenarios. This can help in uncovering edge cases and improving the overall quality of the software.\n\n3. Performance Testing: Langsmith can be used to simulate a large number of users or transactions to test the performance of the software. This can help in identifying bottlenecks and optimizing the performance of the system.\n\n4. Security Testing: Langsmith can be used to generate security test cases to identify vulnerabilities in the software. By simulating different attack scenarios, Langsmith can help in improving the security posture of the software.\n\nOverall, Langsmith can be a valuable tool in the testing process by automating repetitive tasks, generating test data, and helping in identifying issues early in the development cycle.'

# Output chain 2
# Langsmith can help with testing in several ways:

# 1. Automated Testing: Langsmith can be used to generate test cases automatically, saving time and effort in writing test scripts manually. This can help in increasing test coverage and identifying potential issues early in the development cycle.

# 2. Test Data Generation: Langsmith can generate realistic and diverse test data, which can be used to validate the functionality of the software under different scenarios. This can help in uncovering edge cases and improving the robustness of the testing process.

# 3. Test Scenario Generation: Langsmith can create test scenarios based on the requirements and specifications of the software, ensuring that all possible paths and interactions are covered during testing. This can help in identifying potential bugs and ensuring the software meets the desired quality standards.

# 4. Regression Testing: Langsmith can be used to automate regression testing, allowing testers to quickly re-run test cases after code changes to ensure that existing functionality has not been affected. This can help in maintaining the stability and reliability of the software over time.

# Overall, Langsmith can streamline the testing process, improve test coverage, and enhance the quality of the software by providing automated testing capabilities and generating test artifacts efficiently.