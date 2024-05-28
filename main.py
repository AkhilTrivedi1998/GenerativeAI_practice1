from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# finds .env file and loads all the environment variable
# Note that .env files are not commited to github
# openai looks for specific env variable named "OPENAI_API_KEY"
load_dotenv() 

# Get input values via terminal
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Initialize the OpenAI LLM with the API key
# Don't need to pass openai_api_key manually, it will look for env variable named "OPENAI_API_KEY" and since it is already loaded, it will work fine
llm = OpenAI()

# Initializing our prompt template
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
)

# Initializing second prompt template fro test
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a unit test code for the following {language} code: {code}"
)

# Initializing our chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code" # sets the key value of the output returned as 'code' (default is 'text')
)

# Initializing second chain for test
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# Connecting the two chains together (they are gonna act as a single unit)
chain = SequentialChain(
    chains=[code_chain, test_chain], # order is important here
    input_variables=["language", "task"],
    output_variables=["test", "code"]
)

# Generate a response
result = chain({
    "language": args.language,
    "task": args.task
})

print(result['code'])
print(result['test'])
