# Workflow of Language models
Chatgpt and other large language models doesn't work well with large chuncks of text, that is why when interacting with language models we divide the data into small chunks and then get a summary of each chunk (create embeddings for each chunk). This summary is then stored in database (vector store), when the user asks question, the question is summarized and the database is searched to get the most relevant chunks wrt the question. It is then the relevant chunks and the question is grouped together and sent to the language model to get a response.

**Embeddings** - Embeddings are mathematical scores of a chunk of data based on various parameters.

**Vector Database** - These are databases specialized in storing embeddings. These not just store data but can also help us doing some mathematical operations like semantic search on it.

**Langchain** - It is a framework which provides us the tools to execute the entire workflow of working with Language models.

### Basic OpenAI program with Langchain
```
from langchain_openai import OpenAI

# Set your OpenAI API key
api_key = OPENAI_API_KEY

# Initialize the OpenAI LLM with the API key
llm = OpenAI(
    openai_api_key=api_key
)

# Generate a response
result = llm.invoke("Write a very very short poem")
print(result)
```

**Chains** are langchain class which has two parts - a prompt template and a language model. We pass input to a chain (which is added to the prompt template) and an output is returned. We can connect different chains together and use output of one as input to another.

**Input ----> [ prompt template + language model ] ----> Output**

Input is a dictionary and should contain the values specifies in the prompt template. Output is also a dictionary containing the key-value pairs of the input and the generated output with the default key as text (it can be customized)


### Basic OpenAI program with Langchain using Chains
```
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your OpenAI API key
api_key = OPENAI_API_KEY

# Initialize the OpenAI LLM with the API key
llm = OpenAI(
    openai_api_key=api_key
)

# Initializing our prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Initializing our chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# Generate a response
result = code_chain({
    "language": "python",
    "task": "return a list of numbers"
})

print(result["text"])
```

### Basic OpenAI program with Langchain using chains (getting inputs via terminal)
```
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

# Get input values via terminal
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Set your OpenAI API key
api_key = OPENAI_API_KEY

# Initialize the OpenAI LLM with the API key
llm = OpenAI(
    openai_api_key=api_key
)

# Initializing our prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Initializing our chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# Generate a response
result = code_chain({
    "language": args.language,
    "task": args.task
})

print(result["text"])
```
**NOTE:** The -- prefix is used to create optional arguments, which are generally more flexible and user-friendly for scripts that may require multiple arguments in different orders. Positional arguments are simpler but require a specific order

### Basic OpenAI program with Langchain using chains (getting inputs via terminal)
**Using .env file to get OPENAI_API_KEY env variable**
```
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Initializing our chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# Generate a response
result = code_chain({
    "language": args.language,
    "task": args.task
})

print(result["text"])
```
**NOTE:** .env files are never commited to github

### Langchain code for two chains connected together and acting as a unit
**The two chains are connected together using SequentialChain function**
```
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
```
