# LANGCHAIN_API_KEY = "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"
# LANGCHAIN_PROJECT = "LANGCHAIN_1"


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_community.llms.ollama import Ollama
import os
from dotenv import load_dotenv

load_dotenv()
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# for langsmith tracing
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"]='True'

# creating chatbot
system_template = SystemMessagePromptTemplate.from_template("you are a helpful assistant, please provide response to the user input")

# Define the question prompt as a human message template
question_template = HumanMessagePromptTemplate.from_template("question: {question}")

# Combine both templates into a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([system_template, question_template])



# for output..
st.title("Langchain demo with LLAMA Model.")
input_text = st.text_input("Search the input you want..")


#open AI LLM Model..
llm=Ollama(model_name="llama2")
output_parser= StrOutputParser()

#chain
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invok({'question':input_text}))
