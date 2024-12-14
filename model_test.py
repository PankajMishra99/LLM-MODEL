import openai
import streamlit as st
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import fitz

# Convert PDF to text
def pdf_to_txt_file(pdf_path):
    text = ''
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

# Load PDF and convert it to text
pdf_path = r'C:\Users\Pankaj Mishra\Downloads\basicElectronics.pdf'
pdf_text = pdf_to_txt_file(pdf_path)

# Initialize the embedding model
embeddings = OllamaEmbeddings()

# Split the text into chunks if needed (for better performance with large documents)
pdf_chunks = pdf_text.split("\n\n")

# Create a vector store using FAISS
vector_store = FAISS.from_texts(texts=pdf_chunks, embedding=embeddings)

# Set up retrieval from vector store
retriever = vector_store.as_retriever()

# Initialize Ollama LLM
ollama_llm = Ollama(model='llama2')

# Define a PromptTemplate with both 'question' and 'context'
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Context: {context}\n\nQ: {question}\nA: "
)

# Set up the QA chain with the correct prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Streamlit app setup
st.title("Interactive PDF Question-Answering System")
st.write("Ask questions about the content in the PDF document.")

# Text input for the user question
user_question = st.text_input("Enter your question:")

# If a question is entered, run the QA chain and display the answer
if user_question:
    answer = qa_chain.run(user_question)
    st.write("Answer:", answer)
    
