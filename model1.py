import langchain_community
from langchain_ollama import OllamaLLM

from langchain_ollama import OllamaEmbeddings


from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import fitz
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import pytesseract
import speech_recognition  as sr
from docx import Document
import re
from nltk.corpus import stopwords
from transformers import AutoModel,AutoTokenizer
import faiss
import torch
import numpy as np
import json
import flask
from flask import Flask,render_template,url_for,request,Response,session,redirect,flash
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.vectorstores import InMemoryVectorStore
import argparse

import os
from langchain_community.document_loaders import TextLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["LANGCHAIN_TRACING_V2"]="True"

file_path =r"C:\Users\Pankaj Mishra\Downloads\basicElectronics.pdf"

# from file_flask import user_collection,client,register,login

with open("param.json",'r') as file:
    config =json.load(file)

flask_available=config.get("frontend","").lower()=="flask"
app=Flask(__name__) if flask_available else None

#read the input from pdf and change in text.
def pdf_to_text(filename):
    text = ''
    with fitz.open(filename) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()  # Render page to an image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text += page_text
    return text
def image_to_text(filename):
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    return text

def audio_text(filename):
    recognizer =sr.Recognizer()
    with sr.AudioFile(filename) as file:
        audio = recognizer.record(file)
    text = recognizer.recognize_google(audio)
    return text
from tempfile import NamedTemporaryFile
def video_text(filename):
    video = VideoFileClip(filename)
    with NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        video.audio.write_audiofile(temp_audio.name)
        text = audio_text(temp_audio.name)
    return text

def extract_text(text):
    return text

def docs_text(filename):
    text = ''
    try:
        doc = Document(filename)
        for para in doc.paragraphs:
            text += para.text + '\n'  # Fix this line; `.text` is needed.
    except Exception as e:
        raise ValueError(f"Error processing Word document: {e}")
    return text



def main_text(file, file_type):
    try:
        if file_type == 'pdf':
            return pdf_to_text(file)
        elif file_type in ['png', 'jpg', 'jpeg']:
            return image_to_text(file)
        elif file_type == 'audio':
            return audio_text(file)
        elif file_type == 'video':
            return video_text(file)
        elif file_type == 'docs':
            return docs_text(file)
        elif file_type == 'text':
            return extract_text(file)  # Handle text input
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

        
def clean_text(file, file_type=["pdf", "png", "jpg",'jpeg', "wav", "mp4", "txt", "docx"]):
    # for file_type in file_types:
        try:
            text = main_text(file, file_type)
            if text:
                words=[word for word in text.split()]
                cleaned_text = ' '.join(words)
                return cleaned_text
        except Exception as e:
            print(f"Failed to process {file_type}:{e}")
    # return "Unable to process the file"

# print(pdf_to_text(file_path))
    
def chunk_text(texts, file=file_path, file_type=["pdf", "png", "jpg",'jpeg', "wav", "mp4", "txt", "docx"]):
    chunks = []
    for text in texts:
        text = clean_text(file)  
        chunks.append(text) 
        return chunks
##Need to checkt the chynk text or clean text for inegrate the main_text()

def model_name(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model= AutoModel.from_pretrained(model_name)
    tokenizer= AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer


def get_embedding(text):
    embeddings = OllamaEmbeddings(model="llama3",)
    embedding = embeddings.embed_query(text)
    return embedding

# print(get_embedding("what is the Voltage"))

def index_vector(texts):
    if not texts:
        raise ValueError ("Text Input for indexing empty..")

    vectors = [get_embedding(chunk) for chunk in texts]
    vectors = np.array(vectors)
    
    # Initialize FAISS index with the embedding dimension
    d = len(vectors[0])  # Get the dimension of the embedding vector
    index = faiss.IndexFlatL2(d)  # L2 distance for nearest neighbor search
    
    # Add the embeddings to the FAISS index
    index.add(vectors)
    
    # Create a document store that maps document IDs to actual texts
    docstore = {i: {"document": texts[i]} for i in range(len(texts))}
    
    # Create a function to map FAISS index IDs to docstore IDs
    
    # Use the FAISS index and docstore
    return index

# print(print(index_vector("What is the voltage")))

def retrieve_from_faiss(query,index, k=5):
    query_vector = np.array(get_embedding(query)).reshape(1,-1)
    distances, indices = index.search(query_vector,k)
    return  indices,distances

# test the code..
# index = index_vector("Name of Prime Minister of India.")
# print(retrieve_from_faiss("Name of Prime Minister of India.",index,k=5))

# #llm model , need to initlized the model_type in the json format.


def llm_model(question, model_type):
    # Initialize the LLM model
    llm = OllamaLLM(model=model_type)

    # Define the embedding function
    embeddings = OllamaEmbeddings(model=model_type)

    docs = InMemoryVectorStore.from_texts(
        [question],  
        embedding=embeddings
    )
    text_data=[question]
    # Set up FAISS vector store
    vectorstore = FAISS.from_texts(texts=text_data,embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="Context: {context}\n\nQ: {question}\nA: ",
    )

    # Initialize RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
    )

    # Query the chain
    response = qa_chain.invoke(question)
    # response=response['question']
    return response


# Test the model
# print(llm_model("Name of Prime Minister of India.","llama3"))




def qa_chain_function(question, model_type, text):
    try:
        # model, tokenizer = model_name()
        # text=[question]
        chunks = chunk_text(text)
        index = index_vector(chunks)
        I, D = retrieve_from_faiss(question, index)
        if len(I[0]) == 0:
            print("Sorry, I couldn't find any relevant information.")
        relevant_chunks = [chunks[i] for i in I[0] if i <len(chunks)]

        if not relevant_chunks:
            print("No relevant chunks found.")
        
        answer = []
        for chunk in relevant_chunks:
            response = llm_model(str(chunk), model_type)  # Append the answers
            # print("Response from llm_model:", response) 
            if isinstance(response,dict):
                result = response.get("result", "")
                if result:
                    answer.append(result)
                else:
                    raise ValueError("Empty 'result' key in response.")
            else:
                raise ValueError("Response is not a dictionary.")

        return "\n".join(answer).strip()
    except Exception as e:
        raise ValueError(f"Error in QA chain: {e}")

# print(qa_chain_function("Name of Prime Minister of India.","llama3"))


@app.route("/", methods=["GET", "POST"])
def flask_chat():
    answer = ""
    if request.method == "POST":
        try:
            file = request.files.get("file")
            file_type = request.form.get("file_type")
            question = request.form.get("question")

            # Process the file and generate an answer
            if file and question:
                file.save(file.filename)
                file_path =file.filename
                extract_text = main_text(file_path,file_type)
                text =clean_text(extract_text)
                answer = qa_chain_function(question, 'llama3',text)
                return answer
        except Exception as e:
            print(f"Error during processing: {e}")
            answer = "An error occurred. Please try again later."

#     return render_template("chatbot.html", answer=answer)
# if __name__=="__main__":
#      flask_chat()



# # for web interfacing..
# def web_chat():
#     st.title("AI to Question Answer Chatbot..")

#     file_uploader = st.file_uploader("File_uploader",type=["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
#     file_type = st.selectbox("File Type",["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
    
#     #if file_uploader:
#     extracted_text = main_text(file_uploader, file_type)
#     question  = st.text_input("Please enter the question: ")
#        # if question:
#     answer = qa_chain_function(question,extracted_text)
#     st.write("Answer : ",answer)
#     return answer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run Chatbot using Flask or Streamlit.")
#     parser.add_argument("--interface",
#         type=str,
#         default="flask",
#         choices=["flask",'streamlit']
#     )
#     args = parser.parse_args()
#     if args.interface=="flask":
#         app.run(debug=True)
#     elif  args.interface=="streamlit":
#         web_chat()


# if __name__=="__main__":
#     # get_embedding("what is the voltage",'')
#     llm_model("what is the voltage","llama2")







    








