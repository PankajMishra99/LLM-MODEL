import langchain
from langchain.llms import ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import faiss
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

#read the input from pdf and change in text.
def pdf_to_text(filename='source_pdf'):
    text=''
    with fitz.open(filename) as file:
        for page_count in range(file.page_count):
            page = file.load_page(page_count)
            text +=page.get_text()
        return text
def image_to_text(filename='image_source'):
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    return text
def audio_text(filename_audio):
    recognizer =sr.Recognizer()
    with sr.AudioFile(filename_audio) as file:
        audio = recognizer.record(file)
    text = recognizer.recognize_google(audio)
    return text

def video_text(filename_video):
    video=VideoFileClip(filename_video)
    audio_path = '.wav'
    video.audio.write_audiofile(audio_path)
    text= audio_text(audio_path)
    return text
def extract_text(text):
    return text

def docs_text(filename_docs):
    text =''
    doc = Document(filename_docs)
    for para in doc.paragraphs:
        text += para + '\n'
    return text
def main_text(file_path, file_type):
    while True:
        if file_type=='pdf':
            return pdf_to_text(file_path)
        elif file_type=='image':
            return image_to_text(file_path)
        elif file_type =='audio':
            return audio_text(file_path)
        elif file_type=='video':
            return video_text(file_path)
        elif file_type=='docs':
            return docs_text(file_path)
        elif file_type=='file_path':
            return extract_text(file_path)
        else:
            raise ValueError ("Unsupported file type..")
        
def clean_text(text):
    text=text.lower()
    words=re.sub(r'[^\w\s]','',text)
    stop_words =stopwords.words('english')
    words =[word for word in words if word not in stop_words]
    text =''.join(words)
    return text

    
def chunk_text(text, chunk_size = 500):
    text= clean_text(text)
    words = text.split()
    for i in range (0, len(words),chunk_size):
        yield ' '.join(words[i:i + chunk_size])

def model_name(text,model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model= AutoModel.from_pretrained(model_name)
    tokenizer= AutoTokenizer.from_pretrained(model_name)
    input = tokenizer(text,return_tensors='pt',truncation=True,padding=True)
    return {'model':model,'tokenizer':tokenizer,'input':input}


def get_embedding(text):
    model = model_name['model']
    input = model['input']
    with torch.no_grad():
        emedding = model(**input)
        return emedding.np()


def index_vector(text):
    d=384
    index = faiss.IndexFlatL2(d)
    vector = [get_embedding(chunk) for chunk in text]
    index.add(np.array(vector))
    return index


def retrive_from_faiss(query, index):
    query_vector = get_embedding(query)
    D, I = index.search(np.array([query_vector]),k=5)
    return I,D

#llm model
def llm_model(question, context):
    model = ollama(model='llama2')
    prompt_template = PromptTemplate(
        input_variables=[question,context],
         template="Context: {context}\n\nQ: {question}\nA: "
    )
    qa_chain = RetrievalQA.from_chain_type(
        model_name=model,
        chain_type = 'stuff',
        chain_type_kwargs = {'prompt':prompt_template}
    )
    return qa_chain

def qa_chain_function(question,text):
    chunks = list(chunk_text(text))
    index = index_vector(chunks)
    I,D = retrive_from_faiss(question,index)
    reletive_chunk = [chunks[i] for i in I[0]]
    answer=""
    for chunk in reletive_chunk:
        answer = llm_model(question,chunk)
    if answer:
        answer
    return answer

# for web interfacing..
def web_chat():
    st.title("Welcome to AI to Question Answer Chatbot..")

    file_uploader = st.file_uploader("File_uploader",type=["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
    file_type = st.selectbox("File Type",["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
    
    #if file_uploader:
    extracted_text = main_text(file_uploader, file_type)
    question  = st.text_input("Please enter the question: ")
       # if question:
    answer = qa_chain_function(question,extracted_text)
    st.write("Answer : ",answer)
    #return answer

if __name__=="__main__":
    web_chat()





    








