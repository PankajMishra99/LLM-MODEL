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
from pathlib import Path
from werkzeug.utils import secure_filename
import logging

import os
from langchain_community.document_loaders import TextLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["LANGCHAIN_TRACING_V2"]="True"

pytesseract.tesseract_cmd = r"C:\Users\Pankaj Mishra\Desktop\llm_model\LLM-MODEL\Tesseract-OCR\tesseract.exe"
# from file_flask import user_collection,client,register,login

with open("param.json",'r') as file:
    config =json.load(file)

config_model_name = config["llm"]["model_name"]
config_embedding = config["llm"]["embedding_model"]
config_temperature = config["llm"]["model_parameters"]["temperature"]
config_token = config["llm"]["model_parameters"]["max_tokens"]
config_top_p = config["llm"]["model_parameters"]["top_p"]
config_frequency = config["llm"]["model_parameters"]["frequency_penalty"]
config_presence = config["llm"]["model_parameters"]["presence_penalty"]
config_batch_size = config["llm"]["model_parameters"]["batch_size"]

config_presence = config["llm"]["model_parameters"]["presence_penalty"]


config_model = config["llm"]["model"]
config_neareset = config["llm"]["nearest_vector"]

config_parameter = config["llm"]["model_parameters"]




flask_available=config.get("frontend","").lower()=="flask"
app = Flask(__name__) 

app.config['UPLOAD_FOLDER'] =os.path.join(os.getcwd(), "folder")
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'docx'}


# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



#read the input from pdf and change in text.
def pdf_to_text(filename):
    text = ''
    with fitz.open(filename) as pdf:
        for page_num in range(pdf.page_count):
            page_text = pdf.load_page(page_num)
            page_text = page_text.get_text()
            # pix = page.get_pixmap()  # Render page to an image
            # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # page_text = pytesseract.image_to_string(img)
            text += page_text.strip()
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
        if file_type == '.pdf':
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
    


def path_fun():
    curret_dir = Path.cwd()
    dynamic_path = curret_dir/"folder"/"basicElectronics.pdf"
    return dynamic_path

text = path_fun()
text = main_text(text,'.pdf')
# print(main_text(text,'pdf'))



def clean_text(text):
    try:
        words = [word for word in text.split()]
        return ' '.join(words)
    except Exception as e:
        raise ValueError(f"Error during text cleaning: {e}")
    # return "Unable to process the file"

# print(pdf_to_text(file_path))
    
def chunk_text(text,chunk_size=512):
    # chunk_size= int(len(text))
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
##Need to checkt the chynk text or clean text for inegrate the main_text()



from transformers import GPT2Tokenizer, GPT2Model
import numpy as np

def load_model(model_name='gpt2'):
    """
    Load the pre-trained GPT-2 model and tokenizer.

    Args:
    - model_name (str): The name or path to the pre-trained GPT-2 model.

    Returns:
    - model (GPT2Model): Loaded GPT-2 model.
    - tokenizer (GPT2Tokenizer): Loaded GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the eos_token
    model = GPT2Model.from_pretrained(model_name)
    return model, tokenizer

# print(load_model())

def get_embedding(text, model_name='gpt2'):
    """
    Generate the embedding for the input text using GPT-2.

    Args:
    - text (str): Input text to embed.
    - model_name (str): The name or path to the pre-trained GPT-2 model.

    Returns:
    - sentence_embedding (numpy.ndarray): Embedding of the input text.
    """
    model, tokenizer = load_model(model_name)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Forward pass through GPT-2
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
    
    attention_mask = inputs['attention_mask']  # Shape: [batch_size, seq_length]
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    summed_embeddings = torch.sum(last_hidden_state * expanded_mask, dim=1)
    mask_sum = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)  # Avoid division by zero
    sentence_embedding = (summed_embeddings / mask_sum).squeeze().numpy()

    return sentence_embedding

# print(get_embedding('Name of prime minister of india??'))


def index_vector(texts):
    """
    Index a list of texts into a FAISS index.
    
    Args:
    - texts (list): A list of texts to index.
    
    Returns:
    - index (faiss.Index): FAISS index containing the embeddings.
    - docstore (dict): A dictionary mapping document indices to text.
    """
    if not texts:
        raise ValueError("Text Input for indexing is empty..")

    vectors = [get_embedding(chunk) for chunk in texts]
    vectors = np.array(vectors)
    
    d = len(vectors[0])  # Get the dimension of the embedding vector
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    
    docstore = {i: {"document": texts[i]} for i in range(len(texts))}
    
    return index, docstore

# print(index_vector('Name of prime minister of india??'))



def retrieve_from_faiss(query, index, k=5):
    """
    Retrieve the nearest neighbors for a query from the FAISS index.
    
    Args:
    - query (str): The query to search for.
    - index (faiss.Index): The FAISS index to search within.
    - k (int): The number of nearest neighbors to retrieve.
    
    Returns:
    - indices (numpy.ndarray): The indices of the nearest neighbors.
    - distances (numpy.ndarray): The distances to the nearest neighbors.
    """
    query_vector = np.array(get_embedding(query)).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return indices, distances


def update_context_based_on_feedback(previous_context, question, feedback):
    """
    Update the context based on user feedback or previous interactions.
    
    Args:
    - previous_context (str): The previous context provided to the model.
    - question (str): The current question.
    - feedback (str): User's feedback or review on the answer.
    
    Returns:
    - updated_context (str): The new context incorporating feedback.
    """
    updated_context = previous_context + "\nFeedback: " + feedback + "\nNew Question: " + question
    return updated_context


def llm_model(question, model_type, context=None, temperature=0.3, feedback=None):
    """
    Function to handle question answering using an LLM with or without context.
    
    Args:
    - question (str): The question to be answered.
    - model_type (str): The model type for LLM.
    - context (str, optional): Context to assist in answering. Default is None.
    - temperature (float): Temperature for sampling. Default is 0.7.
    - feedback (str, optional): Feedback or review to adjust the response. Default is None.
    
    Returns:
    - response (str): The answer to the question.
    """
    try:
        llm = OllamaLLM(
            model=model_type,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.0,
            max_new_tokens=50,
            config_presence=0.5
        )

        if context:  # Context provided
            if feedback:
                # Update the context based on the feedback and question
                context = update_context_based_on_feedback(context, question, feedback)

            # Index and retrieve the context
            index, docstore = index_vector([context])
            vectorstore = FAISS.from_texts(texts=[context], embedding=OllamaEmbeddings(model=model_type))
            retriever = vectorstore.as_retriever()

            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="Context: {context}\n\nQ: {question}\nA: "
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )

            response = qa_chain.invoke({"query": question})
            return response

        else:  # No context provided, directly answer the question
            index, docstore = index_vector([question])
            vectorstore_question = FAISS.from_texts(texts=[question], embedding=OllamaEmbeddings(model=model_type))
            retriever_question = vectorstore_question.as_retriever()

            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="Context: {context}\n\nQ: {question}\nA: "
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever_question,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )


            response_question = qa_chain.invoke({"query": question})
            return response_question.get("result", "No answer found")

    except Exception as e:
        logging.error(f"Error in LLM model: {e}")
        raise ValueError(f"Error in LLM model: {e}")
    
# print(llm_model("name of c.m of M.P", "llama3.2"))


def qa_chain_function(question, model_type, texts=None):
    """
    Handles QA for both general questions and file-based input.
    
    Args:
        question (str): The question to be answered.
        model_type (str): The model to use for LLM processing.
        texts (str or None): Text extracted from an uploaded file, if provided.
    
    Returns:
        str: The answer or summary based on the input.
    """
    try:
        # Case 1: No file uploaded, respond to general question
        if not texts:
            print("No file uploaded. Answering general question.")
            # Answer the question based on the model, passing an empty string as context
            response = llm_model(question, model_type)
            if isinstance(response, dict):
                return response.get("result", "I couldn't find an answer to your question.")
            else:
                return response

        # Case 2: File uploaded, process its content
        print("File uploaded. Processing text from file.")
        
        # Ensure texts (from the uploaded file) is valid and chunk it
        if not isinstance(texts, str) or not texts.strip():
            return "Uploaded file does not contain valid text for processing."
        
        # Chunk the extracted text from the file
        chunks = chunk_text(texts)
        if not chunks:
            return "Unable to chunk text from the uploaded file."
        
        # Create vector index for chunks
        index = index_vector(chunks)
        print(f"Index created with {len(chunks)} chunks.")
        
        # Retrieve relevant chunks using FAISS or other retrieval methods
        I, D = retrieve_from_faiss(question, index)
        if len(I[0]) == 0:
            return "Sorry, I couldn't find any relevant information in the uploaded file."
        
        relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        if not relevant_chunks:
            return "No relevant chunks found in the uploaded file."
        
        # Process relevant chunks through the model to generate answers
        answer = []
        for chunk in relevant_chunks:
            response = llm_model(question, model_type, str(chunk))
            if isinstance(response, dict):
                result = response.get("result", "")
                if result:
                    answer.append(result)
                else:
                    print("Empty result in response for a chunk.")
            else:
                print("Unexpected response format for a chunk.")
        
        # Deduplicate and join answers
        final_answers = list(set(answer))
        return "\n".join(final_answers).strip() or "No relevant information found in the uploaded file."

    except Exception as e:
        return f"Error in QA chain: {e}"







# print(qa_chain_function("what is high pass filter in given text??", "llama3.2", text))

# # print(text)

@app.route("/chatbot", methods=["GET", "POST"])
def flask_chat():
    answer = ""
    if request.method == "POST":
        try:
            # Get the uploaded file and question
            file = request.files.get("file")
            question = request.form.get("question")

            # Ensure that the file and question are provided
            if not file or not question:
                return "Please upload a file and ask a question."
            


            # Save the file locally (make sure the filename is unique to avoid overwriting)
            file_path = f"uploads/{file.filename}"  # You can create an 'uploads' folder
            file.save(file_path)

            # Extract text from the file
            extract_text = pdf_to_text(file_path)  # Assuming pdf_to_text works with a file path
            if not extract_text:
                return "No text found in the PDF file."

            # Clean the extracted text (assuming clean_text handles unnecessary formatting)
            text1 = clean_text(extract_text)
            print(f"Extracted Text: {text1[:100]}...")  # Display the first 100 characters for debugging

            # Call qa_chain_function with the cleaned text and question
            answer = qa_chain_function(question, 'llama3.2', text1)

        except Exception as e:
            print(f"Error during processing: {e}")
            answer = "An error occurred. Please try again later."

    return render_template("chatbot.html", answer=answer)




# # # # for web interfacing..
# # def web_chat():
# #     st.title("AI to Question Answer Chatbot..")

# #     file_uploader = st.file_uploader("File_uploader",type=["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
# #     file_type = st.selectbox("File Type",["pdf", "png", "jpg", "wav", "mp4", "txt", "docx"])
    
# #     #if file_uploader:
# #     extracted_text = main_text(file_uploader, file_type)
# #     question  = st.text_input("Please enter the question: ")
# #        # if question:
# #     answer = qa_chain_function(question,extracted_text)
# #     st.write("Answer : ",answer)
# #     return answer

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Run Chatbot using Flask or Streamlit.")
# #     parser.add_argument("--interface",
# #         type=str,
# #         default="flask",
# #         choices=["flask",'streamlit']
# #     )
# #     args = parser.parse_args()
# #     if args.interface=="flask":
# #         app.run(debug=True)
# #     elif  args.interface=="streamlit":
# #         web_chat()



# # if __name__=="__main__":
# #     # app.run(debug=True)
# #     print(qa_chain_function("Name of Prime Minister of India.","llama3"," "))







    








