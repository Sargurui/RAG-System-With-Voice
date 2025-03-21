"""
This Flask application provides a platform for uploading, processing, and querying various document types 
(PDF, DOCX, TXT, JSON, CSV) as well as integrating YouTube transcription and web scraping functionalities. 
It uses LangChain for document processing and retrieval, and Azure OpenAI for generating responses to user queries.

Key Features:
- Upload and manage files in various formats.
- Process documents to create retrievers for querying.
- Integrate YouTube video transcription and translation.
- Perform web scraping to extract content from websites.
- Query documents and general questions using Azure OpenAI.
"""

import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader,UnstructuredFileLoader
from youtube import YouTubeProcessor
from webscrape import WebScraper
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import json
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI API Configuration
# os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
# os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
# os.environ['OPENAI_API_VERSION'] = os.getenv('OPENAI_API_VERSION')



# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")



# Use a free embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_document(file_path, file_type):
    """
    Processes a document based on its file type and creates a retriever for querying.

    Args:
        file_path (str): The path to the document file.
        file_type (str): The type of the document (e.g., pdf, docx, txt, json, csv).

    Returns:
        retriever: A retriever object for querying the document.
    """
    if file_type == "pdf":
        loader = UnstructuredPDFLoader(file_path)
    elif file_type == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_type == "txt":
        loader = UnstructuredFileLoader(file_path)
    elif file_type == "json":
        loader = JSONLoader(file_path=file_path, jq_schema=".[] | .content", text_content=False)
    elif file_type == "csv":
        loader = CSVLoader(file_path=file_path)
    docs = loader.load_and_split()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()


file_retrievers = {}


def get_uploaded_files():
    """
    Retrieves a list of all uploaded files categorized by their file types.

    Returns:
        tuple: Lists of files categorized as PDF, DOCX, TXT, JSON, and CSV.
    """
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    docx_files = [f for f in files if f.lower().endswith('.docx')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]
    json_files = [f for f in files if f.lower().endswith('.json')]
    csv_files = [f for f in files if f.lower().endswith('.csv')]
    return pdf_files, docx_files, txt_files, json_files, csv_files

@app.route("/")
def home():
    """
    Redirects the user to the chat page.
    """
    return redirect(url_for("chat_page"))

@app.route("/chat")
def chat_page():
    """
    Renders the chat page with a list of uploaded files.
    """
    pdf_files, docx_files, txt_files, json_files, csv_files = get_uploaded_files()
    return render_template("chat.html", pdf_files=pdf_files, docx_files=docx_files, txt_files=txt_files, json_files=json_files, csv_files=csv_files)

@app.route("/chat_general")
def chat_general_page():
    """
    Renders the general chat page with a list of uploaded files.
    """
    pdf_files, docx_files, txt_files, json_files, csv_files = get_uploaded_files()
    return render_template("chat_general.html", pdf_files=pdf_files, docx_files=docx_files, txt_files=txt_files, json_files=json_files, csv_files=csv_files)


@app.route("/delete_file", methods=["POST"])
def delete_file():
    """
    Deletes a specified file from the uploads folder.

    Returns:
        JSON response indicating success or failure.
    """
    data = request.get_json()
    filename = data.get("filename")
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "File not found"})

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    """
    Handles file uploads and YouTube URL processing.

    Returns:
        JSON response or renders the upload page.
    """
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "message": "No file selected."})
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)                
                return jsonify({"success": True, "message": "File uploaded successfully."})
                
        elif "youtube_url" in request.form:
            youtube_url = request.form["youtube_url"]
            processor = YouTubeProcessor(youtube_url)
            processor.process()
            return jsonify({"success": True, "message": "YouTube URL processed successfully."})
            
        

    pdf_files, docx_files, txt_files, json_files, csv_files = get_uploaded_files()
    return render_template("upload.html", pdf_files=pdf_files, docx_files=docx_files, txt_files=txt_files, json_files=json_files, csv_files=csv_files)

@app.route("/webscrape", methods=["POST"])
def webscrape():
    """
    Performs web scraping for a given website URL.

    Returns:
        JSON response indicating success or failure.
    """
    website_url = request.form["website_url"]
    scraper = WebScraper(base_url=website_url, max_pages=20)
    try:
        scraper.scrape()
        return jsonify({"success": True, "message": "Scraping completed successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/query", methods=["POST"])
def query():
    """
    Handles user queries for a selected document.

    Returns:
        JSON response with the query answer.
    """
    selected_file = request.form["selected_file"]

    pdf_files, docx_files, txt_files, json_files, csv_files = get_uploaded_files()
    all_files = pdf_files + docx_files + txt_files + json_files + csv_files

    if selected_file not in all_files:
        return jsonify({"answer": "Please select a valid document."})

    file_path = os.path.join(UPLOAD_FOLDER, selected_file)
    if selected_file not in file_retrievers:
        if selected_file.lower().endswith('.pdf'):
            file_type = "pdf"
        elif selected_file.lower().endswith('.docx'):
            file_type = "docx"
        elif selected_file.lower().endswith('.txt'):
            file_type = "txt"
        elif selected_file.lower().endswith('.json'):
            file_type = "json"
        elif selected_file.lower().endswith('.csv'):
            file_type = "csv"
        file_retrievers[selected_file] = process_document(file_path, file_type)

    retriever = file_retrievers[selected_file]
    
    user_query = request.form["msg"]
    
#     llm = HuggingFaceHub(
#     repo_id="meta-llama/Llama-3.3-70B-Instruct",  # Remote model
#     model_kwargs={"temperature": 0.7, "max_length": 150}
# )

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7)

    # llm = AzureChatOpenAI(deployment_name="GPT-4o")

    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant. Given the context and question, provide a detailed and accurate answer.\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Only provide the answer based on the context. If the context does not contain the answer, please explicitly state: 'The answer is not available in this file.'"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_query)

    return jsonify({"answer": response})

@app.route("/query_general", methods=["POST"])
def query_general():
    """
    Handles general user queries without document context.

    Returns:
        JSON response with the query answer.
    """
    user_query = request.form["msg"]
    
    # llm = AzureChatOpenAI(deployment_name="GPT-4o")

#     llm = HuggingFaceHub(
#     repo_id="meta-llama/Llama-3.3-70B-Instruct",  # Remote model
#     model_kwargs={"temperature": 0.7, "max_length": 150}
# )


    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant. Given the question, provide a detailed and accurate answer.\n\n"
        "Question: {question}\n"
        "If you do not know the answer, please state that explicitly and do not use any external knowledge."
    )

    rag_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_query)
    
    return jsonify({"answer":response})

@app.route('/view_file', methods=['POST'])
def view_file():
    """
    Retrieves and returns the content of a specified file.

    Returns:
        JSON response with the file content or an error message.
    """
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename not provided'}), 400

    file_path = os.path.join('uploads', filename)  
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': 'File not found'}), 404

    with open(file_path, 'r') as file:
        content = file.read()

    return jsonify({'success': True, 'content': content})

if __name__ == "__main__":
    app.run(debug=False)
