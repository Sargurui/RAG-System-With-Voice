# RAG System With Voice 


![Screenshot](/images/img_1.png)
![Screenshot](/images/img_2.png)
![Screenshot](/images/img_3.png)
![Screenshot](/images/img_4.png)

This project is a Flask-based platform that integrates various functionalities such as document processing, YouTube transcription, web scraping, and querying using advanced language models. It is designed to handle multiple document types and provide intelligent responses to user queries.

## Key Features

- **File Upload and Management**: Upload and manage files in various formats (PDF, DOCX, TXT, JSON, CSV).
- **Document Processing**: Process documents to create retrievers for querying.
- **YouTube Integration**: Transcribe and translate YouTube video audio using Whisper and Google Translate. Provide the video URL for processing.
- **Web Scraping**: Extract text content from websites and save it in JSON format.
- **Querying**: Query documents or ask general questions using advanced language models like LLaMA or Azure OpenAI.
- **Interactive UI**: User-friendly interface for uploading files, querying, and managing content.
- **Voice Integration**: Convert voice input to text for querying or transcription using advanced speech-to-text models.
- **Text-to-Speech**: Convert text responses or documents into audio for playback.

## Setup Instructions

### Prerequisites

1. Python 3.8 or higher
2. Virtual environment (optional but recommended)
3. Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add the following variables:
     ```
     GROQ_API_KEY=<your_groq_api_key>
     # Uncomment and set these if using Azure OpenAI or HuggingFaceHub
     # AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
     # AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
     # OPENAI_API_VERSION=<your_openai_api_version>
     # HUGGINGFACEHUB_API_TOKEN=<your_huggingfacehub_api_token>
     ```


### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

### Upload Files
- Navigate to the "Upload" page to upload files in supported formats (PDF, DOCX, TXT, JSON, CSV).
- Alternatively, provide a YouTube URL for transcription and translation.

### Query Documents
- Select a document from the list and enter your query to retrieve relevant information.

### General Queries
- Use the general chat interface to ask questions without document context. Navigate to the "General Chat" page for this feature.

### Web Scraping
- Enter a website URL to scrape text content and save it as a JSON file.

### Voice Integration
- Use the voice input feature to convert spoken queries into text.
- The platform processes the voice input and provides intelligent responses or transcriptions.
- **Text-to-Speech**: Use the text-to-speech feature to convert text responses or documents into audio. Ensure your speakers are enabled for playback.
- Ensure your microphone is enabled and permissions are granted in your browser.

## Project Structure

- `app.py`: Main Flask application.
- `webscrape.py`: Web scraping utility.
- `youtube.py`: YouTube video processing module.
- `uploads/`: Folder for storing uploaded and processed files.
- `templates/`: HTML templates for the web interface.

## Dependencies

- Flask
- BeautifulSoup (bs4)
- Requests
- Whisper
- Google Translate API
- LangChain
- FAISS
- PyTubeFix
- Torch
- dotenv

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Whisper](https://github.com/openai/whisper)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyTube](https://github.com/pytube/pytube)
