# PDF Chatbot using LangChain

An interactive PDF-based chatbot that allows users to upload a PDF file and ask contextual questions about its content. The app uses [LangChain](https://www.langchain.com/), Google Gemini, and FAISS vector search to deliver accurate and conversational answers based on document context.

## Features

- Upload any PDF file
- Extracts and processes text using `PyPDFLoader`
- Splits text into semantic chunks with `RecursiveCharacterTextSplitter`
- Embeds text using `HuggingFaceEmbeddings`
- Stores and retrieves embeddings with `FAISS`
- Queries handled by Google Gemini 
- Maintains conversation memory with `ConversationBufferMemory`
- Streamlit-powered interactive chat interface

## Tech Stack

- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/)
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot

2. **Create a virtual environment**
     ```bash
     python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

4. **Install dependencies**
     ```bash
    pip install -r requirements.txt

5. **Set up environment variables**
    Create a .env file in the project root:
      ```bash
    GOOGLE_API_KEY=your_google_api_key_here

7. **Running the App**
    ```bash
    streamlit run app.py
  Upload a PDF, ask questions, and get intelligent responses in real-time.

## Example Use Cases
1. Summarizing research papers
2. Extracting key clauses from legal documents
3. Conversationally navigating technical manuals
4 .Interactive help for handbooks or user guides
