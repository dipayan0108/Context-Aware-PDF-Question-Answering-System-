import os
from langchain_community.document_loaders import PyPDFLoader        #to extract text from PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter   #convert text into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings    #to create embeddings for the text chunks
from langchain_community.vectorstores import FAISS   #store and search text embeddings
from langchain_core.prompts import ChatPromptTemplate   #to create a prompt template for the LLM
from langchain_google_genai import ChatGoogleGenerativeAI  #to use Google Gemini for LLM
import tempfile  #create temporary files to handle PDF uploads
from dotenv import load_dotenv  #access secure keys like GOOGLE_API_KEY
from langchain.memory import ConversationBufferMemory  #Stores chat history, enabling the assistant to remember previous interactions
from langchain.chains import ConversationalRetrievalChain   #Used to create a QA chain that includes memory and retrieval capabilities

# Load environment variables
load_dotenv()

class PDFQAPipeline:
    def __init__(self, pdf_file):   #Initializes the pipeline with an uploaded PDF
        """
        Initialize the QA pipeline by processing the uploaded PDF.
        Args:
            pdf_file: Uploaded PDF file object from Streamlit.
        """
        self.documents = self._load_pdf(pdf_file)   #Load and extract text from the PDF
        self.chunks = self._split_documents(self.documents)  #Split the text into manageable chunks
        self.vector_store = self._create_vector_store(self.chunks)  #Converts chunks into embeddings and stores them in a FAISS index
        self.retrieval_chain = self._setup_retrieval_chain(self.vector_store) # Set up the retrieval chain with the vector store and LLM

    def _load_pdf(self, pdf_file):
        """Load and extract text from the PDF file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:  #Creates a temporary file for the uploaded PDF.
            tmp_file.write(pdf_file.read())  #Writes the uploaded PDF content to the temporary file.
            tmp_file_path = tmp_file.name     #Stores the path to the temporary file.
        loader = PyPDFLoader(tmp_file_path) #Uses PyPDFLoader to extract text content from the PDF.
        documents = loader.load()
        os.unlink(tmp_file_path)  # Remove temporary file
        return documents

    def _split_documents(self, documents):    
        """Split documents into chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(     #Configures the text splitter:Each chunk is 1000 characters,
            chunk_size=1000,                                #With 200-character overlap between chunks.
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)   #Splits the loaded documents into smaller chunks for better processing.
        return chunks

    def _create_vector_store(self, chunks):
        """Create a vector store with embeddings for the document chunks."""
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")    #Loads the BAAI/bge-small-en-v1.5 embedding model
        vector_store = FAISS.from_documents(chunks, embeddings) #Creates a FAISS vector store from the document chunks and their embeddings.
        return vector_store

    def _setup_retrieval_chain(self, vector_store): #Prepares the full conversational QA chain.
        """Set up the retrieval chain with the Gemini API and conversational memory."""
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Uses the Gemini 1.5 Flash model for LLM
            google_api_key=os.environ["GOOGLE_API_KEY"], 
            temperature=0.1,    #Loads Gemini model with a low temperature (less randomness) and limits the output length.
            max_output_tokens=512
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  #Stores dialogue history for better context retention.
        #Custom prompt template to instruct the Gemini model to answer based on context + history.
        prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant that answers questions based on the provided PDF content.    
            Use the following context to answer the question: {context}
            Chat History: {chat_history}
            Question: {input}
            Answer:
        """)
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})   #Creates a retriever that fetches top 3 relevant chunks.
        
        """
        Creates the final ConversationalRetrievalChain, integrating:
                    the LLM (Gemini),
                    the retriever (FAISS),
                    memory (chat history),
                    and the custom prompt.
        """

        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return retrieval_chain

    def query(self, user_query):    #Used to handle user input questions.
        """Query the retrieval chain with a user question."""
        response = self.retrieval_chain.invoke({"input": user_query})   #Feeds the query to the chain and gets a response.
        return response["answer"]