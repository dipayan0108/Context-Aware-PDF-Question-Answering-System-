import streamlit as st
from qa_pipeline import PDFQAPipeline #Imports the custom PDF QA class that handles document processing and answering questions
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")    #Sets the browser tab title and icon for the Streamlit app.
    st.title("PDF Chatbot")
    st.write("Upload a PDF and ask questions about its content")

    # Initialize session state
    if 'qa_pipeline' not in st.session_state:   #Checks if the QA pipeline is already initialized in session state.
        st.session_state.qa_pipeline = None     # Initializes the QA pipeline to None.
    if 'chat_history' not in st.session_state:   #Checks if chat history exists in session state.
        st.session_state.chat_history = []     # Initializes an empty chat history to store user queries and responses.

    # PDF upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf") #Allows users to upload a PDF file for processing.
    if uploaded_file is not None and st.session_state.qa_pipeline is None:   #Checks if a file is uploaded and the QA pipeline is not initialized.
        with st.spinner("Processing PDF..."):
            try:
                # Initialize QA pipeline with uploaded PDF
                st.session_state.qa_pipeline = PDFQAPipeline(uploaded_file)   #Creates a new instance of the PDFQAPipeline with the uploaded file.
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # Chat interface
    if st.session_state.qa_pipeline is not None:   #Checks if the QA pipeline is initialized.
        st.write("You can now ask questions about the PDF")
        st.subheader("Ask a Question")
        user_query = st.text_input("Enter your question about the PDF:", key="user_query")  #Input field for user queries.
        if st.button("Submit"):
            if user_query:
                with st.spinner("Generating answer..."):
                    try:
                        response = st.session_state.qa_pipeline.query(user_query)     #Queries the QA pipeline with the user's question and gets a response.
                        st.session_state.chat_history.append({"question": user_query, "answer": response})  #Appends the question and answer to the chat history.
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You**: {chat['question']}")   #Displays the user's question.
            st.markdown(f"**Chatbot**: {chat['answer']}")  #Displays the chatbot's response.
            st.markdown("---") 

if __name__ == "__main__":
    main()