from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
import streamlit as st
from langchain.chains import ConversationalRetrievalChain

# Set your OpenAI API key
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Use the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("Q&A Chatbot")
st.write("Upload a PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Read the PDF file
    doc_reader = PdfReader(uploaded_file)

    # Extract text from the PDF
    raw_text = ""
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
            
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create the document search
    docsearch = FAISS.from_texts(texts, embeddings)

    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    # Initialize chat history list
    chat_history = []

    # Get the user's query
    query = st.text_input("Ask a question about the uploaded document:")

    # Add a generate button
    generate_button = st.button("Generate Answer")

    if generate_button and query:
        with st.spinner("Generating answer..."):
            result = qa({"question": query, "chat_history": chat_history})

            answer = result["answer"]
            source_documents = result['source_documents']

            # Combine the answer and source_documents into a single response
            response = {
                "answer": answer,
                "source_documents": source_documents
                        }
            st.write("response:", response)