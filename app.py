import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load API Key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="PaperPilot", layout="wide")

# --- SMART LOGIN SYSTEM ---
if "page" not in st.session_state:
    st.session_state.page = "login"
if "users" not in st.session_state:
    st.session_state.users = {"test@gmail.com": "1234"} # Demo ID hidden in backend

def login():
    st.title("PaperPilot – GenAI Research Assistant")
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if email in st.session_state.users and st.session_state.users[email] == password:
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            # FIX 1: Removed the password hint! Now it's secure.
            st.error("Invalid Email or Password! Please check your credentials or Create an Account.")
            
    if st.button("Create Account"):
        st.session_state.page = "signup"
        st.rerun()

def signup():
    st.title("Create Account")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        if email and password:
            st.session_state.users[email] = password
            st.success("Account created successfully! Click Back to Login.")
        else:
            st.warning("Please fill all fields")
            
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# --- AI BACKEND LOGIC ---
def process_pdf(pdf_file):
    if not os.path.exists("temp_pdf"):
        os.makedirs("temp_pdf")
    file_path = os.path.join("temp_pdf", pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks, text

def create_vector_db(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# --- MAIN DASHBOARD ---
def dashboard():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Choose Feature", ["Upload Paper", "Generate Summary", "Generate Notes", "Chatbot"])
    
    MODEL_NAME = "llama-3.3-70b-versatile" 

    st.title("PaperPilot Dashboard")

    if page == "Upload Paper":
        st.header("Upload Research Paper")
        file = st.file_uploader("Upload PDF", type=["pdf"])

        if file:
            with st.spinner("Processing PDF and Saving to ChromaDB..."):
                text_chunks, full_text = process_pdf(file)
                vectorstore = create_vector_db(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_context = full_text[:5000] 
                # Reset chat history when new PDF is uploaded
                st.session_state.messages = [] 
            st.success("PDF Processed Successfully! You can now use other features from the menu.")
            st.info("Note: Don't worry if this message disappears when you change tabs, your PDF is safely saved in the background!")

    elif page == "Generate Summary":
        st.header("AI Summary")
        if "doc_context" not in st.session_state:
            st.warning("⚠️ Please upload a PDF first from the 'Upload Paper' menu.")
        else:
            if st.button("Generate Summary"):
                if not groq_api_key:
                    st.error("Groq API Key not found in .env file!")
                else:
                    with st.spinner("Generating Summary..."):
                        llm = ChatGroq(api_key=groq_api_key, model_name=MODEL_NAME)
                        prompt = f"Write a concise summary of the following research paper context:\n\n{st.session_state.doc_context}"
                        response = llm.invoke(prompt)
                        st.write(response.content)

    elif page == "Generate Notes":
        st.header("Structured Notes")
        if "doc_context" not in st.session_state:
            st.warning("⚠️ Please upload a PDF first from the 'Upload Paper' menu.")
        else:
            if st.button("Generate Structured Notes"):
                if not groq_api_key:
                    st.error("Groq API Key not found in .env file!")
                else:
                    with st.spinner("Extracting Key Notes..."):
                        llm = ChatGroq(api_key=groq_api_key, model_name=MODEL_NAME)
                        prompt = f"Create structured notes (Introduction, Key Points, Conclusion) from this text:\n\n{st.session_state.doc_context}"
                        response = llm.invoke(prompt)
                        st.write(response.content)

    elif page == "Chatbot":
        st.header("Ask Questions (RAG Chatbot)")
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload a PDF first from the 'Upload Paper' menu.")
        else:
            # FIX 2: Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if question := st.chat_input("Ask a question about the uploaded paper..."):
                if not groq_api_key:
                    st.error("Groq API Key not found in .env file!")
                else:
                    # Display user message in chat message container
                    st.chat_message("user").markdown(question)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": question})

                    with st.spinner("Groq AI is reading the paper..."):
                        docs = st.session_state.vectorstore.similarity_search(question, k=3)
                        context = "\n".join([doc.page_content for doc in docs])
                        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                        
                        llm = ChatGroq(api_key=groq_api_key, model_name=MODEL_NAME)
                        response = llm.invoke(prompt)
                        
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(response.content)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response.content})

if st.session_state.page == "login":
    login()
elif st.session_state.page == "signup":
    signup()
elif st.session_state.page == "dashboard":
    dashboard()