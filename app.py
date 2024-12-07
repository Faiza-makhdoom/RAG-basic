#LLM problem resolution ---> pip install -qU langchain-google-genai

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # Accumulate text from each page
    return text

# Split extracted text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create in-memory FAISS vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store  # Return in-memory vector store

# Configure conversational chain for question-answering
def get_conversational_chain():
    prompt_template = """
    Answer the question precisely from the provided context, also make sure to provide all the details. 
    If the answer is not in the provided context, just say, "I am sorry, answer is not available in the context."
    Do not provide a wrong answer.If user greets or say thanks to you, reply nicely and accordingly. In case the question is not relevent to the provided document, politely response 
    that "The answer to your question is not available in the provided text. How can I help you further?". you are provided with 
    the chat history too so if the user asks any question indirectly it may be related to the previous questions, 
    so before refusing, also go through the chat history too\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    chain = load_qa_chain(LLM, chain_type="stuff", prompt=prompt)

    return chain
    
# Handle user input to get response based on vector store
def user_input(user_question):
    if "vector_store" not in st.session_state:
        st.write("Please process the PDF files first.")
        return

    # Use in-memory vector store for similarity search
    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    #print(response)
    #st.write("Reply: ", response["output_text"])


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"user": user_question, "ai": response["output_text"]})

    # Display chat history
    response_container = st.container()
    with response_container:
        for i, entry in enumerate(st.session_state.chat_history):
            message(entry["user"], is_user=True, key=f"user_{i}")
            message(entry["ai"], key=f"ai_{i}")

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""

# Main app function
def main():
    st.set_page_config("DocumentGPT")
    st.header("Chat with your PDFs")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt'] = ""

    #user_question = st.text_input("Ask Question about your files", key='prompt_input', on_change=submit)

    # input box apears at bottom
    user_question = st.chat_input("Ask Question about your files")
    if st.session_state.entered_prompt != "":
    # Get user query
        user_question = st.session_state.entered_prompt
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "**Upload your PDF Files and Click on the Submit & Process Button**",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)  # Store in session state
                st.success("Done")

if __name__ == "__main__":
    main()