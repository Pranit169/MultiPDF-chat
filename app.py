import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
    

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks= text_splitter.split_text(text)
    return chunks
    
def get_vectorstores(text_chunks):
    embedding=SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    vectorstores = FAISS.from_texts(
        texts = text_chunks,embedding = embedding)
    
    return vectorstores
    
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="deepset/roberta-base-squad2",model_kwars ={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        
    )
    
    
    
    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multi-PDF",page_icon=":books:")
    
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question")
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader(
            "Upload",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)
                
                vectorstore = get_vectorstores(text_chunks)
                
                conversation = get_conversation_chain(vectorstore)
        

if __name__ == '__main__':
    main()