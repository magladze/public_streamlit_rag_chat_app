import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from InstructorEmbedding import INSTRUCTOR
import sentence_transformers
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub

import os
import hmac
import hashlib
import base64
import urllib.parse

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "vars.env"))
load_dotenv(env_path)

os.environ["STREAMLIT_SERVER_PORT"] = "8503"

def get_pdf_text(pdf_docs):
    text = ''
    for i in pdf_docs:
        pdf_reader = PdfReader(i)

        for j in pdf_reader.pages:
            text += j.extract_text()

    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator=' \n',
        chunk_size=1500,
        chunk_overlap=300,
        length_function= len
    )

    text_chunks = text_splitter.split_text(raw_text)

    return text_chunks

def get_vector_store(text_chunks):
    embedding_instance = OpenAIEmbeddings()
    # embedding_instance = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    session_vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_instance)

    return session_vector_store

def get_convo_chain(vector_store):
    llm = ChatOpenAI(model_name ='gpt-3.5-turbo-16k')
    # llm = HuggingFaceHub(repo_id="TheBloke/ALMA-7B-Pretrain-GPTQ", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return convo_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace('{{MSG}}', 'QUESTION: '+ message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', 'ANSWER: '+  message.content),unsafe_allow_html=True)

def main():

    load_dotenv()

    st.set_page_config(page_title='Chat with pdf files *TEST*', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with PDF files :books:')

    user_question = st.text_input('Ask a question about your imported docs:')

    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace('{{MSG}}', 'hi poof'), unsafe_allow_html=True)
    # st.write(bot_template.replace('{{MSG}}', 'hi peef'), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your Docs')
        pdf_docs = st.file_uploader("Upload PDFs here and click the 'Process' button",
                         accept_multiple_files=True)


        if st.button('Process'):

            with st.spinner('Processing'):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # st.write(raw_text)

                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                #create vector store
                vector_store = get_vector_store(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_convo_chain(vector_store)


    # st.session_state.conversation





if __name__ == '__main__':
    main()
    # load_dotenv()
    # print(os.environ['OPENAI_API_KEY'])