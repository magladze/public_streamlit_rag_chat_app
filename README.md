# Chat with PDF Files using RAG (Retrieval-Augmented Generation)

This project is a Retrieval-Augmented Generation (RAG) chat application built using various AI and machine learning libraries. The application allows users to upload PDF documents and interact with them through a chat interface, asking questions and receiving detailed answers based on the content of the documents.

## Table of Contents
1. [Introduction to RAG](#introduction-to-rag)
2. [Libraries and Tools](#libraries-and-tools)
3. [Code Walkthrough](#code-walkthrough)
    - [PDF Text Extraction](#pdf-text-extraction)
    - [Text Chunking](#text-chunking)
    - [Vector Store Creation](#vector-store-creation)
    - [Conversational Chain](#conversational-chain)
    - [User Input Handling](#user-input-handling)
    - [Main Function](#main-function)
4. [Use Cases](#use-cases)
5. [Future Enhancements](#future-enhancements)

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a powerful technique in natural language processing that combines the strengths of retrieval-based and generation-based models. RAG models first retrieve relevant documents or passages from a large corpus and then generate coherent and contextually appropriate responses based on the retrieved information. This approach is particularly useful for tasks requiring accurate and contextually rich answers, such as question answering, summarization, and interactive chat applications.

## Libraries and Tools

### Streamlit
Streamlit is an open-source app framework for Machine Learning and Data Science projects. It allows for the quick creation of web applications with minimal effort.

### Python-dotenv
This library allows you to load environment variables from a `.env` file into your Python application, making it easier to manage configuration settings securely.

### PyPDF2
PyPDF2 is a library for reading and manipulating PDF files in Python. It provides functionalities to extract text, merge PDFs, and more.

### InstructorEmbedding and Sentence Transformers
These libraries are used for generating high-quality text embeddings, which are numerical representations of text that capture semantic meaning.

### Langchain
Langchain is a library designed to simplify working with language models by providing utilities for text splitting, embeddings, vector stores, and more.

### FAISS
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.

### OpenAI's GPT-3.5-turbo
This is a state-of-the-art language model developed by OpenAI, capable of generating human-like text and understanding context.

### HuggingFace
HuggingFace is a popular platform for natural language processing with a vast collection of pre-trained models and tools.

## Code Walkthrough

### PDF Text Extraction

The function `get_pdf_text` extracts text from uploaded PDF documents.

```python
def get_pdf_text(pdf_docs):
    text = ''
    for i in pdf_docs:
        pdf_reader = PdfReader(i)
        for j in pdf_reader.pages:
            text += j.extract_text()
    return text
```

### Text Chunking

The function `get_text_chunks` splits the raw text into manageable chunks for processing.

```python
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator=' \n',
        chunk_size=1500,
        chunk_overlap=300,
        length_function= len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks
```

### Vector Store Creation

The function `get_vector_store` creates a vector store from the text chunks using embeddings.

```python
def get_vector_store(text_chunks):
    embedding_instance = OpenAIEmbeddings()
    # embedding_instance = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    session_vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_instance)
    return session_vector_store
```

### Conversational Chain

The function `get_convo_chain` sets up a conversational chain using a language model and a vector store.

```python
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
```

### User Input Handling

The function `handle_user_input` processes the user's question and generates a response based on the chat history and the document content.

```python
def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace('{{MSG}}', 'QUESTION: '+ message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', 'ANSWER: '+  message.content),unsafe_allow_html=True)
```

### Main Function

The main function orchestrates the entire application workflow, from loading environment variables to handling user interactions.

```python
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
    with st.sidebar:
        st.subheader('Your Docs')
        pdf_docs = st.file_uploader("Upload PDFs here and click the 'Process' button", accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_convo_chain(vector_store)
if __name__ == '__main__':
    main()
```

## Use Cases

1. **Academic Research**: Researchers can upload academic papers and interactively query specific details or summaries.
2. **Legal Documents**: Lawyers can upload legal documents and quickly find relevant sections or answers to specific legal questions.
3. **Business Reports**: Business analysts can upload reports and extract key insights interactively.

## Future Enhancements

1. **Multiple Document Support**: Enable querying across multiple documents simultaneously.
2. **Enhanced UI/UX**: Improve the user interface for better user experience.
3. **Advanced Analytics**: Integrate advanced analytics to provide deeper insights from documents.
4. **Voice Interaction**: Add support for voice-based interactions.
5. **Real-time Collaboration**: Allow multiple users to interact with the documents in real-time.

## Conclusion

This RAG-based chat application showcases the power of combining retrieval and generation techniques to create an interactive and intelligent system for querying PDF documents. With further enhancements, this tool can be adapted for various professional and academic use cases, making document analysis and interaction more efficient and user-friendly.