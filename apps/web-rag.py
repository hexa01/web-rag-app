import tempfile

from langchain_core.documents import Document
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openrouter.chat_models import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyMuPDFLoader

st.set_page_config(page_title = "Rag Web App", page_icon = "💻", layout="wide")
st.title("Rag Questioning from any website or pdf")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()


if not os.environ.get('OPENROUTER_API_KEY'):
    st.error('OPENROUTER API KEY not set, please set it in the environment variable')
    st.stop()

if not os.environ.get('HF_TOKEN'):
    st.error('HuggingFace TOKEN not set, please set it in the environment variable')
    st.stop()   

input_mode = st.radio("Pick the input method:", ["URL","FILE"], horizontal= True)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_chat_model():
    return ChatOpenRouter(
        model="arcee-ai/trinity-large-preview:free",
        temperature=0.5,
        max_tokens=1024,
        max_retries=3,
    )

if 'documents' not in st.session_state:
    st.session_state.documents = []

if input_mode == 'URL':
    choice = st.radio("Pick the type of url given:",['Website URL','PDF URL'])
    url = st.text_input("Enter the URL LINK here.", value = "https://www.constituteproject.org/constitution/Nepal_2015")
    if st.button("Initialize Rag"):
        with st.spinner("Processing and Loading URL, Please wait!"):

            # Laoding Document from website and splitting into chunks
            if choice == 'Website URL':
                loader = WebBaseLoader(url)
            elif choice == 'PDF URL':
                loader = PyMuPDFLoader(url)
            else:
                st.error('Please choose valid URL option.')
                st.stop()
            try:
                st.session_state.documents.extend(loader.load())
                st.success('URL document successfully loaded!')
            except Exception as e:
                st.error(f'Failed to load URL: {e}')

            
elif input_mode == "FILE":
    uploaded_file = st.file_uploader("Upload the file:", type=['txt','pdf'])
    if st.button("Initialize Rag"):
        with st.spinner("Processing and Loading Documents, Please wait!"):
            if uploaded_file is not None:
                file_type = uploaded_file.name.split(".")[-1]
                if file_type == 'txt':
                    raw_text = uploaded_file.read().decode('utf-8')
                    st.session_state.documents.append(Document(page_content = raw_text))
                    st.success('txt document successfully loaded!')
                    
                elif file_type == 'pdf':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                        try:
                            loader = PyMuPDFLoader(temp_path)
                            st.session_state.documents.extend(loader.load())
                            st.success('pdf document successfully loaded!')
                        finally:
                            os.unlink(temp_path)
                else:
                    st.error('File type not supported, Please upload a txt or pdf file.')
                    st.stop()
            else:
                st.error("Please upload a file")
                st.stop()


if st.session_state.documents and not st.session_state.get('rag_initialized'):
    with st.spinner('Now, Initializing Rag, Please wait!'):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200, separators = ["\n\n","\n"," ",""])
        chunks = text_splitter.split_documents(st.session_state.documents)
        #Embedding the document
        embedding_model = load_embedding_model()
        st.session_state.vector_store = FAISS.from_documents(documents = chunks, embedding= embedding_model)
        st.session_state.bm25_retriever = BM25Retriever.from_documents(chunks, k = 3)
        st.success('Rag Initialized successfully, Now you can ask questions.')
        st.session_state.rag_initialized = True

if st.session_state.vector_store is not None:

    chat_model = load_chat_model()

    prompt = ChatPromptTemplate.from_messages([
        ('system',"You are an assistant for question-answering tasks. Given the following extracted parts of a long document and a question, create a final answer with citations. If the answer is not in the context, say 'I don't know'."),
        ('human',"Context:{context}\n\nQuestion: {question}\n\nFinal Answer (with citation):")
    ])

    chain = prompt | chat_model
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Ask a Question')
        question = st.text_area('Enter your query')
        
        if st.button('Ask'):
            with st.spinner('Finding the answer....'):
                try:
                    
                    vector_retriever = st.session_state.vector_store.as_retriever(search_kwargs = {"k" : 3})
                    bm25_retriever = st.session_state.bm25_retriever

                    # 3. Ensemble (Combines results using RRF)
                    hybrid_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )

                    retrieved_docs = hybrid_retriever.invoke(question)
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    response = chain.invoke({
                        "context":context,
                        "question": question
                    })

                    st.session_state.last_response = response.content
                    st.session_state.last_context = retrieved_docs
                except Exception as e:
                    st.error(e)
                    st.error("Could not generate an answer right now. Please try again.")
                    st.stop()
        
    with col2:
        st.subheader('Answer')
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)

        with st.expander('Show Retrieved Context'):
            if 'last_context' in st.session_state:
                for i,doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f'Relevant Context {i}:')
                    st.markdown(doc.page_content)
                    st.markdown("-"*20)




