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

choice = st.radio("Pick the type of url given:",['Website URL','PDF URL'])
url = st.text_input("Enter the URL LINK here.", value = "https://www.constituteproject.org/constitution/Nepal_2015")

if st.button("Initialize Rag"):
    with st.spinner("Initializing Rag Setup to enable Q/A"):

        # Laoding Document from website and splitting into chunks
        if choice == 'Website URL':
            loader = WebBaseLoader(url)
        elif choice == 'PDF URL':
            loader = PyMuPDFLoader(url)
        else:
            st.error('Please choose valid URL option.')
            st.stop()
        try:
            documents = loader.load()
        except:
            st.error('Invalid or unsupported URL')
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200, separators = ["\n\n","\n"," ",""])
        chunks = text_splitter.split_documents(documents)

        #Embedding the document
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_documents(documents = chunks, embedding= embedding_model)
        st.session_state.bm25_retriever = BM25Retriever.from_documents(chunks)

        st.success('Rag Initialized successfully, Now you can ask questions.')

if st.session_state.vector_store is not None:

    chat_model = ChatOpenRouter(
        model="arcee-ai/trinity-large-preview:free",
        temperature=0.5,
        max_tokens=1024,
        max_retries=3,
    )

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
                    bm25_retriever.k = 3

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
                except Exception:
                    st.error("Could not generate an answer right now. Please try again.")
        
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




