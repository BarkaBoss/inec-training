import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="INEC info bot", layout="wide")
st.title("INEC Knowledge Chatbot")
st.write("Ask questions based on the INEC documents uploaded in the `/data` directory.")

os.environ["GOOGLE_API_KEY"] = "API_KEY"

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Analyzing PDF Documents")
def initialize_rag_system():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an expert assistant for an organization. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that "
        "you don't know. Do not make up information. Keep your answers concise and professional.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


with st.sidebar:
    st.header("Your Documents")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if pdf_files:
        st.success(f"Loaded {len(pdf_files)} PDF file(s):")
        for file in pdf_files:
            st.markdown(f"- `{file}`")
    else:
        st.warning("No PDFs found! Drop your PDFs into the `/data` folder of the project to begin.")

rag_chain = initialize_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if rag_chain is None:
    st.info("Please add PDF documents to your local `data/` folder of the project to begin.")
else:
    if user_query := st.chat_input("Ask something about voting policies, election guideline, or voter registration..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching files..."):
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})