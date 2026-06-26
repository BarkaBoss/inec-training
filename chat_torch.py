import os
import streamlit as st
import torch
import torch.nn as nn
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util

# 1. Page Configuration
st.set_page_config(page_title="Torch Chatbot", page_icon="", layout="wide")
st.title("Chat with a document")
st.write("Scan and extract answers from local PDFs.")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


@st.cache_resource(show_spinner="Loading PyTorch models")
def load_pytorch_models():
    retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

    reader_model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(reader_model_name)

    return retriever_model, tokenizer, model


retriever_model, tokenizer, reader_model = load_pytorch_models()


@st.cache_resource(show_spinner="Vectorizing local PDFs")
def process_documents():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    if not docs:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in splits]

    with torch.no_grad():
        embeddings = retriever_model.encode(texts, convert_to_tensor=True)

    return texts, embeddings


texts, doc_embeddings = process_documents()

with st.sidebar:
    st.header("File(s) in `/data`")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if pdf_files:
        st.success(f"Cached {len(pdf_files)} PDF file(s).")
        for file in pdf_files:
            st.markdown(f"- `{file}`")
    else:
        st.warning("No PDFs found! Drop your PDFs in the `/data` directory to start.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if texts is None:
    st.info("Please add PDF documents to your local `data/` directory and refresh the app.")
else:
    if user_query := st.chat_input("Ask a factual question about your documents..."):

        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching and parsing result chunks"):

                with torch.no_grad():
                    query_embedding = retriever_model.encode(user_query, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=min(3, len(texts)))

                context_chunks = [texts[idx] for idx in top_results.indices.tolist()]
                combined_context = "\n\n".join(context_chunks)

                inputs = tokenizer(
                    user_query,
                    combined_context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = reader_model(**inputs)

                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)

                answer_tokens = inputs.input_ids[0][start_idx: end_idx + 1]
                extracted_string = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                if extracted_string.strip() and start_idx < end_idx:
                    answer = f"**Answer extracted from source:** \n> {extracted_string.strip()}"
                else:
                    answer = "I couldn't locate a definitive answer snippet matching your query inside the context document blocks."

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})