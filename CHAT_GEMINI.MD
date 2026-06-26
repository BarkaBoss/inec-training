# INEC Knowledge Chatbot Code Explanation

# Building a Retrieval-Augmented Generation (RAG) Chatbot using Streamlit, LangChain, ChromaDB, and Google Gemini

---

# Overview

This application is a **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions based on **INEC (Independent National Electoral Commission)** documents stored as PDF files.

Unlike a traditional chatbot that answers from its own knowledge, this application first searches through uploaded PDF documents, retrieves the most relevant information, and then asks Google's Gemini model to generate an answer using only the retrieved context.

The system combines:

* **Streamlit** for the web interface
* **LangChain** for orchestrating the RAG pipeline
* **Google Gemini Embeddings** for semantic search
* **ChromaDB** as an in-memory vector database
* **Gemini Flash** as the Large Language Model (LLM)

---

# Overall Architecture

```
                PDF Documents
                      │
                      ▼
          PyPDFDirectoryLoader
                      │
                      ▼
     RecursiveCharacterTextSplitter
                      │
                      ▼
     Google Embedding Model
                      │
                      ▼
           Chroma Vector Database
                      │
                      ▼
          Similarity Search (Top 3)
                      │
                      ▼
           Retrieved Context
                      │
                      ▼
      Prompt Template + User Question
                      │
                      ▼
        Gemini Flash Large Language Model
                      │
                      ▼
            Generated Response
                      │
                      ▼
        Streamlit Chat Interface
```

---

# Import Statements

```python
import os
import streamlit as st
```

These are standard Python libraries.

| Library   | Purpose                    |
| --------- | -------------------------- |
| os        | File and folder management |
| streamlit | Builds the web application |

---

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
```

Loads every PDF inside the `data` directory automatically.

Instead of loading one file:

```python
loader = PyPDFLoader("guide.pdf")
```

this loader imports every PDF found in the folder.

---

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Splits long documents into overlapping chunks.

Large Language Models have token limits, so documents must be divided into manageable sections.

---

```python
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
```

Imports two Google Gemini services.

### GoogleGenerativeAIEmbeddings

Converts text into numerical vectors.

Example

```
Election guidelines

↓

[-0.23, 0.61, 0.15, ...]
```

These vectors are used for semantic search.

---

### ChatGoogleGenerativeAI

Loads the Gemini language model.

Instead of extracting an exact sentence like BERT QA, Gemini generates a natural-language answer using the retrieved documents.

---

```python
from langchain_community.vectorstores import Chroma
```

Imports **ChromaDB**, a vector database.

Its job is to:

* store document embeddings
* search them efficiently
* retrieve the most relevant chunks

---

```python
from langchain_core.prompts import ChatPromptTemplate
```

Creates structured prompts for the language model.

---

```python
from langchain_core.runnables import RunnablePassthrough
```

Allows the user's question to flow unchanged through the LangChain pipeline.

---

```python
from langchain_core.output_parsers import StrOutputParser
```

Converts the model's output into plain text.

---

# Streamlit Page Configuration

```python
st.set_page_config(
    page_title="INEC info bot",
    layout="wide"
)
```

Configures:

* browser tab title
* page width

---

```python
st.title("INEC Knowledge Chatbot")
```

Displays the application title.

---

```python
st.write(
    "Ask questions based on the INEC documents uploaded..."
)
```

Shows instructions for the user.

---

# Google API Key

```python
os.environ["GOOGLE_API_KEY"] = "API_KEY"
```

Stores the Gemini API key as an environment variable.

The embedding model and Gemini LLM use this key to authenticate requests.

**Note:**

In production, never hardcode API keys. Use:

* `.env`
* environment variables
* Streamlit secrets

instead.

---

# Data Directory

```python
DATA_DIR = "data"
```

Specifies where PDF files are stored.

---

```python
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
```

Automatically creates the folder if it doesn't exist.

---

# Formatting Retrieved Documents

```python
def format_docs(docs):
```

This helper function combines multiple retrieved document chunks into one string.

```python
return "\n\n".join(doc.page_content for doc in docs)
```

Suppose the retriever returns:

```
Chunk 1

Chunk 2

Chunk 3
```

The function combines them into

```
Chunk 1

Chunk 2

Chunk 3
```

which becomes the context sent to Gemini.

---

# Caching the RAG System

```python
@st.cache_resource
```

Caching prevents rebuilding the vector database every time Streamlit refreshes.

Without caching:

```
Load PDFs

↓

Split

↓

Create embeddings

↓

Build Chroma

↓

Every refresh
```

With caching, this work happens only once.

---

# Loading PDFs

```python
loader = PyPDFDirectoryLoader(DATA_DIR)
docs = loader.load()
```

Loads every PDF inside the `data` folder.

---

```python
if not docs:
    return None
```

Stops processing if there are no documents.

---

# Splitting Documents

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Each document is divided into

* 1000-character chunks
* 200-character overlap

Example

```
Chunk 1

ABCDEFGHIJKLMNOP

Chunk 2

MNOPQRSTUVWXYZ
```

The overlap preserves context.

---

# Creating Embeddings

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
```

Each chunk becomes an embedding.

```
Chunk

↓

Vector

↓

Stored inside Chroma
```

Embeddings capture meaning rather than exact words.

---

# Creating the Vector Database

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

Stores every chunk and its embedding inside Chroma.

```
Chunk A

↓

Vector

↓

Stored
```

---

# Creating the Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)
```

Whenever a question is asked,

the retriever returns the **three most relevant chunks**.

```
Question

↓

Embedding

↓

Similarity Search

↓

Top 3 Chunks
```

---

# System Prompt

```python
system_prompt = (
    "You are an expert assistant..."
)
```

The prompt instructs Gemini to

* answer only from retrieved documents
* avoid hallucinating information
* remain concise
* remain professional

This is one of the most important parts of a RAG application.

---

# Prompt Template

```python
prompt = ChatPromptTemplate.from_messages([
```

Creates a conversation template.

```
System

↓

Retrieved Context

↓

User Question
```

Resulting prompt:

```
System:
Use the following context...

Context:
...

Human:
What are voter registration requirements?
```

---

# Loading Gemini

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2
)
```

Loads the Gemini Flash model.

### Temperature

```
temperature = 0.2
```

A low temperature makes responses:

* more factual
* less random
* more consistent

Ideal for question answering.

---

# Building the RAG Chain

```python
rag_chain = (
```

This creates the complete LangChain pipeline.

---

## Step 1

```python
{
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
}
```

The user's question goes to two places.

### Path 1

```
Question

↓

Retriever

↓

Relevant Chunks

↓

format_docs
```

Produces the context.

---

### Path 2

```
Question

↓

RunnablePassthrough
```

Passes the original question unchanged.

---

## Step 2

```python
| prompt
```

Combines

* context
* question

into a structured prompt.

---

## Step 3

```python
| llm
```

Gemini generates the answer.

---

## Step 4

```python
| StrOutputParser()
```

Converts the AI response into plain text.

---

# Complete LangChain Pipeline

```
User Question
        │
        ▼
RunnablePassthrough
        │
        ├──────────────┐
        │              │
        ▼              ▼
 Retriever        Original Question
        │
        ▼
Top 3 Chunks
        │
        ▼
 format_docs
        │
        ▼
 Prompt Template
        │
        ▼
 Gemini Flash
        │
        ▼
 Output Parser
        │
        ▼
 Final Answer
```

---

# Sidebar

```python
with st.sidebar:
```

Creates the left sidebar.

---

```python
pdf_files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".pdf")
]
```

Lists every PDF currently available.

---

# Initializing the System

```python
rag_chain = initialize_rag_system()
```

Builds

* embeddings
* vector database
* retriever
* prompt
* Gemini model
* complete RAG pipeline

---

# Chat History

```python
if "messages" not in st.session_state:
```

Stores previous conversations.

Without this, messages disappear whenever Streamlit reruns.

---

# Displaying Previous Messages

```python
for message in st.session_state.messages:
```

Loops through every stored message.

---

# User Input

```python
st.chat_input(...)
```

Displays the chat input box.

Example:

```
Who can register as a voter?
```

---

# Displaying the User Message

```python
st.chat_message("user")
```

Shows the user's question.

---

# Invoking the RAG Pipeline

```python
answer = rag_chain.invoke(user_query)
```

This single line executes the entire workflow.

Internally it performs:

```
Question

↓

Embedding

↓

Similarity Search

↓

Retrieve Top Chunks

↓

Build Prompt

↓

Gemini

↓

Generate Answer
```

---

# Displaying the Answer

```python
st.markdown(answer)
```

Displays Gemini's response.

---

# Saving Conversation History

```python
st.session_state.messages.append(...)
```

Stores the assistant's answer so it appears after future Streamlit reruns.

---

# Complete Execution Flow

```
Application Starts
        │
        ▼
Load PDF Files
        │
        ▼
Split Documents
        │
        ▼
Generate Embeddings
        │
        ▼
Store in Chroma
        │
        ▼
Create Retriever
        │
        ▼
Create Prompt Template
        │
        ▼
Load Gemini
        │
        ▼
Build RAG Chain
        │
        ▼
User Asks Question
        │
        ▼
Retrieve Top 3 Chunks
        │
        ▼
Insert Context into Prompt
        │
        ▼
Gemini Generates Response
        │
        ▼
Display Answer
```

---

# Technologies Used

| Technology                     | Purpose                      |
| ------------------------------ | ---------------------------- |
| Streamlit                      | User Interface               |
| LangChain                      | RAG orchestration            |
| Google Gemini Embeddings       | Semantic vector generation   |
| ChromaDB                       | Vector storage and retrieval |
| Gemini Flash                   | Large Language Model         |
| RecursiveCharacterTextSplitter | Document chunking            |
| PyPDFDirectoryLoader           | PDF loading                  |

---

# Comparison with the Previous PyTorch Chatbot

| PyTorch QA Chatbot                       | Gemini RAG Chatbot                  |
| ---------------------------------------- | ----------------------------------- |
| Uses Sentence Transformers for retrieval | Uses Gemini Embeddings              |
| Uses BERT QA to extract answers          | Uses Gemini to generate answers     |
| Extractive Question Answering            | Generative Question Answering       |
| Returns exact text from documents        | Produces natural-language responses |
| Uses manual cosine similarity search     | Uses Chroma vector search           |
| Manually handles retrieval               | LangChain automates the pipeline    |

---

# Strengths of This Approach

* Simple and modular RAG architecture.
* Uses state-of-the-art Gemini embeddings.
* Natural language responses instead of extracted snippets.
* Chroma provides efficient similarity search.
* LangChain significantly reduces implementation complexity.
* Easy to expand by adding more PDF documents.
* Professional responses guided by a carefully designed system prompt.

---

# Possible Improvements

* Persist the Chroma database to disk so embeddings are not regenerated each time the application restarts.
* Display the source document name and page number for each retrieved chunk.
* Add support for PDF uploads directly within the Streamlit interface.
* Enable conversational memory for follow-up questions.
* Implement hybrid retrieval using both keyword search (BM25) and vector search.
* Allow users to adjust the number of retrieved documents (`k`) through the interface.
* Stream Gemini responses token-by-token for a smoother user experience.
* Add authentication and secure API key management using environment variables or Streamlit Secrets.
* Include citations or highlighted document excerpts alongside generated answers for improved transparency.
