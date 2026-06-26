# Torch Chatbot Code Explanation

# Chat with a Document using PyTorch, Streamlit, and Transformers

## Overview

This application is a **Retrieval-Augmented Question Answering (RAG)** system built with **PyTorch**, **Streamlit**, **Sentence Transformers**, **Hugging Face Transformers**, and **LangChain**.

Instead of generating answers from memory, the chatbot:

1. Loads PDF documents from a local folder.
2. Splits them into smaller chunks.
3. Converts each chunk into a numerical vector (embedding).
4. Finds the document chunks most similar to the user's question.
5. Uses a BERT Question Answering model to extract the exact answer from those chunks.
6. Displays the extracted answer in a chat interface.

---

# Overall Workflow

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
      SentenceTransformer Embeddings
                  │
                  ▼
        Similarity Search (Cosine)
                  │
                  ▼
    Most Relevant Document Chunks
                  │
                  ▼
        BERT Question Answering
                  │
                  ▼
          Extracted Answer
                  │
                  ▼
         Streamlit Chat Interface
```

---

# Import Statements

```python
import os
import streamlit as st
import torch
import torch.nn as nn
```

These import the standard Python libraries and machine learning framework.

| Library | Purpose |
|----------|----------|
| os | Works with folders and files |
| streamlit | Creates the web application |
| torch | PyTorch deep learning framework |
| torch.nn | Neural network utilities (not directly used here) |

---

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
```

Loads every PDF inside a folder.

Instead of loading one PDF:

```python
loader = PyPDFLoader("book.pdf")
```

this loader automatically loads **all PDFs** inside the `data` directory.

---

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Splits long documents into smaller overlapping pieces.

Example:

Original text:

```
Artificial Intelligence is transforming healthcare...
```

After splitting:

Chunk 1

```
Artificial Intelligence is transforming healthcare...
```

Chunk 2

```
...transforming healthcare by improving diagnosis...
```

Overlapping text helps preserve context.

---

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
```

Loads a pretrained Hugging Face Question Answering model.

The tokenizer converts text into tokens.

The model predicts:

- where the answer starts
- where the answer ends

---

```python
from sentence_transformers import SentenceTransformer, util
```

Used for semantic search.

Instead of keyword matching:

Question:

```
How do solar panels work?
```

Document:

```
Photovoltaic cells convert sunlight into electricity.
```

The model recognizes these are semantically similar.

---

# Page Configuration

```python
st.set_page_config(
    page_title="Torch Chatbot",
    page_icon="",
    layout="wide"
)
```

Configures the Streamlit page.

Parameters:

- browser title
- page icon
- layout width

---

```python
st.title("Chat with a document")
st.write("Scan and extract answers from local PDFs.")
```

Creates the application heading.

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

If the folder doesn't exist, create it automatically.

---

# Loading the PyTorch Models

```python
@st.cache_resource(show_spinner="Loading PyTorch models")
```

Caches the models.

Without caching:

Every page refresh would download and reload the models.

With caching:

The models are loaded only once.

---

## Sentence Transformer

```python
retriever_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

This model converts text into embeddings.

Example:

```
"This is a cat."

↓

[-0.21, 0.88, ...]
```

Every sentence becomes a vector.

---

## Question Answering Model

```python
reader_model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
```

Loads a BERT model fine-tuned on the SQuAD2 dataset.

This model does **extractive question answering**, meaning it selects an answer directly from the provided text.

---

```python
tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
```

Loads the tokenizer.

Example:

```
Artificial Intelligence
```

becomes

```
["Artificial", "Intelligence"]
```

which are converted into token IDs.

---

```python
model = AutoModelForQuestionAnswering.from_pretrained(reader_model_name)
```

Loads the neural network.

---

# Processing Documents

```python
loader = PyPDFDirectoryLoader(DATA_DIR)
docs = loader.load()
```

Reads every PDF inside the `data` folder.

---

```python
if not docs:
    return None, None
```

If there are no PDFs, the application stops processing.

---

# Splitting Documents

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

Each chunk contains:

- 500 characters
- 100-character overlap

Example:

```
Chunk 1
ABCDEFGHIJKLMNO

Chunk 2
KLMNOPQRSTUVWXYZ
```

The overlap preserves context between chunks.

---

```python
splits = text_splitter.split_documents(docs)
```

Splits every PDF into chunks.

---

```python
texts = [doc.page_content for doc in splits]
```

Extracts only the text from each chunk.

---

# Creating Embeddings

```python
with torch.no_grad():
```

Disables gradient computation.

Since we're only performing inference (not training), gradients are unnecessary and disabling them:

- saves memory
- speeds up execution

---

```python
embeddings = retriever_model.encode(
    texts,
    convert_to_tensor=True
)
```

Converts every chunk into an embedding.

Result:

```
Chunk 1 → Vector

Chunk 2 → Vector

Chunk 3 → Vector
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

Lists every PDF in the folder.

---

Displays:

```
report.pdf

book.pdf

notes.pdf
```

---

# Chat History

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Stores previous messages.

Without this, the chat disappears whenever Streamlit reruns.

---

```python
for message in st.session_state.messages:
```

Displays all previous messages.

---

# User Input

```python
user_query = st.chat_input(...)
```

Creates the input box.

Example:

```
What is machine learning?
```

---

# Encoding the User Question

```python
query_embedding = retriever_model.encode(
    user_query,
    convert_to_tensor=True
)
```

The question is converted into a vector.

---

# Similarity Search

```python
cos_scores = util.cos_sim(
    query_embedding,
    doc_embeddings
)[0]
```

Computes cosine similarity.

Formula:

```
Similarity =
A·B
──────
|A||B|
```

Higher similarity means more relevant content.

---

# Selecting Top Results

```python
top_results = torch.topk(
    cos_scores,
    k=min(3, len(texts))
)
```

Returns the three most relevant chunks.

---

# Building Context

```python
context_chunks = [
    texts[idx]
    for idx in top_results.indices.tolist()
]
```

Retrieves the selected text chunks.

---

```python
combined_context = "\n\n".join(context_chunks)
```

Combines them into one passage.

---

# Tokenization

```python
inputs = tokenizer(
    user_query,
    combined_context,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
```

Creates the BERT input.

Input format:

```
[CLS]

Question

[SEP]

Document

[SEP]
```

---

# Running BERT

```python
outputs = reader_model(**inputs)
```

Predicts:

- answer start
- answer end

---

# Finding the Answer

```python
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)
```

Finds the most likely answer boundaries.

---

```python
answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
```

Extracts the predicted tokens.

---

```python
tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)
```

Converts token IDs back into readable text.

Example:

```
[2023, 2003, 1037, 3231]

↓

"This is a test"
```

---

# Handling Missing Answers

```python
if extracted_string.strip() and start_idx < end_idx:
```

Checks if a valid answer exists.

If yes:

```
Answer extracted from source:

Machine learning is...
```

Otherwise:

```
I couldn't locate a definitive answer.
```

---

# Displaying the Answer

```python
st.markdown(answer)
```

Shows the answer inside the chat.

---

```python
st.session_state.messages.append(...)
```

Stores the assistant's response for future display.

---

# Complete Pipeline Summary

```
User Question
      │
      ▼
Sentence Transformer
      │
      ▼
Question Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top 3 Document Chunks
      │
      ▼
Combine Context
      │
      ▼
Tokenizer
      │
      ▼
BERT Question Answering
      │
      ▼
Predict Start and End Tokens
      │
      ▼
Decode Tokens
      │
      ▼
Display Extracted Answer
```

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Streamlit | Web application interface |
| PyTorch | Deep learning framework |
| LangChain | PDF loading and document splitting |
| Sentence Transformers | Semantic embedding generation |
| Hugging Face Transformers | BERT Question Answering |
| Cosine Similarity | Retrieval of relevant document chunks |
| Local PDFs | Knowledge base |

---

# Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Document Chunking
- Embedding Generation
- Cosine Similarity
- Extractive Question Answering
- Transformer Models
- Streamlit Chat Interface
- Model Caching
- Session State Management

---

# Strengths of the Approach

- Runs entirely on local PDFs.
- No external vector database required.
- Fast semantic retrieval using embeddings.
- Accurate answer extraction with BERT.
- Easy to extend with additional PDFs.
- Uses pretrained models without requiring custom training.

---

# Possible Improvements

- Store embeddings in FAISS or Chroma for faster retrieval on large document collections.
- Display the source document and page number alongside each answer.
- Highlight the extracted answer within the original text.
- Support uploading PDFs directly through the Streamlit interface.
- Use GPU acceleration when available.
- Replace the extractive QA model with a generative Large Language Model (LLM) for more natural responses.
- Implement conversation memory to handle follow-up questions.