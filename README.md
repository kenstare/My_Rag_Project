**Project README**

- **Overview:** This repository contains a Retrieval-Augmented Generation (RAG) demo and a small personal knowledge corpus for "Kehinde Akindele". The project uses local PDF documents, OpenAI embeddings and LLMs for retrieval and question-answering, and a Chroma vector store persisted in the chroma_db folder.

# Retrieval-Augmented Generation (RAG) System for Kehinde Akindele
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, OpenAI embeddings, and a Chroma vector store. The system is designed to answer questions about Kehinde Akindele based on a small personal knowledge corpus stored in PDF documents.

# The key components of the system include:
1. Document Loading and Processing: PDF documents are loaded and split into manageable chunks for embedding.
2. Embedding Generation: OpenAI's embedding model is used to convert text chunks into vector representations.
3. Vector Store: A Chroma vector store is used to store and retrieve embeddings efficiently.
4. RAG Chain: A retrieval-augmented generation chain is set up to answer questions based on retrieved documents.


  - **AI & ML Projects:** documents/AI & ML Projects.pdf was created to include three requested project descriptions (world population forecasting; predicting food prices in Nigeria; Indian credit-card/car price prediction) and cloud computing notes (EC2, S3, VPC basics).

  - **Professional Resume:** documents/Professional Resume.pdf was regenerated as a text-based PDF with the name "Kehinde Akindele" and added contact details: akindelekehinde250@gmail.com and https://github.com/kenstare?tab=repositories.

  - **Personal Biography:** documents/Personal Biography.pdf was regenerated with the biography text (Art student, Graphic Design, AI engineering & Cloud Computing focus, from Ekiti State, Nigeria).

- **Key files (current important files):**
  - `rag_system.py` — RAG indexing and small RAG demo chain (uses OpenAI embeddings and ChatOpenAI).
  - rag_system.ipynb
  - documents/Professional Resume.pdf`
  - documents/Personal Biography.pdf
  - documents/AI & ML Projects.pdf
  - chroma_db/ — persisted Chroma database (collection name used: my_info_collection).

- **Environment & Dependencies:**
  - Python 3.10+ recommended (the environment in which the work was performed used Python 3.13).
  - Install dependencies listed in requirements.txt

  - The project reads OPENAI_API_KEY from a .env file (loaded via python-dotenv) 

- **How to run indexing (PowerShell examples):**

1) Ensure the API key is available. Create a .env file in the project root with:

```
OPENAI_API_KEY=sk-...YOUR_KEY_HERE...
```


2) From the project root run:

```powershell
python .\rag_system.py
```

This will:
- Load PDF pages from documents/*.pdf.
- Split them into chunks using the configured text splitter.
- Create embeddings via OpenAI (text-embedding-3-small).

- Store the embeddings and metadata in a Chroma vector store persisted to ./chroma_db.

```python
# inside a Python REPL or script after creating the vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("Who is Kehinde Akindele?")
for r in results:
    print(r.metadata.get('source'), r.page_content[:200])
```

