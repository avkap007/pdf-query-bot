import os
from dotenv import load_dotenv
load_dotenv()# process_pdfs.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

PDF_FOLDER = "pdfs_2025"
INDEX_FOLDER = "index_store"

# Load and split all PDFs
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        print(f"📄 Processing: {filename}")
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

print(f"🔍 Total chunks: {len(all_chunks)}")

# Embed and store
embedding = OpenAIEmbeddings()  # Requires OPENAI_API_KEY set
vectorstore = FAISS.from_documents(all_chunks, embedding)
vectorstore.save_local(INDEX_FOLDER)

print("✅ Embedding complete. Index saved to:", INDEX_FOLDER)
