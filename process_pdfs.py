# process_pdfs.py

import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

PDF_FOLDER = "pdfs_2025"
INDEX_FOLDER = "index_store"
METADATA_FILE = "metadata.json"

# Load metadata into a dict keyed by filename
with open(METADATA_FILE, "r") as f:
    all_metadata = {entry["filename"]: entry for entry in json.load(f)}

# Setup
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(PDF_FOLDER):
    if not filename.endswith(".pdf"):
        continue

    print(f"📄 Processing: {filename}")
    loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
    pages = loader.load()
    chunks = splitter.split_documents(pages)

    file_meta = all_metadata.get(filename, {})  # default to empty dict if missing

    for chunk in chunks:
        chunk.metadata.update({
            "source": filename,
            "review_ref": file_meta.get("review_ref", ""),
            "review_date": file_meta.get("review_date", ""),
            "board_decision_date": file_meta.get("board_decision_date", ""),
            "review_officer": file_meta.get("review_officer", ""),
            "penalty_amount": file_meta.get("penalty_amount", ""),
            "was_penalty_upheld": file_meta.get("was_penalty_upheld", False),
            "due_diligence_found": file_meta.get("due_diligence_found", False),
            "repeat_offense": file_meta.get("repeat_offense", False),
            "sections_violated": file_meta.get("sections_violated", []),
        })
        all_chunks.append(chunk)

print(f"🔍 Total chunks: {len(all_chunks)}")

# Embed and store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_chunks, embedding)
vectorstore.save_local(INDEX_FOLDER)

print(f"✅ Embedding complete. Index saved to: {INDEX_FOLDER}")
