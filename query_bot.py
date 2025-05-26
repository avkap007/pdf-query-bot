import os
from dotenv import load_dotenv
load_dotenv()# import os
# os.environ["OPENAI_API_KEY"] = "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"  # Replace with your real key
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

INDEX_FOLDER = "index_store"

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore.as_retriever()
)

while True:
    query = input("\nAsk a question about the 2025 PDFs: ")
    if query.lower() in {"exit", "quit"}:
        break
    result = qa.run(query)
    print(f"\n🧠 Answer: {result}")
