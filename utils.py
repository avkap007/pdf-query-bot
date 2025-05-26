from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

class PSAQABot:
    def __init__(self, index_path="index_store"):
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            retriever=self.retriever
        )

    def ask(self, question: str):
        return self.chain({"question": question}, return_only_outputs=True)

    def get_top_docs(self, question: str, k: int = 5):
        return self.vectorstore.similarity_search(question, k=k)

    def format_metadata(self, metadata: dict):
        fields = [
            ("Review Ref", metadata.get("review_ref")),
            ("Review Officer", metadata.get("review_officer")),
            ("Review Date", metadata.get("review_date")),
            ("Board Decision Date", metadata.get("board_decision_date")),
            ("Penalty Amount", metadata.get("penalty_amount")),
            ("Was Penalty Upheld?", "✅ Yes" if metadata.get("was_penalty_upheld") else "❌ No"),
            ("Due Diligence Found?", "✅ Yes" if metadata.get("due_diligence_found") else "❌ No"),
            ("Repeat Offense", "✅ Yes" if metadata.get("repeat_offense") else "❌ No"),
            ("Sections Violated", ", ".join(metadata.get("sections_violated", [])))
        ]
        return "\n".join([f"**{label}:** {value or '—'}" for label, value in fields])
