from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class PSAQABot:
    def __init__(self, index_path="index_store", metadata_path="metadata.json", pdf_folder="pdfs_2025"):
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

        system_prompt = (
            "You are an expert assistant for WorkSafeBC review officers. "
            "Your task is to provide detailed, factual answers based on "
            "the provided context from decision letters. When asked a "
            "question, synthesize the information from the relevant text "
            "and present it clearly. If the context is insufficient, "
            "state that the answer is not available in the provided documents."
            "\n\n"
            "CONTEXT:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "QUESTION: {input}"),
            ]
        )

        # Using a more powerful model and increasing temperature for more detailed answers
        llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, question: str):
        response = self.chain.invoke({"input": question})
        return response

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
        return "\n".join([f"- **{label}:** {value or '—'}" for label, value in fields])
