from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import json
import re
import os
import openai

class PSAQABot:
    def __init__(self, index_path="index_store", metadata_path="metadata.json", pdf_folder="pdfs_2025"):
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

        # Custom prompt to encourage longer answers
        custom_template = """Use the following context from retrieved documents to answer the user's question.
If the context contains information relevant to the question, provide an answer of 150-200 words based on that information.
If you don't know the answer based on the context, just say that you don't know.

{page_content}

Question: {question}
Answer:"""
        CUSTOM_PROMPT = PromptTemplate(template=custom_template, input_variables=["page_content", "question"])

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(temperature=0, max_tokens=700),
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": CUSTOM_PROMPT,
                "document_variable_name": "page_content"
            }
        )
        # Load all metadata for direct lookup
        with open(metadata_path, "r") as f:
            self.all_metadata = json.load(f)
        
        self.pdf_folder = pdf_folder

    def ask(self, question: str):
        # Try direct penalty lookup first
        penalty_result = self.try_penalty_lookup(question)
        if penalty_result:
            return penalty_result
        # Fallback to LLM
        llm_response = self.chain({"question": question}, return_only_outputs=True)
        top_docs = self.get_top_docs(question)

        return {
            "answer": llm_response["answer"],
            "sources": llm_response.get("sources"),
            "top_docs": top_docs
        }

    def try_penalty_lookup(self, question: str):
        # Detect if question is about penalty for a specific review/file
        penalty_patterns = [
            r"final penalty.*(r\d{7,})",
            r"penalty amount.*(r\d{7,})",
            r"penalty.*in (r\d{7,})",
            r"penalty.*for (r\d{7,})",
            r"(r\d{7,}).*penalty"
        ]
        for pat in penalty_patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                ref = m.group(1).lower()
                # Try to match by filename
                for entry in self.all_metadata:
                    fname = entry.get("filename", "").lower()
                    if ref in fname:
                        amt = entry.get("penalty_amount")
                        if amt:
                            top_docs = self.get_top_docs(question)
                            return {
                                "answer": f"The final penalty in {ref.upper()} was ${amt}. (from metadata)",
                                "sources": "metadata.json (direct lookup)",
                                "top_docs": top_docs
                            }
                # Try to match by review_ref if available
                for entry in self.all_metadata:
                    if entry.get("review_ref", "").lower() == ref:
                        amt = entry.get("penalty_amount")
                        if amt:
                            top_docs = self.get_top_docs(question)
                            return {
                                "answer": f"The final penalty in {ref.upper()} was ${amt}. (from metadata)",
                                "sources": "metadata.json (direct lookup)",
                                "top_docs": top_docs
                            }
        return None

    def ask_about_document(self, question: str, document_path: str):
        """Answer a question based on the full text of a specific document."""
        full_doc_path = os.path.join(self.pdf_folder, document_path)
        try:
            loader = PyPDFLoader(full_doc_path)
            docs = loader.load()
            full_text = "\n".join([doc.page_content for doc in docs])

            # Use LLM directly with the full document text
            client = openai.OpenAI()
            prompt = (
                "Based on the following document text, answer the user's question.\n\n"
                f"Document: {document_path}\n\n"
                f"Document Text: {full_text[:6000]}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error asking about document {document_path}: {e}")
            return f"Sorry, I couldn't process the request for document {document_path}."

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
        # Format each field on a new line using Markdown bold for labels
        formatted_lines = []
        for label, value in fields:
            formatted_lines.append(f"**{label}:** {value or '—'}")

        # Include the summary field with a clear label
        summary = metadata.get("summary")
        if summary:
            # Use double newline for better spacing in Streamlit
            formatted_lines.append(f"\n**Summary:** {summary}")

        # Join all lines with double newlines for clear separation
        return "\n\n".join(formatted_lines)
