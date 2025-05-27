import streamlit as st
from dotenv import load_dotenv
from utils import PSAQABot

load_dotenv()

import os
print("KEY FOUND:", os.getenv("OPENAI_API_KEY")[:5])  # Print first few chars only

bot = PSAQABot()

st.title("📄 PS Assist – PDF Q&A")
st.markdown("Ask a question about the 2025 decision letters:")

# 🧪 Prompt Suggestions
with st.expander("💡 Try asking one of these..."):
    st.markdown("- *What was the final penalty in R0325542?*")
    st.markdown("- *Which decisions cite section 18.3.1(3)?*")
    st.markdown("- *Was due diligence found in crane-related incidents?*")
    st.markdown("- *What criteria did the Board consider for penalties?*")

query = st.text_input("🔍 Your question")

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

def format_answer(text: str) -> str:
    """Format the LLM answer with line breaks for readability."""
    # Insert double line breaks after full stops if it's long
    paragraphs = text.strip().split(". ")
    spaced = [p.strip().capitalize() for p in paragraphs if p]
    return "\n\n".join(f"{line}." for line in spaced)

if query:
    with st.spinner("Thinking..."):
        result = bot.ask(query)
        st.session_state.doc_chunks = result.get("top_docs", []) # store for follow-ups

    # Display the answer within the green box
    st.success(f"🧠 Answer:\n\n{format_answer(result['answer'])}")

    if result.get("sources", "").startswith("metadata.json"):
        st.info("This answer was retrieved directly from the metadata (not the LLM).")

    # Display the most relevant document prominently
    top_docs = result.get("top_docs", [])
    if top_docs:
        most_relevant_doc = top_docs[0]
        st.markdown(f"### 📄 Most Relevant Document – Review Ref: {most_relevant_doc.metadata.get('review_ref', 'Unknown')}")
        st.markdown(bot.format_metadata(most_relevant_doc.metadata))
        st.markdown("---")

        # Display remaining documents as secondary results
        if len(top_docs) > 1:
            st.markdown("### 📑 Other Relevant Documents")
            for i, doc in enumerate(top_docs[1:]):
                meta = bot.format_metadata(doc.metadata)
                with st.expander(f"Result {i+2} – {doc.metadata.get('source', 'Unknown')}"):
                    st.markdown(meta)
                    st.markdown("---")
                    st.write(doc.page_content[:200] + "...")

        # Keep the follow-up logic associated with the most relevant doc for now
        if top_docs:
             # Follow-up input for the most relevant doc
            st.markdown("---") # Separator before follow-up for clarity
            followup_input_main = st.text_input(f"Ask follow-up about the Most Relevant Document (Review Ref: {most_relevant_doc.metadata.get('review_ref', 'Unknown')})", key="followup_input_main")
            if followup_input_main:
                # Use the new method to ask about the specific document
                followup_answer = bot.ask_about_document(followup_input_main, most_relevant_doc.metadata.get('source', ''))
                # formatted_response = format_answer(followup["answer"])
                st.markdown("🧠 **Detailed Response:**")
                st.markdown(f"{followup_answer}") # Directly use the answer string

