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
        top_docs = bot.get_top_docs(query)
        st.session_state.doc_chunks = top_docs  # store for follow-ups

    st.success("🧠 Answer:")
    st.markdown(format_answer(result["answer"]))

    st.markdown("### 📄 Relevant Documents")
    for i, doc in enumerate(top_docs):
        meta = bot.format_metadata(doc.metadata)
        with st.expander(f"Result {i+1} – {doc.metadata.get('source', 'Unknown')}"):
            st.markdown(meta)
            st.markdown("---")
            st.write(doc.page_content[:500] + "...")
            if st.button(f"Tell me more about Result {i+1}", key=f"more_{i}"):
                followup = bot.ask(f"Tell me more about this decision:\n\n{doc.page_content}")
                st.markdown(f"🧠 **Detailed Response:**\n\n{format_answer(followup['answer'])}")

            # Follow-up input
            followup_input = st.text_input(f"Ask follow-up about Result {i+1}", key=f"followup_input_{i}")
            # if followup_input:
            #     followup = bot.ask(f"{followup_input}\n\nContext:\n{doc.page_content}")
            #     st.markdown(f"🧠 **Detailed Response:**\n\n{format_answer(followup['answer'])}")
            if followup_input:
                followup = bot.ask(f"{followup_input}\n\nContext:\n{doc.page_content}")
                formatted_response = format_answer(followup["answer"])
                st.markdown("🧠 **Detailed Response:**")
                st.markdown(f"```\n{formatted_response}\n```")

