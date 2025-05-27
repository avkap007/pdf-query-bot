# extract_metadata.py

import os
import re
import json
import openai
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

PDF_FOLDER = "pdfs_2025"
OUTPUT_FILE = "metadata.json"

# Helper to robustly extract fields

def extract_fields(text):
    def find(pattern, default=""):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default

    # Section extraction (comma-separated list)
    sections = []
    sec_match = re.search(r"sections? ([\d.,()a-zA-Z ]+) of the Occupational Health and Safety Regulation", text)
    if sec_match:
        raw = sec_match.group(1)
        sections = [s.strip() for s in re.split(r",| and ", raw) if s.strip()]
    else:
        # fallback: find all 'section X' mentions
        sections = re.findall(r"section[s]? ([\d.()a-zA-Z/]+)", text, re.IGNORECASE)

    return {
        "review_ref": find(r"Review Reference #[:\s]+(R\d{7,})"),
        "review_date": find(r"Date: +([A-Za-z]+ \d{1,2}, \d{4})"),
        "board_decision_date": find(r"Board Decision under Review: ([A-Za-z]+ \d{1,2}, \d{4})"),
        "review_officer": find(r"Review Officer: ([A-Za-z .'-]+)"),
        "penalty_amount": extract_final_penalty(text),
        "was_penalty_upheld": is_penalty_upheld(text),
        "due_diligence_found": bool(re.search(r"(found|determined).*due diligence.*(exercised|shown)", text, re.IGNORECASE)),
        "repeat_offense": bool(re.search(r"\brepeat (offense|violation)\b", text, re.IGNORECASE)),
        "sections_violated": sections
    }

# Helper to extract final penalty amount

def extract_final_penalty(text):
    # Look for patterns like "final penalty amount should be $\d" in the last part of the document
    conclusion_match = re.search(r"(in summary|final conclusion|decision)(.*?)\\n\\n", text, re.IGNORECASE | re.DOTALL)
    search_text = text # Default to full text if no clear conclusion section found

    if conclusion_match:
        search_text = conclusion_match.group(2) # Search within the conclusion section

    penalty_match = re.search(r"final penalty(?: amount)?(?: should be)?\s?\$([0-9,]+\.\d{2})", search_text, re.IGNORECASE)
    if penalty_match:
        return penalty_match.group(1)

    # Fallback: look for any penalty amount mentioned towards the end of the document
    # This is less reliable but better than nothing if the specific phrase isn't found
    end_of_doc_search = text[-1000:] # Look in the last 1000 characters
    penalty_match_fallback = re.search(r"\$([0-9,]+\.\d{2})[^\n.]*", end_of_doc_search, re.IGNORECASE)
    if penalty_match_fallback:
        return penalty_match_fallback.group(1)

    return ""

# Helper to determine if penalty was upheld

def is_penalty_upheld(text):
    # Look for terms in the conclusion section indicating the outcome of the review
    conclusion_match = re.search(r"(in summary|final conclusion|decision)(.*?)\\n\\n", text, re.IGNORECASE | re.DOTALL)
    search_text = text # Default to full text if no clear conclusion section found

    if conclusion_match:
        search_text = conclusion_match.group(2) # Search within the conclusion section

    # Positive indicators (case-insensitive)
    uphold_patterns = [
        r"(confirm|uphold|maintain)(?:s|ed)? .*penalt",
        r"penalty .* upheld",
        r"decision .* confirmed",
        r"decision .* maintained"
    ]
    for pattern in uphold_patterns:
        if re.search(pattern, search_text, re.IGNORECASE):
            return True

    # Negative indicators (case-insensitive) - if any of these are found in conclusion, it's NOT upheld
    # Note: This assumes if it's not explicitly upheld, it was likely varied or rescinded in a way that counts as not upheld.
    rescind_patterns = [
        r"(rescind|vary|cancel)(?:s|ed)? .*penalt",
        r"penalty .* varied",
        r"penalty .* cancelled",
        r"decision .* rescinded"
    ]
    for pattern in rescind_patterns:
        if re.search(pattern, search_text, re.IGNORECASE):
            return False

    # If no clear indicators in conclusion, fallback to searching entire text for general uphold terms
    # This is less reliable, but better than nothing.
    if re.search(r"\b(confirm|uphold|maintain)\b.*penalt", text, re.IGNORECASE):
         return True

    return False # Default to False if no clear uphold indicator is found

# LLM summary helper

def get_llm_summary(text, fields):
    # Include extracted fields in the prompt
    field_details = "\n".join([f"{key.replace('_', ' ').title()}: {value}" for key, value in fields.items() if key != 'filename' and value])
    
    prompt = (
        "Based on the provided review decision letter and the following extracted key details:\n\n"
        f"{field_details}\n\n"
        "Summarize the key aspects of the review, focusing on the Issue, Reasons, and Decision sections. "
        "INCORPORATE THE PROVIDED KEY DETAILS into your summary where relevant. "
        "State clearly whether the original penalty was upheld, varied, or rescinded. "
        "The summary should be 5-7 sentences and capture the most important findings and the ultimate outcome of the review.\n\n"
        "Review Decision Letter Text (first 6000 chars):\n"
        + text[:6000]  # Truncate for token limit
    )
    try:
        client = openai.OpenAI()  # This uses your OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM summary failed: {e}")
        return None

# Heuristic summary fallback

def get_heuristic_summary(text):
    # Try to find the first paragraph after 'Introduction and Background'
    intro = re.split(r"introduction and background", text, flags=re.IGNORECASE)
    if len(intro) > 1:
        after_intro = intro[1].strip()
        paras = [p.strip() for p in after_intro.split("\n\n") if p.strip()]
        if paras:
            return paras[0][:1000]  # limit length
    # fallback: first 3 sentences
    sentences = re.split(r"(?<=[.!?]) +", text)
    return " ".join(sentences[:3])

def main():
    metadata = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            docs = loader.load()
            full_text = "\n".join([doc.page_content for doc in docs])
            fields = extract_fields(full_text)
            fields["filename"] = filename
            # LLM summary (fallback to heuristic)
            summary = get_llm_summary(full_text, fields)
            if not summary:
                summary = get_heuristic_summary(full_text)
            fields["summary"] = summary
            metadata.append(fields)
            print(f"✅ Processed: {filename}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n📦 Metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
