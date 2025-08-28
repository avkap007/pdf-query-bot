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

def clean_pdf_text(text):
    """Cleans text by removing common headers, footers, and other artifacts."""
    # Remove headers like "Review #R... Page X"
    text = re.sub(r"Review\s*#R\d+\s*Page\s*\d+", "", text)
    # Remove "REVIEW DECISION" headers
    text = re.sub(r"REVIEW\s+DECISION", "", text)
    # Remove isolated page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_fields(text):
    """Extracts metadata fields using highly specific and robust regex."""
    
    def find(pattern, default=""):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else default

    review_ref = find(r"Review\s+Reference\s*#:\s*([A-Z0-9\-]+)")
    review_date = find(r"\nDate:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})")
    board_date = find(r"Board\s+Decision\s+under\s+Review:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})")
    officer = find(r"Review\s+Officer:\s*([A-Za-z\s,.-]+?)(?:\n|Introduction)")
    penalty = find(r"\$\s?([0-9,]+\.\d{2})")

    # More nuanced boolean logic
    # Penalty is upheld if the officer denies the employer's request to cancel it, or explicitly confirms it.
    penalty_request_denied = bool(re.search(r"deny\s+the\s+employer’s\s+request", text, re.IGNORECASE))
    penalty_confirmed = bool(re.search(r"penalty\s+is\s+(confirmed|upheld)", text, re.IGNORECASE))
    was_penalty_upheld = penalty_request_denied or penalty_confirmed

    # Due diligence is found only if explicitly stated. Check for negatives.
    due_diligence_exercised = bool(re.search(r"employer\s+(exercised|established)\s+due\s+diligence", text, re.IGNORECASE))
    due_diligence_not_exercised = bool(re.search(r"did\s+not\s+exercise\s+due\s+diligence", text, re.IGNORECASE))
    due_diligence_found = due_diligence_exercised and not due_diligence_not_exercised

    repeat_offense = bool(re.search(r"\brepeat\s+(?:offense|violation)", text, re.IGNORECASE))
    
    sections_text = find(r"Sections?\s+Violated:\s*(.*?)(?:Introduction|Summary|Background)")
    sections = re.findall(r'([0-9]+\.[0-9]+(?:\([0-9a-zA-Z]+\))?|[0-9]+\([0-9a-zA-Z]+\))', sections_text) if sections_text else []

    return {
        "review_ref": review_ref,
        "review_date": review_date,
        "board_decision_date": board_date,
        "review_officer": officer.strip().replace("\n", " "),
        "penalty_amount": penalty,
        "was_penalty_upheld": was_penalty_upheld,
        "due_diligence_found": due_diligence_found,
        "repeat_offense": repeat_offense,
        "sections_violated": sorted(list(set(sections)))
    }

def main():
    metadata = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            docs = loader.load()
            
            # Clean the text before metadata extraction and chunking
            full_text = "\n".join([clean_pdf_text(doc.page_content) for doc in docs])
            
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
