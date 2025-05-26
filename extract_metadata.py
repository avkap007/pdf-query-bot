# extract_metadata.py

import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = "pdfs_2025"
OUTPUT_FILE = "metadata.json"

def extract_fields(text):
    def find(pattern, default=""):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    return {
        "review_ref": find(r"Review Reference Number[:\s]+([A-Z0-9\-]+)"),
        "review_date": find(r"Review Decision Date[:\s]+([\d\-]+)"),
        "board_decision_date": find(r"Board Decision.*Date[:\s]+([\d\-]+)"),
        "review_officer": find(r"Review Officer[:\s]+([A-Za-z\s,.\-]+)"),
        "penalty_amount": find(r"\$\s?([0-9,]+\.\d{2})"),
        "was_penalty_upheld": bool(re.search(r"\b(confirm|uphold|maintain)\b.*penalt", text, re.IGNORECASE)),
        "due_diligence_found": bool(re.search(r"(found|determined).*due diligence.*(exercised|shown)", text, re.IGNORECASE)),
        "repeat_offense": bool(re.search(r"\brepeat (offense|violation)\b", text, re.IGNORECASE)),
        "sections_violated": re.findall(r"section[s]?\s+([0-9]+\(?[0-9a-zA-Z\/]*\)?)", text, re.IGNORECASE)
    }

def main():
    metadata = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            docs = loader.load()
            full_text = "\n".join([doc.page_content for doc in docs])
            fields = extract_fields(full_text)
            fields["filename"] = filename
            metadata.append(fields)
            print(f"✅ Processed: {filename}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n📦 Metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
