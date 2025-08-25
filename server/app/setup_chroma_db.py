"""
Ingest FAQs from a CSV into ChromaDB. CSV must contain columns: question, answer
Usage:
  python -m server.app.setup_chroma_db                 # looks for final_faqs_for_chroma.csv in repo root or alongside this file
  python -m server.app.setup_chroma_db path\to\final_faqs_for_chroma.csv  # explicit path
"""

import os
import sys
from uuid import uuid4
from typing import Tuple
import pandas as pd

# Add the 'app' directory to the path so we can import our services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chromadb_service import get_or_create_collection, add_documents_to_collection


def _resolve_csv_path(arg_path: str | None) -> str:
    if arg_path and os.path.isfile(arg_path):
        return arg_path
    # Try repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    candidate_root = os.path.join(repo_root, 'final_faqs_for_chroma.csv')
    if os.path.isfile(candidate_root):
        return candidate_root
    # Try alongside this file
    candidate_local = os.path.join(os.path.dirname(__file__), 'final_faqs_for_chroma.csv')
    if os.path.isfile(candidate_local):
        return candidate_local
    raise FileNotFoundError("final_faqs_for_chroma.csv not found. Provide a path: python -m server.app.setup_chroma_db path\\to\\final_faqs_for_chroma.csv")


def ingest_faqs_to_chroma(csv_path: str | None = None) -> dict:
    print("🚀 Starting FAQ ingestion into ChromaDB...")
    try:
        path = _resolve_csv_path(csv_path)
        df = pd.read_csv(path)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if not {'question', 'answer'}.issubset(df.columns):
            raise ValueError("CSV must contain 'question' and 'answer' columns")
        df = df[['question', 'answer']].dropna(how='any')
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()
        df = df[(df['question'] != '') & (df['answer'] != '')]

        if df.empty:
            raise ValueError('No valid question/answer rows found in CSV')

        faq_collection = get_or_create_collection(name="faq_collection")

        documents = df['question'].tolist()
        metadatas = [{"question": q, "answer": a} for q, a in zip(df['question'], df['answer'])]
        ids = [str(uuid4()) for _ in documents]

        add_documents_to_collection(faq_collection, documents, ids, metadatas)
        print(f"✅ FAQ ingestion complete. Added {len(documents)} rows from '{path}'.")
        return {"status": "success", "message": f"FAQ collection populated with {len(documents)} rows."}
    except Exception as e:
        print(f"❌ An error occurred during FAQ ingestion: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    ingest_faqs_to_chroma(csv_arg)