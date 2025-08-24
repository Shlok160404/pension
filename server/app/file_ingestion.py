# server/app/utils/file_ingestion.py

import fitz # PyMuPDF
from typing import List
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from app.chromadb_service import get_or_create_collection, add_documents_to_collection

# A reusable text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# A reusable sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def ingest_pdf_to_chroma(file_path: str, user_id: int):
    """
    Parses a PDF, chunks its text, and ingests the documents into ChromaDB.
    """
    try:
        # 1. Parse the PDF and extract text
        doc = fitz.open(file_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text()

        # 2. Chunk the text
        chunks = text_splitter.split_text(full_text)
        print(f"📄 Extracted {len(chunks)} chunks from the PDF.")

        # 3. Create a ChromaDB collection for this user
        collection = get_or_create_collection(f"user_{user_id}_docs")
        
        # 4. Prepare data for ingestion
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [{"user_id": user_id, "source": file_path, "chunk_index": i} for i, _ in enumerate(chunks)]
        
        # 5. Add documents to the collection
        add_documents_to_collection(collection, chunks, ids, metadatas)
        print(f"🚀 Successfully ingested PDF for user {user_id}.")

    except Exception as e:
        print(f"❌ Failed to ingest PDF: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "success", "message": f"PDF successfully ingested for user {user_id}"}