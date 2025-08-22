from fastapi import FastAPI
import chromadb
import os

app = FastAPI()

# Initialize ChromaDB with cloud storage path
client = chromadb.PersistentClient(path="/app/chroma_db")

@app.get("/")
def read_root():
    return {"message": "ChromaDB Service Running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "collections": len(client.list_collections())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
