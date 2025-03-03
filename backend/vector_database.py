import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from text_splitter import process_pdfs  # Import the function to process PDFs
from text_splitter import *

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "research-papers-index"

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist. Please create it first.")

index = pc.Index(PINECONE_INDEX_NAME)

def embed_text(text):
    """Generate embeddings using OpenAI's model."""
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def store_chunks_in_vector_db(directory="papers"):
    """Process PDFs and store chunks in the Pinecone vector database with filenames."""
    all_chunks = []  # List to store all chunks with metadata
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if not text:
                print(f"Skipping {filename}, no text extracted.")
                continue
            
            chunks = split_text(text)
            
            # Store chunks along with their filenames
            for i, chunk in enumerate(chunks):
                all_chunks.append((chunk, filename))
            
            print(f"Extracted {len(chunks)} chunks from {filename}")

    # Store in Pinecone
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        upsert_data = [
            (f"chunk-{i+j}", embed_text(chunk[0]), {"text": chunk[0], "filename": chunk[1]})
            for j, chunk in enumerate(batch)
        ]
        index.upsert(upsert_data)
        print(f"Stored {len(batch)} chunks in vector DB.")

if __name__ == "__main__":
    print("Processing PDFs and storing chunks in vector database...")
    store_chunks_in_vector_db("papers")
    print("All chunks have been stored in the vector database.")

