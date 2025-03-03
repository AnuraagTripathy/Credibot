import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks using a recursive character text splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks

def process_pdfs(directory="papers"):
    """Find PDF files, extract content, split into chunks, and return all chunks."""
    all_chunks = []  # List to store all chunks from all PDFs
    
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
            all_chunks.extend(chunks)  # Add chunks of the current PDF to the list
            print(f"Extracted {len(chunks)} chunks from {filename}")
            
            # Print some chunks for verification
            print("Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
                print(f"Chunk {i+1}:\n{chunk}\n---")
    
    return all_chunks  # Return all chunks from all PDFs

if __name__ == "__main__":
    chunks = process_pdfs()
    print(f"Total chunks extracted: {len(chunks)}")
