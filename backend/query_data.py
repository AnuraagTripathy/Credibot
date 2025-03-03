import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

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

# Define prompt template with specific instructions for response format
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based solely on the above context: {query}

IMPORTANT INSTRUCTIONS:

1. Provide a detailed and thorough answer based only on the information in the context.
2. Start your answer with "According to this research paper," or "According to these research papers," as appropriate.
3. Do NOT mention document filenames like "paper_1.pdf" or any local file identifiers.
4. Summarize the key findings, methodologies, and conclusions from the research in detail.
5. If the context contains multiple studies with different conclusions, explain the different perspectives.
6. If the context is insufficient to answer the question fully, ignore the information which is not related to the query.
7. If there is a direct answer to it, quote the text from the research paper.
"""

def embed_text(text):
    """Generate embeddings using OpenAI's model."""
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def query_database_and_generate_response(query_text, top_k=5, model="gpt-3.5-turbo"):
    """
    Query the Pinecone vector database for similar text chunks,
    build a context-based prompt, and generate a detailed response using OpenAI.
    """
    # Get query embedding and search Pinecone
    query_embedding = embed_text(query_text)
    
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Store document names and scores in a list of tuples
    source_documents = []
    
    # Collect the text chunks to build the context
    context_chunks = []
    for match in query_response["matches"]:
        # Get the filename and score
        filename = match["metadata"].get("filename", "Unknown Document")
        score = match["score"]
        text_chunk = match["metadata"]["text"]
        
        # Store the document info
        source_documents.append((filename, score))
        
        # Add the text to context chunks
        context_chunks.append(text_chunk)
    
    # Join all chunks to create the context
    context = "\n\n".join(context_chunks)
    
    # Format the prompt template with the context and query
    formatted_prompt = PROMPT_TEMPLATE.format(context=context, query=query_text)
    
    # Generate response using OpenAI
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a research assistant that provides detailed, accurate summaries of scientific papers. You answer questions based only on the provided context, without adding external information or opinions."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.3,  # Lower temperature for more focused responses
        max_tokens=800    # Allow for longer, more detailed responses
    )
    
    # Return both the generated response and the source documents
    return response.choices[0].message.content, source_documents

if __name__ == "__main__":
    query_text = "Can cannabis help with depression?"
    top_k = 5  # Increased to get more context
    model = "gpt-3.5-turbo"  # Using GPT-3.5-Turbo which should be available
    
    # Generate and print the response
    response, source_documents = query_database_and_generate_response(query_text, top_k, model)
    
    print(f"Query: {query_text}\n")
    print("Response:")
    print(response)