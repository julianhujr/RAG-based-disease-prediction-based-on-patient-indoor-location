
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from tqdm import tqdm
from src.embed import AutoEmbedding
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS

def manage_rag_db(faiss_path, docs_path, model_name=None, embedding_type=None, chunk_size=None, chunk_overlap=None):
    """
    Manage RAG database: initialize if index does not exist, extend if it does.

    Parameters:
        faiss_path (str): Path to save the FAISS DB
        docs_path (str): Path where documents are located
        embedding_type (str, optional): Type of embedding model
        model_name (str, optional): Name of the embedding model, required during initialization
        chunk_size (int, optional): Size of document chunks, required during initialization
        chunk_overlap (int, optional): Size of chunk overlap, required during initialization
    """
    # Check if FAISS index exists
    index_file = os.path.join(faiss_path, "index.faiss")
    if not os.path.exists(index_file):
        # Initialization logic
        if model_name is None or embedding_type is None or chunk_size is None or chunk_overlap is None:
            raise ValueError("model_name, embedding_type, chunk_size, and chunk_overlap must be provided to initialize a new index")
        
        # Create directory if it does not exist
        if not os.path.exists(faiss_path):
            os.makedirs(faiss_path)
        
        # Read all .md files in docs_path
        docs = []
        metadata = []
        filenames = []
        for filename in os.listdir(docs_path):
            if filename.endswith('.md'):
                filenames.append(filename)
                file_path = os.path.join(docs_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    docs.append(content)
                    metadata.append({"source": filename})
        
        # Split documents
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.create_documents(docs, metadata)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Config:")
        print(f"Using device: {device}")
        print(f"Embedding model: {model_name}")
        print(f"Embedding type: {embedding_type}")
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {chunk_overlap}")
        
        # Create FAISS index
        db = FAISS.from_documents(
            chunked_docs,
            AutoEmbedding(model_name, embedding_type, model_kwargs={"device": device})
        )
        
        # Save index
        print(f"Saving the FAISS index...")
        db.save_local(faiss_path)
        print(f"Embedded documents: {filenames}")
        print(f"Index saved to {faiss_path}")
        
        # Save embedded filenames
        with open(os.path.join(faiss_path, 'embedded_docs.json'), 'w') as json_file:
            json.dump(filenames, json_file, indent=4)
        
        # Save configuration
        config = {
            "model_name": model_name,
            "embedding_type": embedding_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        with open(os.path.join(faiss_path, 'config.json'), 'w') as json_file:
            json.dump(config, json_file, indent=4)
        
        print(f"Initialization complete.")
        print(f"Config saved to {os.path.join(faiss_path, 'config.json')}")
    
    else:
        # Extension logic
        # Load configuration
        with open(os.path.join(faiss_path, 'config.json'), 'r') as json_file:
            config = json.load(json_file)

        # Check for consistency
        if (model_name is not None and model_name != config['model_name']) or \
           (chunk_size is not None and chunk_size != config['chunk_size']) or \
           (chunk_overlap is not None and chunk_overlap != config['chunk_overlap']):
            print("Current embedding configuration does not match the existing RAG DB config:")
            print("Using the existing config...")

        # Update local variables with existing config
        model_name = config['model_name']
        embedding_type = config['embedding_type']
        chunk_size = config['chunk_size']
        chunk_overlap = config['chunk_overlap']

        # Load embedded filenames
        with open(os.path.join(faiss_path, 'embedded_docs.json'), 'r') as json_file:
            embedded_docs = json.load(json_file)
        
        # Find new documents
        new_docs = []
        new_metadatas = []
        new_filenames = []
        for filename in os.listdir(docs_path):
            if filename.endswith('.md') and filename not in embedded_docs:
                new_filenames.append(filename)
                file_path = os.path.join(docs_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    new_docs.append(content)
                    new_metadatas.append({"source": filename})
        
        if not new_filenames:
            print("No new documents to embed")
            return
        
        # Split new documents
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.create_documents(new_docs, new_metadatas)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Config:")
        print(f"Using device: {device}")
        print(f"Embedding model: {model_name}")
        print(f"Embedding type: {embedding_type}")
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {chunk_overlap}")
        
        # Load existing FAISS index
        index = FAISS.load_local(
            faiss_path,
            AutoEmbedding(model_name, embedding_type, model_kwargs={"device": device}),
            allow_dangerous_deserialization=True
        )
        
        # Add new documents to index
        index.add_documents(chunked_docs)
        
        # Save updated index
        print(f"Saving the FAISS index...")
        index.save_local(faiss_path)
        print(f"Embedded documents: {new_filenames}")
        print(f"Updated FAISS index saved to {faiss_path}")
        
        # Update the list of embedded filenames
        embedded_docs.extend(new_filenames)
        with open(os.path.join(faiss_path, 'embedded_docs.json'), 'w') as json_file:
            json.dump(embedded_docs, json_file, indent=4)

# Example usage
if __name__ == "__main__":
    with open('./config/config.json', "r") as f:
        config = json.load(f)
    manage_rag_db(**config['rag_db'])
